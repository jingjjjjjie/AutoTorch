"""Run final-layer Grad-CAM++ as batch-one shards across multiple GPU slots."""

from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from advanced_visualization.core.model_router import ModelRoute, registered_model_routes
from advanced_visualization.core.registered_gradcam import initialize_artifact


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path("/mnt4/advanced_visualization")
COMPOSE_FILE = REPO_ROOT / "advanced_visualization" / "docker-compose.yml"
STATE_PATH = ARTIFACT_ROOT / "gradcam_sharded_state.json"
LOG_ROOT = ARTIFACT_ROOT / "sharded_logs"


@dataclass(frozen=True)
class Task:
    model_id: str
    layer: str
    framework: str
    shard_index: int


@dataclass(frozen=True)
class Slot:
    gpu: int
    index: int

    @property
    def key(self) -> str:
        return f"gpu{self.gpu}:slot{self.index}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu-slots",
        action="append",
        required=True,
        metavar="GPU:COUNT",
    )
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--allocator-limit-mib", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--retry-delay", type=int, default=60)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing overlays. By default only missing overlays are made.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete selected models' existing overlays before starting.",
    )
    return parser.parse_args()


def _slots(values: list[str]) -> list[Slot]:
    result: list[Slot] = []
    for value in values:
        gpu_text, count_text = value.split(":", 1)
        gpu = int(gpu_text)
        count = int(count_text)
        if count < 1:
            raise ValueError(f"Invalid GPU slot count: {value}")
        result.extend(Slot(gpu, index) for index in range(count))
    return result


def _gpu_uuid(gpu: int) -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()[0]


def _clean(routes: list[ModelRoute]) -> None:
    commands = []
    for route in routes:
        artifact_dir = shlex.quote(str(route.artifact_dir))
        commands.append(
            f"find {artifact_dir} -type f "
            "\\( -name '*.webp' -o -name 'gradcam_generation_state*.json' "
            "-o -name 'gradcam_run_state.json' \\) -delete"
        )
    command = " && ".join(commands)
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "run",
            "--rm",
            "--no-deps",
            "--entrypoint",
            "/bin/sh",
            "tensorflow-vansmall-live",
            "-c",
            command,
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    remaining = [
        path
        for route in routes
        for path in route.artifact_dir.glob("gradcam/**/*.webp")
    ]
    if remaining:
        raise RuntimeError(f"Cleanup left {len(remaining)} WebP artifacts.")


def _state_path(task: Task, num_shards: int) -> Path:
    return (
        ARTIFACT_ROOT
        / task.model_id
        / (
            f"gradcam_generation_state.{task.layer}."
            f"shard-{task.shard_index:03d}-of-{num_shards:03d}.json"
        )
    )


def _command(
    task: Task,
    route: ModelRoute,
    slot: Slot,
    num_shards: int,
    allocator_limit_mib: int,
    batch_size: int,
    overwrite: bool,
) -> tuple[list[str], dict[str, str]]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(slot.gpu),
            "AUTOTORCH_GRADCAM_DEVICE": "cuda:0",
            "AUTOTORCH_CUDA_MEMORY_LIMIT_MB": str(allocator_limit_mib),
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
        }
    )
    shard_args = [
        "--num-shards",
        str(num_shards),
        "--shard-index",
        str(task.shard_index),
    ]
    if overwrite:
        shard_args.append("--overwrite")
    if task.framework == "pytorch":
        module = "advanced_visualization.cli.generate_registered_gradcam"
        batch_args = []
        if batch_size > 1:
            module = "advanced_visualization.cli.generate_registered_gradcam_batched"
            batch_args = ["--batch-size", str(batch_size)]
        return (
            [
                sys.executable,
                "-m",
                module,
                task.model_id,
                "--layer",
                task.layer,
                *batch_args,
                *shard_args,
            ],
            environment,
        )
    if task.framework == "tensorflow":
        if batch_size != 1:
            raise ValueError("TensorFlow sharded batching is not implemented.")
        gpu_uuid = _gpu_uuid(slot.gpu)
        return (
            [
                "docker",
                "compose",
                "-f",
                str(COMPOSE_FILE),
                "run",
                "--rm",
                "--no-deps",
                "-e",
                "CUDA_DEVICE_ORDER=PCI_BUS_ID",
                "-e",
                "CUDA_VISIBLE_DEVICES=0",
                "-e",
                f"NVIDIA_VISIBLE_DEVICES={gpu_uuid}",
                "-e",
                f"VANSMALL_GPU_MEMORY_LIMIT_MB={allocator_limit_mib}",
                "tensorflow-vansmall-live",
                "python",
                "-m",
                "advanced_visualization.tensorflow_service.batch",
                task.model_id,
                "--layer",
                task.layer,
                *shard_args,
            ],
            environment,
        )
    raise ValueError(f"Unsupported framework: {task.framework}")


def _atomic_state(payload: dict) -> None:
    temporary = STATE_PATH.with_name(f".{STATE_PATH.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, STATE_PATH)


def main() -> None:
    args = parse_args()
    slots = _slots(args.gpu_slots)
    num_shards = len(slots)
    routes_by_id = registered_model_routes()
    selected = set(args.model_id)
    routes = [
        route
        for model_id, route in routes_by_id.items()
        if not selected or model_id in selected
    ]
    if not routes:
        raise SystemExit("No router-registered models selected.")

    if args.clean:
        print("CLEAN START", flush=True)
        _clean(routes)
        print("CLEAN DONE", flush=True)
    for route in routes:
        initialize_artifact(route)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    tasks: queue.Queue[Task] = queue.Queue()
    total_tasks = 0
    for route in routes:
        for layer in route.final_layers:
            for shard_index in range(num_shards):
                tasks.put(Task(route.model_id, layer.key, route.framework, shard_index))
                total_tasks += 1

    lock = threading.Lock()
    completed: list[dict] = []
    active: dict[str, dict] = {}
    failures: list[dict] = []

    def write_state(status: str = "running") -> None:
        _atomic_state(
            {
                "status": status,
                "phase": "final",
                "batch_size": args.batch_size,
                "num_shards": num_shards,
                "total_tasks": total_tasks,
                "completed_tasks": len(completed),
                "queued_tasks": tasks.qsize(),
                "active": dict(active),
                "failures": list(failures),
            }
        )

    def worker(slot: Slot) -> None:
        while True:
            try:
                task = tasks.get_nowait()
            except queue.Empty:
                return
            route = routes_by_id[task.model_id]
            attempt = 0
            while True:
                attempt += 1
                command, environment = _command(
                    task,
                    route,
                    slot,
                    num_shards,
                    args.allocator_limit_mib,
                    args.batch_size,
                    args.overwrite,
                )
                log_dir = LOG_ROOT / task.model_id / task.layer
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"shard-{task.shard_index:03d}.log"
                with lock:
                    active[slot.key] = {
                        "model_id": task.model_id,
                        "layer": task.layer,
                        "shard_index": task.shard_index,
                        "attempt": attempt,
                    }
                    write_state()
                    print(
                        f"START {slot.key} {task.model_id}:{task.layer} "
                        f"shard={task.shard_index}/{num_shards} attempt={attempt}",
                        flush=True,
                    )
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(
                        f"\nSTART {slot.key} attempt={attempt} "
                        f"time={time.time()}\n"
                    )
                    log.flush()
                    returncode = subprocess.call(
                        command,
                        cwd=REPO_ROOT,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                valid_state = False
                state_path = _state_path(task, num_shards)
                if returncode == 0 and state_path.is_file():
                    result = json.loads(state_path.read_text(encoding="utf-8"))
                    valid_state = (
                        result.get("complete") is True
                        and result.get("failed") == 0
                    )
                if valid_state:
                    with lock:
                        active.pop(slot.key, None)
                        completed.append(
                            {
                                "gpu": slot.gpu,
                                "slot": slot.index,
                                "model_id": task.model_id,
                                "layer": task.layer,
                                "shard_index": task.shard_index,
                            }
                        )
                        write_state()
                        print(
                            f"DONE {slot.key} {task.model_id}:{task.layer} "
                            f"shard={task.shard_index}/{num_shards}",
                            flush=True,
                        )
                    break
                with lock:
                    failure = {
                        "gpu": slot.gpu,
                        "slot": slot.index,
                        "model_id": task.model_id,
                        "layer": task.layer,
                        "shard_index": task.shard_index,
                        "attempt": attempt,
                        "returncode": returncode,
                    }
                    failures.append(failure)
                    write_state()
                    print(f"RETRY {failure}", flush=True)
                time.sleep(max(5, args.retry_delay))
            tasks.task_done()

    with lock:
        write_state()
    threads = [
        threading.Thread(target=worker, args=(slot,), name=slot.key)
        for slot in slots
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    with lock:
        write_state("complete")
    print(
        json.dumps(
            {
                "status": "complete",
                "phase": "final",
                "batch_size": args.batch_size,
                "num_shards": num_shards,
                "tasks": total_tasks,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
