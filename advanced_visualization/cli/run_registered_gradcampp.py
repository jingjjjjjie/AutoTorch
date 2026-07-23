"""Run all registered Grad-CAM++ jobs final-layer-first with a GPU memory guard."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from advanced_visualization.core.model_router import ModelRoute, registered_model_routes
from advanced_visualization.core.registered_gradcam import initialize_artifact


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "advanced_visualization" / "docker-compose.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--phase", choices=["all", "final", "lower"], default="all")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--memory-limit-mib", type=int, default=3000)
    parser.add_argument("--allocator-limit-mib", type=int, default=2400)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--preflight-final", action="store_true")
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--ready-used-mib", type=int, default=500)
    parser.add_argument("--wait-interval", type=int, default=60)
    return parser.parse_args()


def _gpu_used_mib(gpu: int) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip().splitlines()[0])


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


def _command(
    route: ModelRoute,
    layer: str,
    args: argparse.Namespace,
    *,
    limit: int | None,
) -> tuple[list[str], dict[str, str]]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "AUTOTORCH_GRADCAM_DEVICE": "cuda:0",
            "AUTOTORCH_CUDA_MEMORY_LIMIT_MB": str(args.allocator_limit_mib),
            "VANSMALL_GPU_MEMORY_LIMIT_MB": str(args.allocator_limit_mib),
            "VANSMALL_CUDA_VISIBLE_DEVICES": str(args.gpu),
            "ADVANCED_VISUALIZATION_ARTIFACT_ROOT": "/mnt4/advanced_visualization",
        }
    )
    if route.framework == "pytorch":
        command = [
            sys.executable,
            "-m",
            "advanced_visualization.cli.generate_registered_gradcam",
            route.model_id,
            "--layer",
            layer,
        ]
    elif route.framework == "tensorflow":
        gpu_uuid = _gpu_uuid(args.gpu)
        command = [
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
            f"VANSMALL_GPU_MEMORY_LIMIT_MB={args.allocator_limit_mib}",
            "tensorflow-vansmall-live",
            "python",
            "-m",
            "advanced_visualization.tensorflow_service.batch",
            route.model_id,
            "--layer",
            layer,
        ]
    else:
        raise ValueError(f"Unsupported Grad-CAM framework: {route.framework}")
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command, environment


def _run_guarded(
    route: ModelRoute,
    layer: str,
    args: argparse.Namespace,
    *,
    limit: int | None,
) -> int:
    command, environment = _command(route, layer, args, limit=limit)
    baseline = _gpu_used_mib(args.gpu)
    print(f"START {route.model_id} layer={layer} framework={route.framework}", flush=True)
    process = subprocess.Popen(command, cwd=REPO_ROOT, env=environment)
    peak = 0
    while process.poll() is None:
        used = _gpu_used_mib(args.gpu)
        job_used = max(0, used - baseline)
        peak = max(peak, job_used)
        if job_used >= args.memory_limit_mib:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
            raise RuntimeError(
                f"GPU {args.gpu} increased by {job_used} MiB while running "
                f"{route.model_id}:{layer}; hard limit is {args.memory_limit_mib} MiB."
            )
        time.sleep(1.0)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    print(f"DONE {route.model_id} layer={layer} peak_job_gpu_mib={peak}", flush=True)
    return peak


def _jobs(routes: list[ModelRoute], phase: str):
    if phase in {"all", "final"}:
        for route in routes:
            for layer in route.final_layers:
                yield route, layer.key
    if phase in {"all", "lower"}:
        for route in routes:
            for layer in route.non_final_layers:
                yield route, layer.key


def main() -> None:
    args = parse_args()
    routes_by_id = registered_model_routes()
    selected = set(args.model_id)
    routes = [
        route
        for model_id, route in routes_by_id.items()
        if not selected or model_id in selected
    ]
    if not routes:
        raise SystemExit("No router-registered models selected.")
    baseline = _gpu_used_mib(args.gpu)
    while args.wait_for_gpu and baseline > args.ready_used_mib:
        print(
            f"WAIT GPU {args.gpu} used={baseline} MiB; starting at "
            f"<={args.ready_used_mib} MiB.",
            flush=True,
        )
        time.sleep(max(5, args.wait_interval))
        baseline = _gpu_used_mib(args.gpu)
    if baseline > args.ready_used_mib:
        raise SystemExit(
            f"GPU {args.gpu} already uses {baseline} MiB; refusing to start until "
            f"usage is <= {args.ready_used_mib} MiB."
        )
    for route in routes:
        initialize_artifact(route)

    peak = baseline
    if args.preflight_final and args.phase in {"all", "final"}:
        print("Running one-image final-layer preflight for every model.", flush=True)
        for route in routes:
            for layer in route.final_layers:
                peak = max(peak, _run_guarded(route, layer.key, args, limit=1))

    completed = []
    for route, layer in _jobs(routes, args.phase):
        peak = max(peak, _run_guarded(route, layer, args, limit=args.limit))
        completed.append({"model_id": route.model_id, "layer": layer})
        state = {
            "gpu": args.gpu,
            "memory_limit_mib": args.memory_limit_mib,
            "peak_job_gpu_mib": peak,
            "completed": completed,
        }
        state_path = Path("/mnt4/advanced_visualization/gradcam_run_state.json")
        temporary = state_path.with_name(f".{state_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, state_path)
    print(
        json.dumps(
            {
                "status": "complete",
                "jobs": len(completed),
                "peak_job_gpu_mib": peak,
                "memory_limit_mib": args.memory_limit_mib,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
