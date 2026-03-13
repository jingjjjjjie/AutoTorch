'''
torchrun spawns 2 separate processes, each running the entire train.py script:

for example torchrun --nproc_per_node=2 train.py
torchrun spawns 2 separate processes, each running the entire train.py script

Process 0: runs train.py → setup_ddp() → LOCAL_RANK=0
Process 1: runs train.py → setup_ddp() → LOCAL_RANK=1
'''
import os
import torch
import torch.distributed as dist
from functools import wraps
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp()-> int:
    '''
    Initialize PyTorch Distributed Data Parallel (DDP) environment
    and configure the current process to use the correct GPU.
    Prints GPU info from all ranks.

    Returns a integer specifying the rank
    '''
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    _print_gpu_info(local_rank)
    return local_rank

def cleanup_ddp():
    '''Destroy the process group.'''
    dist.destroy_process_group()

def is_main_process() -> bool:
    '''
    Check if this is the main process.
    - DDP mode: returns True only for rank 0
    - Non-DDP mode: always returns True (single process is always main)
    '''
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def main_process_only(func):
    '''Decorator to run function only on main process (rank 0).'''
    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
    return wrapper


def wrap_model_ddp(model: torch.nn.Module, local_rank: int) -> torch.nn.Module:
    '''Wrap model with DDP.'''
    return DDP(model, device_ids=[local_rank])

def _print_gpu_info(local_rank: int, title: str = "DDP Training"):
    '''Utility function to print GPU info from all ranks in order.'''
    if is_main_process():
        print("=" * 40)
        print(title)
        print("=" * 40)

    for rank in range(dist.get_world_size()):
        if local_rank == rank:
            props = torch.cuda.get_device_properties(local_rank)
            free_mem, total_mem = torch.cuda.mem_get_info(local_rank)
            print(f"[{local_rank}] {props.name} - {free_mem / 1e9:.1f}GB free / {total_mem / 1e9:.1f}GB total")
        dist.barrier()

    if is_main_process():
        print("=" * 40)

