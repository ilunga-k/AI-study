import torch
import contextlib


def init_torch_seeds(seed=0):
    """Set random seed for torch and CUDA."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def torch_distributed_zero_first(local_rank: int):
    """Yield function ensuring that code runs only on rank 0 first in DDP."""
    if local_rank not in (-1, 0) and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    yield
    if local_rank == 0 and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
