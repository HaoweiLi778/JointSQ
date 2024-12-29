import torch
import torch.distributed as dist
import os
import random
import math

def setup(rank, world_size, n_gpus):
    """
    Initialize the distributed environment.
    rank: Current process rank
    world_size: Total number of processes
    n_gpus: Number of available GPUs
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    # Initialize the NCCL backend
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Assign multiple processes to each GPU
    torch.cuda.set_device(rank % n_gpus)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def run(rank, world_size, tensor_size=100000, n_gpus=4):
    """
    Function executed by each process.
    rank: Current process rank
    world_size: Total number of processes
    tensor_size: Length of the tensor on each node
    n_gpus: Number of available GPUs
    """
    print(f"Rank {rank} initializing on GPU {rank % n_gpus}...")
    setup(rank, world_size, n_gpus)

    # Each "node" creates a tensor of length `tensor_size`
    device = torch.device(f"cuda:{rank % n_gpus}")
    for i in range(10):
        tensor = torch.randn(tensor_size, device=device).reshape(-1)
        # Prepare a container and perform all_gather
        gathered_values = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_values, tensor)

    # Ensure all processes synchronize
    dist.barrier()

    # # Example: Perform a global sum (All-Reduce operation)
    # dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # # Print the result (only the first 5 elements)
    # print(f"Rank {rank} on GPU {rank % n_gpus} after all_reduce: {tensor[:5]} (sum of all ranks)")

    cleanup()

if __name__ == "__main__":
    import torch.multiprocessing as mp

    # Configuration parameters
    n_gpus = torch.cuda.device_count()  # Get the number of GPUs
    world_size = 16  # Simulate 16 distributed nodes
    tensor_size = 1000000  # The length of the tensor maintained by each node

    # Ensure the number of GPUs is sufficient
    assert world_size % n_gpus == 0, "World size must be a multiple of the number of GPUs!"
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    # Use multiprocessing to simulate the distributed environment
    mp.spawn(run, args=(world_size, tensor_size, n_gpus), nprocs=world_size)

    end_event.record()
    torch.cuda.synchronize()
    print(f"time: {start_event.elapsed_time(end_event):.6f} ms")
