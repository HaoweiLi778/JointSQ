import torch
import torch.distributed as dist
import os
import random
import math

def get_top_x_indices(tensor, x, sample_ratio=1.0, strided_sample=True,
                      compress_upper_bound=1.3, compress_lower_bound=0.8,
                      max_adaptation_iters=10, resample=True):
    """
    Get the indices of the top-x elements in the tensor with sampling and adaptive adjustment.

    Args:
        tensor (torch.Tensor): Input tensor.
        x (int): Number of top elements to retrieve.
        sample_ratio (float): Sampling ratio (0, 1]; 1.0 means no sampling.
        strided_sample (bool): Use strided sampling if True, otherwise random sampling.
        compress_upper_bound (float): Upper bound for adaptive adjustment.
        compress_lower_bound (float): Lower bound for adaptive adjustment.
        max_adaptation_iters (int): Maximum iterations for adjustment.
        resample (bool): Whether to resample during adjustment.

    Returns:
        indices (torch.Tensor): Indices of the top-x elements.
    """
    tensor = tensor.view(-1)  # Flatten the tensor
    numel = tensor.numel()  # Total number of elements
    radio = x / numel

    # Step 1: Compute importance (absolute values)
    importance = tensor.abs()

    # Step 2: Sampling
    if sample_ratio < 1.0:
        num_samples = int(max(1, numel * sample_ratio))  # Number of samples
        if strided_sample:
            # Strided sampling
            sample_stride = max(1, numel // num_samples)
            sample_start = random.randint(0, sample_stride - 1)  # Random starting point
            samples = importance[sample_start::sample_stride]
        else:
            # Random sampling
            samples = importance[torch.randint(0, numel, (num_samples,), device=tensor.device)]
    else:
        # Use the entire tensor if no sampling
        samples = importance
        num_samples = numel

    top_k_samples = min(x, math.ceil(num_samples * radio))
    threshold = torch.min(torch.topk(samples, top_k_samples, largest=True, sorted=False)[0])

    # Step 3: Mask elements above the threshold
    mask = importance >= threshold
    indices = mask.nonzero(as_tuple=False).view(-1)  # Get indices
    num_indices = indices.numel()  # Count selected elements

    # Step 4: Adaptive adjustment
    for _ in range(max_adaptation_iters):
        if num_indices > x:
            # Too many elements
            if num_indices > x * compress_upper_bound:
                if resample:
                    # Resample top-x elements from current indices
                    indices = indices[
                        torch.topk(importance[indices], x, largest=True, sorted=False)[1]
                    ]
                    break
                else:
                    # Increase threshold
                    threshold *= compress_upper_bound
            else:
                break
        elif num_indices < compress_lower_bound * x:
            # Too few elements, decrease threshold
            threshold *= compress_lower_bound
        else:
            break

        # Recompute mask and indices
        mask = importance >= threshold
        indices = mask.nonzero(as_tuple=False).view(-1)
        num_indices = indices.numel()

    # Ensure only top-x elements are returned
    indices = indices[:x]

    return indices


def mckp_greedy(tensor, max_bit):
    """
    Solve a multi-choice knapsack problem (MCKP) greedily.
    """
    tensor = tensor.abs()
    device = tensor.device

    # Expand tensor for group processing
    squared_tensor = torch.pow(tensor, 2)
    expanded_tensor = squared_tensor.unsqueeze(1).repeat(1, 4)

    # Compute Incremental Profit Density (IPD) and select top candidates
    coefficients = torch.tensor([0, 0.46875, 0.029296875, 0.0009715625], device=device)
    multiplied_tensor = expanded_tensor * coefficients
    multiplied_tensor = torch.flatten(multiplied_tensor)
    SP = get_top_x_indices(multiplied_tensor, math.ceil(max_bit / 2), sample_ratio=0.01, strided_sample=False)

    # Replace selected items based on bit-width constraints
    replaced_tensor = torch.where(SP % 4 == 0, torch.tensor(0, device=device),
                                  torch.where(SP % 4 == 1, torch.tensor(2, device=device),
                                              torch.where(SP % 4 == 2, torch.tensor(2, device=device),
                                                          torch.tensor(4, device=device)))) 

    # Stop when knapsack capacity is reached
    w = torch.cumsum(replaced_tensor, dim=0)
    idx = torch.bucketize(max_bit, w)
    idx = int(idx)
    choose = SP[:idx + 1]

    # Generate mask for quantization bit-width
    mask = torch.zeros_like(tensor)
    choose_mod = choose % 4
    indices = (choose / 4).to(torch.long)

    mask[indices[choose_mod == 1]] = 2
    mask[indices[choose_mod == 2]] = 4
    mask[indices[choose_mod == 3]] = 8

    return mask


def quantize(x, input_compress_settings={}):
    """
    Quantize the input tensor `x` to specified bit width and store as `torch.uint8`.

    Args:
        x (torch.Tensor): Input tensor.
        input_compress_settings (dict): Compression parameters, including:
            - 'n' (int): Quantization bit width (supports 2-bit, 4-bit, and 8-bit).

    Returns:
        packed_x (torch.Tensor): Quantized tensor stored as uint8.
    """
    compress_settings = {'n': 4}
    compress_settings.update(input_compress_settings)
    n = compress_settings['n']

    if n not in [2, 4, 8]:
        raise ValueError("Only 2-bit, 4-bit, and 8-bit quantization are supported.")

    device = x.device
    x = x.float().to(device)

    x_norm = torch.norm(x, p=float('inf'))
    if x_norm == 0:
        return torch.zeros_like(x, dtype=torch.uint8, device=device)

    sgn_x = ((x > 0).float() - 0.5) * 2
    levels = 2 ** n
    p = torch.abs(x) / x_norm

    renormalize_p = p * (levels - 1)
    floor_p = torch.floor(renormalize_p)
    compare = torch.rand_like(floor_p, device=device)
    margin = (compare < (renormalize_p - floor_p)).float()

    quantized_values = (floor_p + margin).to(torch.uint8)

    if n == 8:
        packed_x = quantized_values
    elif n == 4:
        length = quantized_values.size(0)
        if length % 2 != 0:
            padding = 2 - (length % 2)
            quantized_values = torch.cat([quantized_values, torch.zeros(padding, dtype=quantized_values.dtype, device=quantized_values.device)])
        high_bits = (quantized_values[::2] & 0xF) << 4
        low_bits = quantized_values[1::2] & 0xF
        packed_x = (high_bits | low_bits).to(torch.uint8)
    elif n == 2:
        length = quantized_values.size(0)
        if length % 4 != 0:
            padding = 4 - (length % 4)
            quantized_values = torch.cat([quantized_values, torch.zeros(padding, dtype=quantized_values.dtype, device=quantized_values.device)])
        packed_x = torch.zeros((quantized_values.numel() + 3) // 4, dtype=torch.uint8, device=device)
        for i in range(4):
            packed_x |= ((quantized_values[i::4] & 0x3) << (6 - 2 * i))

    return packed_x


def setup(rank, world_size, n_gpus):
    """
    Initialize the distributed environment.
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank % n_gpus)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def run(rank, world_size, tensor_size=100000, n_gpus=4):
    """
    Function to run for each process.
    """
    print(f"Rank {rank} initializing on GPU {rank % n_gpus}...")
    setup(rank, world_size, n_gpus)

    device = torch.device(f"cuda:{rank % n_gpus}")
    for i in range(10):
        tensor = torch.randn(tensor_size, device=device).reshape(-1)
        max_items = math.ceil(torch.abs(tensor).numel() * 32 * 0.001)
        mask1 = mckp_greedy(tensor, max_items)
        tensor_2 = quantize(tensor[mask1 == 2], input_compress_settings={'n': 2})
        tensor_4 = quantize(tensor[mask1 == 4], input_compress_settings={'n': 4})
        tensor_8 = quantize(tensor[mask1 == 8], input_compress_settings={'n': 8})

        result = torch.cat((tensor_2, tensor_4, tensor_8))
        local_length = result.size(0)
        lengths = torch.tensor([local_length], dtype=torch.int64, device=result.device)
        gathered_lengths = [torch.zeros_like(lengths) for _ in range(world_size)]
        dist.all_gather(gathered_lengths, lengths)

        max_length = max([int(length.item()) for length in gathered_lengths])
        if local_length < max_length:
            padding = torch.zeros(max_length - local_length, dtype=result.dtype, device=result.device)
            result = torch.cat([result, padding])
        gathered_values = [torch.zeros_like(result) for _ in range(world_size)]

        dist.all_gather(gathered_values, result)

    dist.barrier()
    cleanup()

if __name__ == "__main__":
    import torch.multiprocessing as mp

    n_gpus = torch.cuda.device_count()
    world_size = 16
    tensor_size = 1000000

    assert world_size % n_gpus == 0, "World size must be divisible by GPU count!"
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    mp.spawn(run, args=(world_size, tensor_size, n_gpus), nprocs=world_size)
    end_event.record()
    torch.cuda.synchronize()
    print(f"time: {start_event.elapsed_time(end_event):.6f} ms")
