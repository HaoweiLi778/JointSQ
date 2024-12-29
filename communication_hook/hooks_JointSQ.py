import math
import numpy as np
import torch
import torch.distributed as dist
import random

from torch.distributed.algorithms import ddp_comm_hooks
from scipy.io import savemat
from scipy.stats import wasserstein_distance
from enum import Enum
from functools import partial


class correlation_state(object):  # Equivalent to memory
    """
    Used to compress gradients and save local compression index matrices 'mask'.
    correction_mod: mode of correction 'mask'.
    options: 'adaptive' -- default, exponential adaptation increases the frequency of correction, like: 1, 2, 3,...
              int value -- specifies the frequency of correction.
    compression_rate: compression_rate = 0.1 means only 10% of the gradient will be retained.
    momentum: a parameter used for gradient accumulation (gradients that are not transmitted).
    """

    def __init__(self, process_group):
        self.process_group = process_group

        '''
        mask: A 0/1 matrix with the same size as the gradient matrix, used to determine whether the gradient at the corresponding position should be compressed.
        need_correction: Whether the mask matrix needs to be corrected.
        bucket_index: Since multiple buckets share the same state, bucket_index is set to match the correct mask matrix.
        step: Number of optimizations since last correction.
        '''
        self.mask = None
        self.compression_rate = 0.032
        self.velocities = None
        self.momentums = None
        self.need_init = True
        self.need_update = False
        self.first_buffer = False
        self.bucket_index = 0
        self.need_correction = True

    def get_top_x_indices(self, tensor, x, sample_ratio=1.0, strided_sample=True,
                          compress_upper_bound=1.3, compress_lower_bound=0.8,
                          max_adaptation_iters=10, resample=True):
        """
        Get the indices of the top x elements in a tensor, including sampling and adaptive adjustment logic.

        Args:
            tensor (torch.Tensor): Input tensor.
            x (int): Number of top x elements to retrieve.
            sample_ratio (float): Sampling ratio, range (0, 1]. 1.0 means no sampling.
            strided_sample (bool): Whether to use strided sampling. True means sampling with a fixed stride.
            compress_upper_bound (float): Upper bound for adaptive adjustment.
            compress_lower_bound (float): Lower bound for adaptive adjustment.
            max_adaptation_iters (int): Maximum number of iterations for adaptive adjustment.
            resample (bool): Whether to perform top-k operation again during adjustment.

        Returns:
            indices (torch.Tensor): Indices of the top x elements.
        """
        tensor = tensor.view(-1)  # Flatten the tensor
        numel = tensor.numel()  # Total number of elements
        ratio = x / numel

        # Step 1: Calculate importance (absolute value)
        importance = tensor.abs()

        # Step 2: Sampling
        if sample_ratio < 1.0:
            num_samples = int(max(1, numel * sample_ratio))  # Number of samples
            if strided_sample:
                # Strided sampling
                sample_stride = max(1, numel // num_samples)
                sample_start = random.randint(0, sample_stride - 1)  # Random start point
                samples = importance[sample_start::sample_stride]
            else:
                # Random sampling
                samples = importance[torch.randint(0, numel, (num_samples,), device=tensor.device)]
        else:
            # No sampling, use the entire tensor
            samples = importance
            num_samples = numel

        top_k_samples = min(x, math.ceil(num_samples * ratio))
        # Step 3: Find the top k largest elements in the sampled data
        threshold = torch.min(torch.topk(samples, top_k_samples, largest=True, sorted=False)[0])

        # Generate a mask based on the threshold to filter out elements
        mask = importance >= threshold
        indices = mask.nonzero(as_tuple=False).view(-1)  # Get indices
        num_indices = indices.numel()  # Number of selected elements

        # Step 4: Adaptive adjustment
        for _ in range(max_adaptation_iters):
            if num_indices > x:
                # If too many elements are selected
                if num_indices > x * compress_upper_bound:
                    if resample:
                        # Perform top-k operation again, directly selecting the top x elements
                        indices = indices[
                            torch.topk(importance[indices], x, largest=True, sorted=False)[1]
                        ]
                        break
                    else:
                        # Increase the threshold
                        threshold *= compress_upper_bound
                else:
                    break
            elif num_indices < compress_lower_bound * x:
                # If too few elements are selected, decrease the threshold
                threshold *= compress_lower_bound
            else:
                break

            # Recalculate the mask and indices based on the new threshold
            mask = importance >= threshold
            indices = mask.nonzero(as_tuple=False).view(-1)
            num_indices = indices.numel()

        # Ensure only the top x elements are returned
        indices = indices[:x]

        return indices

    # MCKP-Greedy is used to determine the quantization strategy
    def mckp_greedy(self, tensor, max_bit):
        tensor = tensor.abs()
        device = tensor.device

        # Group knapsack problem, with four items in each group
        squared_tensor = torch.pow(tensor, 2)
        expanded_tensor = squared_tensor.unsqueeze(1).repeat(1, 4)

        # Calculate the Incremental Profit Density for each item and select the highest
        coefficients = torch.tensor([0, 0.46875, 0.029296875, 0.0009715625], device=device)  # Based on equations (13) and (14) in the paper.
        multiplied_tensor = expanded_tensor * coefficients
        multiplied_tensor = torch.flatten(multiplied_tensor)
        SP = self.get_top_x_indices(multiplied_tensor, math.ceil(max_bit / 2), sample_ratio=0.01, strided_sample=False)
        # Each time an item is selected, the preceding item in the same group must be discarded. 
        # For example, selecting 8-bit requires discarding 4-bit, hence replace with 4.
        replaced_tensor = torch.where(SP % 4 == 0, torch.tensor(0, device=device),
                                      torch.where(SP % 4 == 1, torch.tensor(2, device=device),
                                                  torch.where(SP % 4 == 2, torch.tensor(2, device=device),
                                                              torch.tensor(4, device=device))))

        # Stop selecting items when the knapsack capacity is reached.
        w = torch.cumsum(replaced_tensor, dim=0)
        idx = torch.bucketize(max_bit, w)
        idx = int(idx)
        choose = SP[:idx + 1]

        # Obtain the mask for quantization bit-width.
        mask = torch.zeros_like(tensor)
        choose_mod = choose % 4
        indices = (choose / 4).to(torch.long)

        mask[indices[choose_mod == 1]] = 2
        mask[indices[choose_mod == 2]] = 4
        mask[indices[choose_mod == 3]] = 8

        return mask

    # Quantization
    def quantize(self, x, input_compress_settings={}):
        compress_settings = {'n': 6}
        compress_settings.update(input_compress_settings)
        n = compress_settings['n']
        x = x.float()
        x_norm = torch.norm(x, p=float('inf'))
        sgn_x = ((x > 0).float() - 0.5) * 2
        p = torch.div(torch.abs(x), x_norm)
        renormalize_p = torch.mul(p, n)
        floor_p = torch.floor(renormalize_p)
        compare = torch.rand_like(floor_p)
        final_p = renormalize_p - floor_p
        margin = (compare < final_p).float()
        xi = (floor_p + margin) / n
        Tilde_x = x_norm * sgn_x * xi

        return Tilde_x

    # Accuracy compensation
    def compensate(self, tensor: torch.Tensor):
        if self.velocities is None:
            self.velocities = tensor
        elif len(self.velocities) < self.bucket_index + len(tensor):
            self.velocities = torch.concat([self.velocities, tensor])
        else:
            self.velocities[self.bucket_index: self.bucket_index + len(tensor)].add_(tensor)

    def update(self, error: torch.Tensor):
        self.velocities[self.bucket_index: self.bucket_index + len(error)] = error

    def get_mask(self, rank, base_grad: torch.Tensor, local_grad_list: list, bucket_index: int): 
        mask_this_bucket = torch.zeros_like(base_grad)  # Initialize to 0
        start = 0  # Pointer

        for grad in local_grad_list:
            size = grad.shape
            length = len(grad.view(-1))
            # Here you can choose which parts of the model to compress, for example, you can only compress the convolutional layers: 
            if len(size) == 4:
                vec = self.velocities[bucket_index + start: bucket_index + start + length].reshape(-1)  # NOTE
                # Set the compression ratio
                max_items = math.ceil(torch.abs(vec).numel() * 32 * self.compression_rate)
                mask = self.mckp_greedy(torch.abs(vec), max_items)
                vec_q = torch.zeros_like(vec)
                vec_q[mask == 2] = self.quantize(vec[mask == 2], input_compress_settings={'n': 2})
                vec_q[mask == 4] = self.quantize(vec[mask == 4], input_compress_settings={'n': 8})
                vec_q[mask == 8] = self.quantize(vec[mask == 8], input_compress_settings={'n': 128})
                mask_this_bucket[start: start + length] = vec_q
            else:  
                vec = self.velocities[bucket_index + start: bucket_index + start + length].reshape(-1)  # NOTE
                mask_this_bucket[start: start + length] = vec
            start = start + length

        if bucket_index == 0:
            self.mask = mask_this_bucket
        else:
            self.mask = torch.cat([self.mask, mask_this_bucket], dim=-1)

def _allreduce_fut(
        process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future[torch.Tensor]:
    group_to_use = process_group if process_group is not None else dist.group.WORLD  # Obtain process group
    tensor.div_(group_to_use.size())

    return (
        dist.all_reduce(tensor, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )


def _get_all_gather_list(tensor: torch.Tensor, world_size: int):
    all_gather_list = [torch.zeros_like(tensor, dtype=tensor.dtype, device=tensor.device) for _ in range(world_size)]

    return all_gather_list


def default_all_reduce(
        process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    return _allreduce_fut(process_group, bucket.buffer())


def debug_all_reduce(
        process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    Same functionality as default allreduce, but allows customization of some functions.
    """

    tensor = bucket.buffer()
    real_rate = len(torch.where(tensor == 0)[0]) / tensor.shape[0]

    return _allreduce_fut(process_group, bucket.buffer())


# Custom communication hook (gradient handling method and communication method)
def correlation_gc_hook(state: correlation_state, bucket: dist.GradBucket
                        ) -> torch.futures.Future[torch.Tensor]:
    # bucket.buffer()  Gradient: a one-dimensional tensor
    # bucket.gradients()  Gradient list layer by layer
    # bucket.parameters() Parameters
    # bucket.is_last()  Is this the last buffer
    # bucket.index()    Index of the buffer

    # Obtain process group/process ID/number of nodes
    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    rank = process_group.rank() if process_group is not None else dist.get_rank()
    world_size = group_to_use.size()

    async_op = True  # Must be True
    ''' In cases where training is performed on hardware with high computing performance (e.g., NVIDIA A100), large node performance gaps, or large models (e.g., VGG16), it is recommended to set 'blocked = True'! '''
    blocked = True

    tensor = bucket.buffer()  # Model gradient as a one-dimensional tensor
    local_grad_memory = bucket.buffer().clone()

    state.compensate(tensor)  # Fetching velocity/momentum (model)
    vec_this_bucket = state.velocities[state.bucket_index: state.bucket_index + len(local_grad_memory)]  # Velocity (bucket)
    state.get_mask(rank, local_grad_memory, bucket.gradients(), state.bucket_index)
    # Transfer compressed gradient
    local_mask = state.mask[state.bucket_index: state.bucket_index + len(local_grad_memory)]
    error = vec_this_bucket - local_mask
    fut = dist.all_reduce(local_mask, op=dist.ReduceOp.SUM, group=group_to_use, async_op=async_op).get_future()
    if blocked:
        dist.barrier()

    def decompressed(fut):
        if blocked:
            dist.barrier()
        aggregated_grad = fut.wait()[0]  # Aggregated quantized gradient

        # Average the aggregated gradient
        aggregated_grad.div_(world_size)

        # Copy the result to the target buffer
        decompressed_grad = bucket.buffer()
        decompressed_grad.copy_(aggregated_grad)

        if blocked:
            dist.barrier()
        return decompressed_grad

    a = fut.then(decompressed)

    # Update velocity/momentum
    state.update(error) 

    state.bucket_index += len(local_grad_memory)
    if bucket.is_last():
        state.bucket_index = 0

    return a


def _correlation_comm_hook_wrapper(comm_hook, model, state, correction_mod='adaptive'):
    model.register_comm_hook(
        correlation_state(process_group=state),
        comm_hook)


def _ddp_comm_hook_wrapper(comm_hook, model, state):
    model.register_comm_hook(state, comm_hook)


class DDPCommHookType(Enum):
    All_Reduce = partial(_ddp_comm_hook_wrapper, comm_hook=default_all_reduce)
    Debug_All_Reduce = partial(_ddp_comm_hook_wrapper, comm_hook=debug_all_reduce)
    correlation_GC = partial(_correlation_comm_hook_wrapper, comm_hook=correlation_gc_hook)
