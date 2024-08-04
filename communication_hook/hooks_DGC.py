import math
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.algorithms import ddp_comm_hooks

from enum import Enum
from functools import partial


class correlation_state(object):  # Equivalent to memory
    """
    used to compress gradients and save local compression index matrices 'mask'
    correction_mod: mod of correction 'mask'
    options: 'adaptive' -- default, exponential adaptation increases the frequency of correction, like: 1, 2, 3,...
              int value -- specified the frequency of correction
    compression_rate: compression_rate = 0.1 means only 10% of the gradient will be retained
    momentum: a parameters used to gradient accumulation (gradients that are not transmitted)
    """

    def __init__(self, process_group, compression_rate: float = 0.032):
        self.process_group = process_group

        '''
        mask: the 0,1 matrix with the same size as the gradient matrix is used to judge whether the gradient at the corresponding position should be compressed 
        need_correction: whether the mask matrix need to be corrected
        bucket_index: because multiple buckets share the same state, bucket_index is set to match the correct mask matrix
        step: number of optimizations since last corrected
        '''
        self.mask = None

        self.velocities = None
        self.momentums = None
        self.need_init = True
        self.need_update = False
        self.first_buffer = False

        self.bucket_index = 0
        self.need_correction = True
        self.step = 1
        self.compression_rate = min(max(compression_rate, 0), 1)

    def compensate(self, tensor: torch.Tensor):
        m = 0.9
        if self.momentums == None:
            self.momentums = tensor
            self.velocities = tensor
        elif len(self.momentums) < self.bucket_index + len(tensor):
            self.momentums = torch.concat([self.momentums, tensor])
            self.velocities = torch.concat([self.velocities, tensor])
        else:
            self.momentums[self.bucket_index: self.bucket_index + len(tensor)].mul_(m).add_(tensor)
            self.velocities[self.bucket_index: self.bucket_index + len(tensor)].add_(
                self.momentums[self.bucket_index: self.bucket_index + len(tensor)])

    def update(self, tensor: torch.Tensor, index):
        self.momentums[self.bucket_index: self.bucket_index + len(tensor)]. \
            index_fill_(0, index, 0)
        self.velocities[self.bucket_index: self.bucket_index + len(tensor)]. \
            index_fill_(0, index, 0)

    # Get the index of the non-zero gradient after Sparsification
    def get_mask(self, base_grad: torch.Tensor, local_grad_list: list, bucket_index: int):
        mask_this_bucket = torch.zeros(base_grad.shape, dtype=torch.int8, device=base_grad.device)  # Initialize to 0
        start = 0  # pointer

        for grad in local_grad_list:
            size = grad.shape
            length = len(grad.view(-1))
            # Here you can choose which parts of the model to compress, for example, you can only compress the convolutional layers: 
            if len(size) == 4:
                vec = self.velocities[bucket_index + start: bucket_index + start + length].reshape (-1)  # NOTE
                row = math.floor((vec.shape[0]) * (1 - self.compression_rate))  # The number of gradients to zero is rounded down
                idx1 = torch.argsort(torch.abs(vec), descending=False)  # Sort from smallest to largest, argsort only returns indexes
                idx = idx1[:row]  # Indexes for smaller row rows
                com_vec = vec.index_fill(0, idx, 0)
                indices = com_vec.nonzero().view(-1)  # Indexes with non-zero rows
                mask_this_bucket[indices + start] = 1
            else:
                mask_this_bucket[start: start + length] = 1
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
    Has the same functionality as default allreduce, but you can customize some functions
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

    async_op = True  # must be True
    ''' In the case of training on hardware with high computing performance(e.g. NVIDIA A100), large node performance gap, or large models(e.g. VGG16), it is recommended to set 'blocked = True'! '''
    blocked = True

    tensor = bucket.buffer()  # Model gradient as a one-dimensional tensor
    local_grad_memory = bucket.buffer().clone()

    state.compensate(tensor)  # Fetching velocity/momentum (model)
    vec_this_bucket = state.velocities[state.bucket_index: state.bucket_index + len(local_grad_memory)]  # Velocity (bucket)
    state.get_mask(local_grad_memory, bucket.gradients(), state.bucket_index) # Obtain mask

    # Transfer index
    local_mask = state.mask[state.bucket_index: state.bucket_index + len(local_grad_memory)]
    local_index = local_mask.nonzero().view(-1)  # Non-zero index
    index_list = _get_all_gather_list(local_index, world_size)  # Create an empty buffer for all_gather
    '''all gather index'''
    fut = dist.all_gather(index_list, local_index, group=group_to_use, async_op=async_op).get_future()
    if blocked:
        dist.barrier()

    # To clarify, the transmission index and gradient are separated, but in practice they can be combined. 
    def compressed(fut):
        if blocked:
            dist.barrier()
        index_list = fut.wait()[0]  # Receive
        compressed_grad = vec_this_bucket[index_list[rank]]  # Non-sparse velocity to be transmitted, one-dimensional
        compressed_grad = compressed_grad.half()    # float32 -> float16

        '''all gather compressed gradient'''
        fut = dist.all_gather(_get_all_gather_list(compressed_grad, world_size), compressed_grad,
                              group=group_to_use, async_op=async_op).get_future()
        if blocked:
            dist.barrier()
        return fut.wait()

    def decompressed(fut):
        if blocked:
            dist.barrier()
        compressed_grad_list = fut.wait()[0]


        # TODO: Sparse gradient aggregation
        grad = torch.zeros_like(local_grad_memory, dtype=local_grad_memory.dtype, device=local_grad_memory.device)
        for i in range(world_size):
            grad[index_list[i]] = grad[index_list[i]] + compressed_grad_list[i]

        decompressed_grad = bucket.buffer()
        decompressed_grad.copy_(grad.div_(world_size))

        if blocked:
            dist.barrier()
        return decompressed_grad

    a = fut.then(compressed).then(decompressed)

    # Update velocity/momentum
    state.update(tensor, local_index)  # Note: Updating must be done after transmitting velocities; otherwise, the compressed_grad_list will be all zeros.

    state.bucket_index += len(local_grad_memory)
    if bucket.is_last():
        state.bucket_index = 0

    return a


def _correlation_comm_hook_wrapper(comm_hook, model, state, correction_mod='adaptive', compression_rate=0.001):
    model.register_comm_hook(
        correlation_state(process_group=state, compression_rate=compression_rate),
        comm_hook)


def _ddp_comm_hook_wrapper(comm_hook, model, state):
    model.register_comm_hook(state, comm_hook)


#  Generate hook
class DDPCommHookType(Enum):
    All_Reduce = partial(_ddp_comm_hook_wrapper, comm_hook=default_all_reduce)
    Debug_All_Reduce = partial(_ddp_comm_hook_wrapper, comm_hook=debug_all_reduce)
    correlation_GC = partial(_correlation_comm_hook_wrapper, comm_hook=correlation_gc_hook)
