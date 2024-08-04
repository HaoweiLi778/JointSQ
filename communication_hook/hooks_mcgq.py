import math
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.algorithms import ddp_comm_hooks
from scipy.io import savemat
from scipy.stats import wasserstein_distance
from enum import Enum
from functools import partial
import random

def monte_carlo_gradient_quantization(gradients, K):
    device = gradients.device
    
    N = int(len(gradients) * K)
    
    glPDF = torch.abs(gradients) / torch.sum(torch.abs(gradients))
    glCDF = torch.cumsum(glPDF, dim=0)
    r = random.random()
    random_numbers = torch.zeros(N,device=device)
    i = torch.arange(N,device=device)
    random_numbers = (i+r) / N

    random_numbers = torch.clip(random_numbers, 0.0, glCDF[-1])

    mapped_indices = torch.bucketize(random_numbers, glCDF)
    
    mask_length = len(glCDF)
    mask = torch.zeros(mask_length, dtype=torch.int64, device=device)
    mask.scatter_add_(0, mapped_indices, torch.ones_like(mapped_indices, dtype=torch.int64))
    sign = torch.sign(gradients)

    mask = (mask * sign)/N * (torch.sum(torch.abs(gradients)))

    return mask

def quantize(x,input_compress_settings={}):
    compress_settings={'n':128}
    compress_settings.update(input_compress_settings)
    n=compress_settings['n']
    x=x.float()
    x_norm=torch.norm(x,p=float('inf'))
    sgn_x=((x>0).float()-0.5)*2
    p=torch.div(torch.abs(x),x_norm)
    renormalize_p=torch.mul(p,n)
    floor_p=torch.floor(renormalize_p)
    compare=torch.rand_like(floor_p)
    final_p=renormalize_p-floor_p
    margin=(compare < final_p).float()
    xi=(floor_p+margin)/n
    
    Tilde_x=x_norm*sgn_x*xi
    
    return Tilde_x


class correlation_state(object):  # Equivalent to memory
    """
    used to compress gradients and save local compression index matrices 'mask'
    correction_mod: mod of correction 'mask'
    options: 'adaptive' -- default, exponential adaptation increases the frequency of correction, like: 1, 2, 3,...
              int value -- specified the frequency of correction
    compression_rate: compression_rate = 0.1 means only 10% of the gradient will be retained
    momentum: a parameters used to gradient accumulation (gradients that are not transmitted)
    """

    def __init__(self, process_group):
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

        self.list = []
        self.list1 = []

        self.bucket_index = 0
        self.need_correction = True
        self.step = 1


    def compensate(self, tensor: torch.Tensor):
        if self.velocities == None:
            self.velocities = tensor
        elif len(self.velocities) < self.bucket_index + len(tensor):
            self.velocities = torch.concat([self.velocities, tensor])
        else:
            self.velocities[self.bucket_index: self.bucket_index + len(tensor)].add_(tensor)

    def update(self, tensor: torch.Tensor, index):
        self.velocities[self.bucket_index: self.bucket_index + len(tensor)]. \
            index_fill_(0, index, 0)

    def get_mask(self,rank,base_grad: torch.Tensor, local_grad_list: list, bucket_index: int):
        
        mask_this_bucket = torch.zeros_like(base_grad) # Initialize to 0
        start = 0  # pointer
        for grad in local_grad_list:
            size = grad.shape
            length = len(grad.view(-1))
            # Here you can choose which parts of the model to compress, for example, you can only compress the convolutional layers: 
            if len(size) == 4:
                vec = self.velocities[bucket_index + start: bucket_index + start + length].reshape (-1)  # NOTE
                # Set the compression parameter
                vec_q =monte_carlo_gradient_quantization(vec,0.125)         
                mask_this_bucket[start: start + length] = vec_q
            else:  
                vec = self.velocities[bucket_index + start: bucket_index + start + length].reshape (-1)  # NOTE
                mask_this_bucket[start: start + length] = vec
            start = start + length
        
        if bucket_index == 0:
            self.mask = mask_this_bucket
        else:
            self.mask = torch.cat([self.mask, mask_this_bucket], dim=-1)


def _allreduce_fut(
        process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future[torch.Tensor]:
    group_to_use = process_group if process_group is not None else dist.group.WORLD  
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

    # print(state.bucket_index)
    state.compensate(tensor)  # Fetching velocity/momentum (model)
    vec_this_bucket = state.velocities[state.bucket_index: state.bucket_index + len(local_grad_memory)]  # Velocity (bucket)
    state.get_mask(rank , local_grad_memory, bucket.gradients(), state.bucket_index)
    # 传输index
    local_mask = state.mask[state.bucket_index: state.bucket_index + len(local_grad_memory)]
    local_index = local_mask.nonzero().view(-1)  # Non-zero index
    index_list = _get_all_gather_list(local_mask, world_size)  # Create an empty buffer for all_gather
    '''all gather index'''
    fut = dist.all_gather(index_list, local_mask, group=group_to_use, async_op=async_op).get_future()
    if blocked:
        dist.barrier()

    def decompressed(fut):
        if blocked:
            dist.barrier()
        compressed_grad_list = fut.wait()[0]
        grad = torch.zeros_like(local_grad_memory, dtype=local_grad_memory.dtype, device=local_grad_memory.device)
        for i in range(world_size):
            grad1 = compressed_grad_list[i]
            grad= grad + grad1

        decompressed_grad = bucket.buffer()
        decompressed_grad.copy_(grad.div_(world_size))

        if blocked:
            dist.barrier()
        return decompressed_grad

    a = fut.then(decompressed)

    # Update velocity/momentum
    state.update(tensor, local_index)  # Note: Updating must be done after transmitting velocities; otherwise, the compressed_grad_list will be all zeros.

    state.bucket_index += len(local_grad_memory)
    if bucket.is_last():
        state.bucket_index = 0

    return a


def _correlation_comm_hook_wrapper(comm_hook, model, state, correction_mod='adaptive', compression_rate=0.001):
    model.register_comm_hook(
        correlation_state(process_group=state),
        comm_hook)


def _ddp_comm_hook_wrapper(comm_hook, model, state):
    model.register_comm_hook(state, comm_hook)


class DDPCommHookType(Enum):
    All_Reduce = partial(_ddp_comm_hook_wrapper, comm_hook=default_all_reduce)
    Debug_All_Reduce = partial(_ddp_comm_hook_wrapper, comm_hook=debug_all_reduce)
    correlation_GC = partial(_correlation_comm_hook_wrapper, comm_hook=correlation_gc_hook)

