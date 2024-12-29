import torch
import torch.distributed as dist
import os
import random
import math

def setup(rank, world_size, n_gpus):
    """
    初始化分布式环境。
    rank: 当前进程编号
    world_size: 总进程数
    n_gpus: 可用的 GPU 数量
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    

    # 初始化 NCCL 后端
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # 每张 GPU 分配多个进程
    torch.cuda.set_device(rank % n_gpus)

def cleanup():
    """清理分布式环境。"""
    dist.destroy_process_group()

def run(rank, world_size, tensor_size=100000, n_gpus=4):
    """
    每个进程运行的函数。
    rank: 当前进程编号
    world_size: 总进程数
    tensor_size: 每个节点张量的长度
    n_gpus: 可用的 GPU 数量
    """
    print(f"Rank {rank} initializing on GPU {rank % n_gpus}...")
    setup(rank, world_size, n_gpus)

    # 每个 "节点" 创建一个长度为 tensor_size 的张量
    device = torch.device(f"cuda:{rank % n_gpus}")
    for i in range(1):
        tensor = torch.randn(tensor_size, device=device).reshape (-1) 
        # 准备容器并执行 all_gather
        gathered_values = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_values, tensor)


    # 确保所有进程同步
    dist.barrier()

    # # 示例：全局求和（All-Reduce 操作）
    # dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # # 打印结果（只打印前 5 个元素）
    # print(f"Rank {rank} on GPU {rank % n_gpus} after all_reduce: {tensor[:5]} (sum of all ranks)")

    cleanup()

if __name__ == "__main__":
    import torch.multiprocessing as mp

    # 配置参数
    n_gpus = torch.cuda.device_count()  # 获取 GPU 数量
    world_size = 16  # 模拟 16 个分布式节点
    tensor_size = 1000000  # 每个节点维护的张量长度

    # 确保 GPU 数量足够
    assert world_size % n_gpus == 0, "世界大小必须是 GPU 数量的整数倍！"
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    # 使用多进程模拟分布式环境
    mp.spawn(run, args=(world_size, tensor_size, n_gpus), nprocs=world_size)

    end_event.record()
    torch.cuda.synchronize()
    print(f"time: {start_event.elapsed_time(end_event):.6f} ms")