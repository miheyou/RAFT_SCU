import sys
sys.path.append('core')

import argparse
import numpy as np

from config.parser import parse_args

import torch
import torch.optim as optim

from raft import RAFT
from datasets import fetch_dataloader
from utils.utils import load_ckpt
from loss import sequence_loss
from ddp_utils import *

os.system("export KMP_INIT_AT_FORK=FALSE")

def fetch_optimizer(args, model):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr, args.num_steps + 100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear'
    )
    return optimizer, scheduler

def train(args, rank=0, world_size=1, use_ddp=False):
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # ✅ 修复：定义 device
    device_id = rank  # 可用于 DDP，但 .to() 必须用 device 对象

    model = RAFT(args)
    model = model.to(device)  # ✅ 修复：使用 torch.device，不是 device_id

    if args.restore_ckpt is not None:
        load_ckpt(model, args.restore_ckpt)
        print(f"restore ckpt from {args.restore_ckpt}")

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # 可选，多卡同步BN

    # 如果你要启用 DDP，取消下面这行的注释，并确保正确传入 device_ids=[rank]
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], static_graph=True)

    model.train()
    train_loader = fetch_dataloader(args, rank=rank, world_size=world_size, use_ddp=use_ddp)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    VAL_FREQ = 10000
    epoch = 0
    should_keep_training = True

    while should_keep_training:
        train_loader.sampler.set_epoch(epoch)
        epoch += 1
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            # 注意：这里假设 data_blob 中的每个元素都已经正确加载为 Tensor
            # 并且你希望将它们移动到 GPU（cuda non_blocking）
            image1, image2, flow, valid = [x.cuda(non_blocking=True) for x in data_blob]
            output = model(image1, image2, flow_gt=flow, iters=args.iters)
            loss = sequence_loss(output, flow, valid, args.gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

            if total_steps % VAL_FREQ == VAL_FREQ - 1 and rank == 0:
                PATH = f'checkpoints/{total_steps+1}_{args.name}.pth'
                torch.save(model.state_dict(), PATH)
                print(f"Saved checkpoint to {PATH}")

            if total_steps > args.num_steps:
                should_keep_training = False
                break

            total_steps += 1

    # 最终保存一次
    PATH = f'checkpoints/{args.name}.pth'
    if rank == 0:
        torch.save(model.state_dict(), PATH)
        print(f"Saved final model to {PATH}")

    return PATH

def main(rank, world_size, args, use_ddp):
    if use_ddp:
        print(f"Using DDP [rank={rank}, world_size={world_size}]")
        setup_ddp(rank, world_size)

    train(args, rank=rank, world_size=world_size, use_ddp=use_ddp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args = parse_args(parser)
    args.name += f"_exp{str(np.random.randint(100))}"

    # 初始化 DDP，返回 smp 和 world_size
    smp, world_size = init_ddp()

    if world_size > 1:
        # 多进程模式（DDP）
        from torch.multiprocessing import spawn
        spawn_ctx = spawn(main, nprocs=world_size, args=(world_size, args, True), join=True)
    else:
        # 单进程模式（普通单卡训练，无 DDP）
        main(0, 1, args, False)

    print("Done!")
