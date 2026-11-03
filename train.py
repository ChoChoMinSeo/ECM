# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
import argparse
import shutil
import sys
from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import pandas as pd

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from src.loss import R_Loss, D_Loss, compute_aux_loss
from src.utils_train import load_checkpoints, net_aux_optimizer

from src.models.flic import FrequencyAwareTransFormer
from count_parameter import model_analysis
from src.utils import mkfulldir, str2bool, AverageMeter, cal_psnr, fix_seed,make_diff_heatmap
from build_dataset import *
from torchvision.utils import save_image
from torch.nn.functional import interpolate

torch.backends.cudnn.benchmark = True
import warnings
warnings.filterwarnings('ignore')

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("--dir", type=str, default="debug", )
    parser.add_argument("--devices", type=str, default="0,", )
    
    parser.add_argument("--wk", type=str, default="local", )
    parser.add_argument("--tcp", default="3001", type=str, )
    parser.add_argument("--train_len_per_task", default=100, type=int, help="Number of epochs (default: %(default)s)", )
    parser.add_argument("--tasks", type=str, nargs="*", default=("Kodak",'Y1Y2Y3','Residue','Document','KITTI','LineDrawing'))
    parser.add_argument("--test_tasks", type=str, nargs="*", default=("Kodak",'Y1Y2Y3','Residue','Document','KITTI','LineDrawing'))
    
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs (default: %(default)s)", )
    parser.add_argument("--subsample", default=1, type=int )
    
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate (default: %(default)s)", )
    parser.add_argument("--reset_lr", default=None, type=float, help="Learning rate (default: %(default)s)", )
    parser.add_argument("--reset_aux_lr", default=None, type=float, help="Learning rate (default: %(default)s)", )
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloaders threads (default: %(default)s)", )
    parser.add_argument("--lmbda", type=float, nargs="*", default=(0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483))
    parser.add_argument("--qs", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (default: %(default)s)" )
    parser.add_argument("--train_subsample", type=int, default=None, help="Batch size (default: %(default)s)")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Test batch size (default: %(default)s)", )
    parser.add_argument("--aux_learning_rate", type=float, default=1e-3, help="Auxiliary loss learning rate (default: %(default)s)", )
    parser.add_argument("--patch_size", type=int, nargs=2, default=(256, 256), help="Size of the patches to be cropped (default: %(default)s)", )
    parser.add_argument("--cuda", type=str2bool, default=f"True")
    parser.add_argument("--save_qualitative", type=str2bool, default="False")
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--clip_max_norm", default=10.0, type=float, help="gradient clipping max norm (default: %(default)s", )
    parser.add_argument("--checkpoint", type=str, nargs="*", help="Path to a checkpoint")
    parser.add_argument("--checkpoint_only_weight", type=str2bool, default=f"False")
    parser.add_argument("--log_freq", type=int, default=50, )
    parser.add_argument("--test_only", type=str2bool, default=f"False")
    
    parser.add_argument("--natu", type=int, default=1, )
    parser.add_argument("--resi", type=int, default=1, )
    parser.add_argument("--docu", type=int, default=1, )
    parser.add_argument("--y1y2", type=int, default=1, )
    parser.add_argument("--kitt", type=int, default=1, )
    parser.add_argument("--line", type=int, default=1, )
        
    parser.add_argument("--multi_channels", type=int, default=10, )    
    parser.add_argument("--multi_channels_tca", type=int, default=3, )    
    
    parser.add_argument("--n_classes", default=1, type=int )
    
    parser.add_argument("--DDP", type=str2bool, default="True", )
    parser.add_argument("--freeze_base", type=str2bool, default="True", )
    
    parser.add_argument("--world_size", type=int, default=1, )
    parser.add_argument("--gpu", type=int, default=0, )

    args = parser.parse_args(argv)
    return args


class CustomDataParallel(torch.nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "AdamW", "lr": args.learning_rate},
        "aux": {"type": "AdamW", "lr": args.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


def save_checkpoint(state, is_best, filedir="", filename="checkpoint", suffix=""):
    torch.save(state, filename + '.pth.tar')
    if is_best:
        shutil.copyfile(filename + '.pth.tar', filedir + '/best_loss' + suffix + '.pth.tar')


def ddp(argv):
    args = parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    
    if args.dir == 'debug':
        main(gpu = 0, ngpus_per_node=ngpus_per_node,args=args)
    else:
        mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, args,))


def main(gpu, ngpus_per_node, args):
    torch.cuda.set_device(gpu)

    if args.seed is not None:
        fix_seed(args.seed)

    if args.DDP:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=f'tcp://127.0.0.1:{args.tcp}',
            world_size=ngpus_per_node,
            rank=gpu)

    if args.dir == "debug":
        args.num_workers = 0

    args.dir = os.path.join(f"../exp_results", args.dir)
    args.ckpt_dir = args.dir + "/checkpoint"
    args.log_dir = args.ckpt_dir + "/log"
    mkfulldir(args.log_dir)
    args.tmp_result_dir = args.ckpt_dir + "/tmp_results"
    mkfulldir(args.tmp_result_dir)
    args.test_result_dir = args.tmp_result_dir + "/test"
    mkfulldir(args.test_result_dir)
        
        
    args.tasks = tuple(args.tasks)
    if 'Kodak' in args.test_tasks:
        args.test_tasks = list(args.test_tasks) + ['natural_clic']
    args.test_tasks = list(args.test_tasks)
    args.test_tasks = tuple(args.test_tasks)
    
    train_dataset = get_train_dataset(
        args,
        subsample=args.subsample,
        max_len=args.train_len_per_task
    )
    test_dataset = get_test_dataset(
        args,
    )
    test_dataset2 = get_test_dataset2(
        args,
    )
    if args.DDP:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed, drop_last=True)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    args.device = device

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False if args.DDP else True,
        pin_memory=(device == "cuda"),
        sampler=train_sampler if args.DDP else None,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    test_dataloader2 = DataLoader(
        test_dataset2,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    model = FrequencyAwareTransFormer(num_class=args.n_classes, multi_channels=args.multi_channels, multi_channels_tca=args.multi_channels_tca).to(args.device)
    if args.freeze_base:
        for name,param in model.named_parameters():
            if 'multi' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    model_analysis(model)
                
    model = DDP(model, device_ids=[gpu], output_device=[gpu], find_unused_parameters=True)

    optimizer, aux_optimizer = net_aux_optimizer(model, args)

    r_loss, d_loss = R_Loss().to(args.device), D_Loss().to(args.device)

    log_keys_train = [
        "Total Loss",
        "Rate Loss",
        "Distortion Loss",
        "Aux Loss",
        "Rate Loss y",
        "Rate Loss z",
        "PSNR",
    ]

    log_keys_test = [
        "Total Loss",
        "Rate Loss",
        "Distortion Loss",
        "Aux Loss",
        "Rate Loss y",
        "Rate Loss z",
        "PSNR",
        "Best Loss",
        "Best Epoch"
    ]
    meters_dict_train = {k: AverageMeter() for k in log_keys_train}
    meters_dict_test_per_task = {}

    for task_name in args.test_tasks:
        meters_dict_test_per_task[f"{task_name}"] = {k: AverageMeter() for k in log_keys_test[:-2]}
    log_keys_test = log_keys_test[:-2]
    for task_name in args.test_tasks:
        log_keys_test.append(f'Total Loss {task_name}')
    log_keys_test += ["Best Loss","Best Epoch"]
    meters_dict_test = {k: AverageMeter() for k in log_keys_test[:-2]}
        
    model, optimizer, aux_optimizer, _, last_epoch, global_step, best_loss, best_epoch, log_train, log_test = load_checkpoints(
        args,
        model,
        optimizer,
        aux_optimizer,
        None,
        log_keys_train,
        log_keys_test,
        len(train_dataloader)
    )
    
    if args.reset_lr is not None:
        optimizer.param_groups[0]["lr"] = args.reset_lr
    if args.reset_aux_lr is not None:
        aux_optimizer.param_groups[0]["lr"] = args.reset_aux_lr
    if args.aux_learning_rate == 0:
        aux_optimizer = None
        
    if args.test_only:
        last_epoch = 0
        args.epochs = 1
    for epoch in range(last_epoch, args.epochs):
        # --- train one epoch ---
        if not args.test_only:
            for i in range(len(train_dataloader.dataset.datasets)):
                train_dataloader.dataset.datasets[i].set_epoch()
            train_sampler.set_epoch(epoch)
            
            model.train()
            device = next(model.parameters()).device

            for i, batch in enumerate(tqdm(train_dataloader, desc=f"{args.dir} Epoch [{epoch} / {args.epochs}]")):
                dist.barrier()      
                  
                x, x_gt, cls_idx = batch
                x = x.to(device)
                x_gt = x_gt.to(device)
                
                optimizer.zero_grad(set_to_none=True)
                if aux_optimizer is not None:
                    aux_optimizer.zero_grad(set_to_none=True)

                x_hat, y_likelihoods, z_likelihoods = model(x,cls_idx)
                
                loss_distortion = d_loss(x_hat, x_gt)
                psnr = cal_psnr(x_hat, x_gt)
                loss_distortion = loss_distortion.mean()
                
                num_pixels = x.shape[0] * x.shape[2] * x.shape[3]
                loss_rate, loss_rate_y, loss_rate_z = r_loss(num_pixels, y_likelihoods, z_likelihoods)
                loss_total = loss_distortion * (255**2) * args.lmbda[args.qs] + loss_rate 
                loss_total.backward()
                if args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
                optimizer.step()
                aux_loss = compute_aux_loss([model.module.aux_loss() if args.DDP else model.aux_loss()], backward=True)
                if aux_optimizer is not None:
                    aux_optimizer.step()

                if dist.get_rank() == args.gpu:
                    out_criterion = {
                        "Total Loss": loss_total,
                        "Rate Loss": loss_rate,
                        "Distortion Loss": loss_distortion,
                        "Aux Loss": aux_loss,
                        "Rate Loss y": loss_rate_y,
                        "Rate Loss z": loss_rate_z,
                        "PSNR": psnr,
                    }

                    for k in log_keys_train:
                        meters_dict_train[k].update(float(out_criterion[k]))

                    if (global_step + 1) % args.log_freq == 0:
                        log = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}\t" \
                              f"{epoch}\t" \
                              f"{global_step}\t" \
                              f"{optimizer.param_groups[0]['lr']}\t"

                        for k in log_keys_train:
                            log += f"{meters_dict_train[k].avg()}\t"
                        log = log[:-1]
                        log_train.logging(log)

                        for k in log_keys_train:
                            meters_dict_train[k].init()

                global_step += 1
            
        # --- test one epoch ---
        model.eval()
        device = next(model.parameters()).device

        for k in log_keys_test[:-2]:
            meters_dict_test[k].init()
            if k in log_keys_train:
                for task_name in meters_dict_test_per_task.keys():
                    meters_dict_test_per_task[task_name][k].init()
        result_df = pd.DataFrame({
            'total loss':[],
            'rate loss': [],
            'rate loss y': [],
            'rate loss z': [],
            'psnr': [],
        })
        with torch.no_grad():
            cur_task = -1
            for num_batch, batch in enumerate(tqdm((test_dataloader))):
                dist.barrier()
                x, x_gt, cls_idx = batch
                x = x.to(device)
                x_gt = x_gt.to(device)

                x_hat, y_likelihoods, z_likelihoods = model(x, cls_idx)

                loss_distortion = d_loss(x_hat, x_gt)
                psnr = cal_psnr(x_hat, x_gt)

                num_pixels = x.shape[0] * x.shape[2] * x.shape[3]
                loss_rate, loss_rate_y, loss_rate_z = r_loss(num_pixels, y_likelihoods, z_likelihoods)
                loss_total = loss_distortion *(255**2) * args.lmbda[args.qs]+ loss_rate 
                
                aux_loss = 0
                aux_loss = compute_aux_loss([model.module.aux_loss() if args.DDP else model.aux_loss()], backward=False)

                if dist.get_rank() == args.gpu:
                    out_criterion = {
                        "Total Loss": loss_total,
                        "Rate Loss": loss_rate,
                        "Distortion Loss": loss_distortion,
                        "Aux Loss": aux_loss,
                        "Rate Loss y": loss_rate_y,
                        "Rate Loss z": loss_rate_z,
                        "PSNR": psnr,
                    }

                    for k in log_keys_test[:-2]:
                        tmp = torch.argmax(cls_idx).item()
                        cur_task_name = f'{dataset_name[tmp]}'
                            
                        if "Total Loss" in k:
                            if k == f"Total Loss {cur_task_name}" or k == "Total Loss":
                                meters_dict_test[k].update(float(out_criterion["Total Loss"])) 
                        elif "Rate Loss" in k:
                            if k == 'Rate Loss y' or k == 'Rate Loss z':
                                meters_dict_test[k].update(float(out_criterion[k])) 
                            elif k == f"Rate Loss {cur_task_name}" or k == "Rate Loss":
                                meters_dict_test[k].update(float(out_criterion["Rate Loss"])) 
                        elif "PSNR" in k:
                            if k == f"PSNR {cur_task_name}" or k == "PSNR":
                                meters_dict_test[k].update(float(out_criterion["PSNR"])) 
                        else:
                            meters_dict_test[k].update(float(out_criterion[k])) 
                        if k in out_criterion.keys():
                            meters_dict_test_per_task[cur_task_name][k].update(float(out_criterion[k]))
                        
                if args.save_qualitative:
                    saveroot = mkfulldir(args.test_result_dir +
                        f"/epoch{epoch:03d}/{token_idx_to_name[tmp]}/{num_batch}")
                    save_image(x, saveroot + f"/0x_orig.png")
                    save_image(x_hat, saveroot + f"/1x_hat_{psnr:.3f}.png")
                    save_image(make_diff_heatmap(x, x_hat),
                                        saveroot + f"/2err_warp_orig.png")
                    bitsmap_y = (-torch.log2(y_likelihoods)).sum(dim=1, keepdim=True)
                    bitsmap_z = (
                            interpolate(-torch.log2(z_likelihoods), scale_factor=4) / 16).sum(
                        dim=1, keepdim=True)
                    bitsmap_mv = interpolate(bitsmap_y + bitsmap_z, scale_factor=16)
                    bitsmap_mv /= bitsmap_mv.max()
                    save_image(bitsmap_mv,
                                saveroot + f"/3bitsmap_{float(loss_rate):.4f}.png")
                result_df.loc[num_batch] = {
                    'total loss': loss_total.item(),
                    'rate loss': loss_rate.item(),
                    'rate loss y': loss_rate_y.item(),
                    'rate loss z': loss_rate_z.item(),
                    'psnr': psnr,
                }  
            if dist.get_rank() == args.gpu:
                loss = meters_dict_test["Total Loss"].avg()

                is_best = loss < best_loss
                best_loss = min(loss, best_loss)
                if is_best:
                    best_epoch = epoch

                log = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}\t" \
                    f"{epoch}\t" \
                    f"{global_step}\t" \
                    f"{optimizer.param_groups[0]['lr']}\t"
        len_test1 = num_batch+1           
        if 'natural_clic' in args.test_tasks:
            with torch.no_grad():
                cur_task = -1
                for num_batch, batch in enumerate(tqdm((test_dataloader2))):
                    dist.barrier()
                    x, x_gt, cls_idx = batch
                    x = x.to(device)
                    x_gt = x_gt.to(device)

                    x_hat, y_likelihoods, z_likelihoods = model(x, cls_idx)

                    loss_distortion = d_loss(x_hat, x_gt)
                    psnr = cal_psnr(x_hat, x_gt)

                    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]
                    loss_rate, loss_rate_y, loss_rate_z = r_loss(num_pixels, y_likelihoods, z_likelihoods)
                    loss_total = loss_distortion *(255**2) * args.lmbda[args.qs]+ loss_rate 
                    
                    aux_loss = 0
                    aux_loss = compute_aux_loss([model.module.aux_loss() if args.DDP else model.aux_loss()], backward=False)

                    if dist.get_rank() == args.gpu:
                        out_criterion = {
                            "Total Loss": loss_total,
                            "Rate Loss": loss_rate,
                            "Distortion Loss": loss_distortion,
                            "Aux Loss": aux_loss,
                            "Rate Loss y": loss_rate_y,
                            "Rate Loss z": loss_rate_z,
                            "PSNR": psnr,
                        }

                        for k in log_keys_test[:-2]:
                            if "Total Loss" in k:
                                if 'natural_clic' in k or k == "Total Loss":
                                    meters_dict_test[k].update(float(out_criterion["Total Loss"])) 
                            elif "Rate Loss" in k:
                                if k == 'Rate Loss y' or k == 'Rate Loss z':
                                    meters_dict_test[k].update(float(out_criterion[k])) 
                                elif 'natural_clic' in k or k == "Rate Loss":
                                    meters_dict_test[k].update(float(out_criterion["Rate Loss"])) 
                            elif "PSNR" in k:
                                if 'natural_clic' in k or k == "PSNR":
                                    meters_dict_test[k].update(float(out_criterion["PSNR"])) 
                            else:
                                meters_dict_test[k].update(float(out_criterion[k])) 
                            if k in out_criterion.keys():
                                meters_dict_test_per_task['natural_clic'][k].update(float(out_criterion[k]))
                            
                    if args.save_qualitative:
                        saveroot = mkfulldir(args.test_result_dir +
                            f"/epoch{epoch:03d}/natural_clic/{num_batch}")
                        save_image(x, saveroot + f"/0x_orig.png")
                        save_image(x_hat, saveroot + f"/1x_hat_{psnr:.3f}.png")
                        save_image(make_diff_heatmap(x, x_hat),
                                            saveroot + f"/2err_warp_orig.png")
                        bitsmap_y = (-torch.log2(y_likelihoods)).sum(dim=1, keepdim=True)
                        bitsmap_z = (
                                interpolate(-torch.log2(z_likelihoods), scale_factor=4) / 16).sum(
                            dim=1, keepdim=True)
                        bitsmap_mv = interpolate(bitsmap_y + bitsmap_z, scale_factor=16)
                        bitsmap_mv /= bitsmap_mv.max()
                        save_image(bitsmap_mv,
                                    saveroot + f"/3bitsmap_{float(loss_rate):.4f}.png")
                    result_df.loc[len_test1 + num_batch] = {
                        'total loss': loss_total.item(),
                        'rate loss': loss_rate.item(),
                        'rate loss y': loss_rate_y.item(),
                        'rate loss z': loss_rate_z.item(),
                        'psnr': psnr,
                    }         
                if dist.get_rank() == args.gpu:
                    loss = meters_dict_test["Total Loss"].avg()

                    is_best = loss < best_loss
                    best_loss = min(loss, best_loss)
                    if is_best:
                        best_epoch = epoch

                    log = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}\t" \
                        f"{epoch}\t" \
                        f"{global_step}\t" \
                        f"{optimizer.param_groups[0]['lr']}\t"
        
        result_df.to_csv(os.path.join(args.test_result_dir, f"epoch{epoch:03d}qs{args.qs}_by_image.csv"))
        for k in log_keys_test[:-2]:
            log += f"{meters_dict_test[k].avg()}\t"
        
        log += f"{best_loss}\t{best_epoch}"

        log_test.logging(log)

        filedir = args.ckpt_dir
        filename = args.ckpt_dir + '/' + str(epoch).zfill(3)
        
        fig, axes = plt.subplots(1, 1, figsize=(7, 7))
        rates = ''
        psnrs = ''
        for i,task_name in enumerate(args.test_tasks):
            cur_rate = meters_dict_test_per_task[task_name]['Rate Loss'].avg()
            cur_psnr = meters_dict_test_per_task[task_name]['PSNR'].avg()
            axes.plot(cur_rate,cur_psnr,"o--", c = task_colors[task_name],label = task_name)
            try:
                axes.plot(initial_token_stats[task_name]['rate'],initial_token_stats[task_name]['psnr'],"*--", c = task_colors[task_name])
                axes.plot(trained_token_stats[task_name]['rate'],trained_token_stats[task_name]['psnr'],"*-", c = task_colors[task_name])
                
            except:0
            rates += f'{cur_rate:.4f}/ '
            psnrs += f'{cur_psnr:.4f}/ '
        axes.set_xlabel(rates)
        axes.set_ylabel(psnrs)
        axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),ncol=3)
        fig.tight_layout()
        axes.grid(True)
        fig.savefig(os.path.join(args.test_result_dir, f"epoch{epoch:03d}qs{args.qs}.png"))
        fig.clf()
        plt.close()  
        
        df = {}
        for task_name in args.test_tasks:
            cur_loss = meters_dict_test_per_task[task_name]['Total Loss'].avg()
            cur_rate = meters_dict_test_per_task[task_name]['Rate Loss'].avg()
            cur_psnr = meters_dict_test_per_task[task_name]['PSNR'].avg()
            cur_rate_y = meters_dict_test_per_task[task_name]['Rate Loss y'].avg()
            cur_rate_z = meters_dict_test_per_task[task_name]['Rate Loss z'].avg()
            df[task_name] = {
                'avg total loss':cur_loss,
                'avg rate loss':cur_rate,
                'avg rate loss y':cur_rate_y,
                'avg rate loss z':cur_rate_z,
                'avg psnr':cur_psnr
            }
        df = pd.DataFrame(df).transpose()
        df.to_csv(os.path.join(args.test_result_dir, f"epoch{epoch:03d}qs{args.qs}.csv"))
        
        if not args.test_only and (epoch + 1):
            save_checkpoint(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else None,
                    "loss": loss,
                    "best_loss": best_loss,
                    "best_epoch": best_epoch,
                },
                is_best,
                filedir=filedir,
                filename=filename
            )
            
        if args.test_only:
            break


if __name__ == "__main__":
    ddp(sys.argv[1:])
