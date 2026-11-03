# Copyright 2026 MinSeo Cho

import torch
import torch.distributed as dist
from tqdm import tqdm
from src.utils import mkfulldir, cal_psnr, make_diff_heatmap, save_colored_segmentation_masks_blend
from src.datasets.datasets import onehot_convert
from src.loss import R_Loss, D_Loss, compute_aux_loss
from torchvision.utils import save_image
from torch.nn.functional import interpolate
import pandas as pd

def test(test_dataloader, device, model, args, log_keys_test: list, task_name:str,
         meters_dict_test: dict, meters_dict_test_per_task: dict, epoch: int, result_df: pd.DataFrame, est = True, mode = 'nat'):
    r_loss, d_loss = R_Loss().to(args.device), D_Loss().to(args.device)
    for num_batch, batch in enumerate(tqdm((test_dataloader))):
        dist.barrier()
        x, cls_idx = batch
        x = x.to(device)
        if not est:
            if mode == 'nat':
                cls_idx = onehot_convert([0],args.n_classes,args.eta).unsqueeze(0)
            elif mode == 'doc':
                cls_idx = onehot_convert([1],args.n_classes,args.eta).unsqueeze(0)
        else:
            mode = 'est'
        # cls_idx = onehot_convert([0],args.n_classes,args.eta).unsqueeze(0)

        x_hat, y_likelihoods, z_likelihoods, mode_maps,_ = model(x, cls_idx, return_mode = True, est=est)

        loss_distortion = d_loss(x_hat, x)
        psnr = cal_psnr(x_hat, x)

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
                    if task_name in k or k == "Total Loss":
                        meters_dict_test[k].update(float(out_criterion["Total Loss"])) 
                elif "Rate Loss" in k:
                    if k == 'Rate Loss y' or k == 'Rate Loss z':
                        meters_dict_test[k].update(float(out_criterion[k])) 
                    elif task_name in k or k == "Rate Loss":
                        meters_dict_test[k].update(float(out_criterion["Rate Loss"])) 
                elif "PSNR" in k:
                    if task_name in k or k == "PSNR":
                        meters_dict_test[k].update(float(out_criterion["PSNR"])) 
                else:
                    meters_dict_test[k].update(float(out_criterion[k])) 
                if k in out_criterion.keys():
                    meters_dict_test_per_task[task_name][k].update(float(out_criterion[k]))
                
            # if args.save_qualitative and num_batch<10 and task_name =='Modemap':
            if args.save_qualitative:
                saveroot = mkfulldir(args.test_result_dir +
                    f"/epoch{epoch:03d}_{mode}/{args.qs}/{task_name}/{num_batch}")
                save_image(x, saveroot + f"/0x_orig.png")
                save_image(x_hat, saveroot + f"/1x_hat_{psnr:.3f}.png")
                save_image(make_diff_heatmap(x, x_hat),
                                    saveroot + f"/2err.png")
                bitsmap_y = (-torch.log2(y_likelihoods)).sum(dim=1, keepdim=True)
                bitsmap_z = (
                        interpolate(-torch.log2(z_likelihoods), scale_factor=4) / 16).sum(
                    dim=1, keepdim=True)
                bitsmap_mv = interpolate(bitsmap_y + bitsmap_z, scale_factor=16)
                bitsmap_mv /= bitsmap_mv.max()
                save_image(bitsmap_mv,
                            saveroot + f"/3bitsmap_{float(loss_rate):.4f}.png")
                if mode_maps[0] is not None:
                    save_colored_segmentation_masks_blend(interpolate(mode_maps[0], scale_factor=2*args.multi_patch_size2, mode='nearest'), saveroot + f"/4mode_map_enc1.png")
                    save_colored_segmentation_masks_blend(interpolate(mode_maps[1], scale_factor=4*args.multi_patch_size4, mode='nearest'), saveroot + f"/4mode_map_enc2.png")
                    save_colored_segmentation_masks_blend(interpolate(mode_maps[2], scale_factor=2*args.multi_patch_size2, mode='nearest'), saveroot + f"/4mode_map_dec1.png")
                    save_colored_segmentation_masks_blend(interpolate(mode_maps[3], scale_factor=4*args.multi_patch_size4, mode='nearest'), saveroot + f"/4mode_map_dec2.png")
                    
        new_row = pd.DataFrame([{
            'total loss': loss_total.item(),
            'rate loss': loss_rate.item(),
            'rate loss y': loss_rate_y.item(),
            'rate loss z': loss_rate_z.item(),
            'psnr': psnr,
        }])
        result_df = pd.concat([result_df,new_row],ignore_index=True)
    return meters_dict_test, meters_dict_test_per_task, result_df