# Train IMS
 python ./train.py --num_workers 4 --batch_size 8 --wk local --devices 0 --learning_rate 1e-4\
 --checkpoint_only_weight True --checkpoint ../exp_results/pretrained_models/FAT_pretrained_models/ckpt_0483_IMS_init.pth\
 --dir IMS_qs5_freeze --qs 5 --train_subsample 1 --test_only False --save_qualitative False --epochs 10 --tcp 5002\
 --train_len_per_task 2000 --freeze_base True --n_classes 6\
 --multi_channels 90 --multi_channels_tca 30\
 --tasks Y1Y2Y3 Residue Document KITTI LineDrawing --test_tasks Kodak Y1Y2Y3 Residue Document KITTI LineDrawing\
 --docu 1 --resi 2 --y1y2 2 --kitt 2 --line 2
 
python ./train.py --num_workers 4 --batch_size 8 --wk local --devices 0 --learning_rate 1e-5\
 --checkpoint_only_weight False --checkpoint ../exp_results/IMS_qs5_freeze/checkpoint/009.pth.tar\
 --dir IMS_qs5_freeze --qs 5 --train_subsample 1 --test_only False --save_qualitative False --epochs 80 --tcp 5002\
 --train_len_per_task 2000 --freeze_base True --n_classes 6\
 --multi_channels 90 --multi_channels_tca 30\
 --tasks Y1Y2Y3 Residue Document KITTI LineDrawing --test_tasks Kodak Y1Y2Y3 Residue Document KITTI LineDrawing\
 --docu 1 --resi 2 --y1y2 2 --kitt 2 --line 2
 
 python ./train.py --num_workers 4 --batch_size 8 --wk local --devices 0 --learning_rate 1e-5\
 --checkpoint_only_weight True --checkpoint ../exp_results/IMS_qs5_freeze/checkpoint/best_loss.pth.tar\
 --dir IMS_qs5 --qs 5 --train_subsample 1 --test_only False --save_qualitative False --epochs 20 --tcp 5002\
 --train_len_per_task 2000 --freeze_base False --n_classes 6\
 --multi_channels 90 --multi_channels_tca 30\
 --tasks Kodak Y1Y2Y3 Residue Document KITTI LineDrawing --test_tasks Kodak Y1Y2Y3 Residue Document KITTI LineDrawing\
 --natu 10 --docu 1 --resi 2 --y1y2 2 --kitt 2 --line 2

 
 # Train BMS
  python ./train_bms.py --num_workers 4 --batch_size 1 --wk local --devices 0 --learning_rate 1e-4\
 --checkpoint_only_weight True --checkpoint ../exp_results/pretrained_models/FAT_pretrained_models/ckpt_0483_BMS_init_6modes.pth\
 --dir BMS_6modes_mode_estimator --qs 5 --train_subsample 1 --test_only False --save_qualitative False --epochs 5 --tcp 5002\
 --train_len_per_task 200 --freeze_base True --freeze_multi True --freeze_mode False --n_classes 6\
 --multi_channels 90 --multi_channels_tca 30 --cutmix_prob 1.0 --rd_lmbda 0.1\
 --multi_patch_size 16 --patch_size 128 128\
 --tasks Kodak Y1Y2Y3 Residue Document KITTI LineDrawing --test_tasks Kodak\
 --natu 1 --docu 1 --resi 2 --y1y2 2 --kitt 2 --line 2

 python ./train_bms.py --num_workers 4 --batch_size 1 --wk local --devices 0 --learning_rate 1e-5\
 --checkpoint_only_weight True --checkpoint ../exp_results/BMS_6modes_mode_estimator/checkpoint/004.pth.tar\
 --dir BMS_6modes_mode_e2e --qs 5 --train_subsample 1 --test_only False --save_qualitative False --epochs 10 --tcp 5002\
 --train_len_per_task 100 --freeze_base True --freeze_multi False --freeze_mode False --n_classes 6\
 --multi_channels 90 --multi_channels_tca 30 --cutmix_prob 1.0 --rd_lmbda 0.1\
 --multi_patch_size 16 --patch_size 128 128\
 --tasks Kodak Y1Y2Y3 Residue Document KITTI LineDrawing --test_tasks Document\
 --natu 1 --docu 1 --resi 2 --y1y2 2 --kitt 2 --line 2