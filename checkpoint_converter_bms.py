import torch
from src.models.flic_bms import FrequencyAwareTransFormer
from count_parameter import model_analysis

lmbdas = ['0483','0250','0130','0067','0035','0018']
lmbdas.reverse()
num_class = 6
multi_channels = 90
multi_channels_tca = 30
for idx, lmbda in enumerate(lmbdas):
    if idx != 5:
        continue
    checkpoint_path = f'../exp_results/pretrained_models/IMS/qs{idx}.pth.tar'
    save_path = f'../exp_results/pretrained_models/FAT_pretrained_models/ckpt_{lmbda}_BMS_init_6modes.pth.tar'
    checkpoint = torch.load(checkpoint_path)
    new_checkpoint = torch.load(checkpoint_path)
    model = FrequencyAwareTransFormer(num_class=num_class, multi_channels=multi_channels, multi_channels_tca=multi_channels_tca, patch_size = 16)
    model_analysis(model)
    with torch.no_grad():
        for name, param in model.named_parameters():
            new_checkpoint['state_dict'][name] = param
            if name in checkpoint['state_dict'].keys() and checkpoint['state_dict'][name].shape == param.shape:
                new_checkpoint['state_dict'][name] = checkpoint['state_dict'][name]
            elif 'multi' in name:
                new_checkpoint['state_dict'][name]= checkpoint['state_dict'][name][:num_class-1]
                
    # model.load_state_dict(new_checkpoint['state_dict'],strict=True)
    torch.save(new_checkpoint, save_path)
            