import torch
from src.models.flic import FrequencyAwareTransFormer
from count_parameter import model_analysis

lmbdas = ['0483','0250','0130','0067','0035','0018']
num_class = 6
multi_channels = 90
multi_channels_tca = 30

for lmbda in lmbdas:
    checkpoint_path = f'../exp_results/pretrained_models/FAT_pretrained_models/ckpt_{lmbda}_re.pth'
    save_path = f'../exp_results/pretrained_models/FAT_pretrained_models/ckpt_{lmbda}_IMS_init.pth'
    checkpoint = torch.load(checkpoint_path)
    new_checkpoint = torch.load(checkpoint_path)
    model = FrequencyAwareTransFormer(num_class=num_class, multi_channels=multi_channels, multi_channels_tca=multi_channels_tca)
    model_analysis(model)
    with torch.no_grad():
        for name, param in model.named_parameters():
            new_checkpoint['state_dict'][name] = param
            if name in checkpoint['state_dict'].keys() and checkpoint['state_dict'][name].shape == param.shape:
                new_checkpoint['state_dict'][name] = checkpoint['state_dict'][name]
            elif 'mlp' in name:
                if 'fc1' in name: # input
                    if 'main_weight' in name:
                        cur_data = name.replace('.main_weight','')+'.weight'
                        new_checkpoint['state_dict'][name] = checkpoint['state_dict'][cur_data].unsqueeze(0)
                    elif 'multi_weight' in name:
                        cur_data = name.replace('.multi_weights','')+'.weight'
                        for i in range(5):
                            new_checkpoint['state_dict'][name][:,:,multi_channels_tca*i:multi_channels_tca*(i+1)] = checkpoint['state_dict'][cur_data][256*i:256*(i+1),list(range(multi_channels_tca))].unsqueeze(0).repeat(num_class-1,1,1,1,1)
                    elif 'main_bias' in name:
                        cur_data = name.replace('.main_bias','')+'.bias'
                        new_checkpoint['state_dict'][name] = checkpoint['state_dict'][cur_data].unsqueeze(0)
                    elif 'multi_bias' in name:
                        cur_data = name.replace('.multi_bias','')+'.bias'
                        new_checkpoint['state_dict'][name] = checkpoint['state_dict'][cur_data].unsqueeze(0).repeat(num_class-1,1)
                elif 'fc2' in name: # output
                    if 'main_weight' in name:
                        cur_data = name.replace('.main_weight','')+'.weight'
                        new_checkpoint['state_dict'][name] = checkpoint['state_dict'][cur_data].unsqueeze(0)
                    elif 'multi_weight' in name:
                        cur_data = name.replace('.multi_weights','')+'.weight'
                        for i in range(5):
                            new_checkpoint['state_dict'][name][:,:,multi_channels_tca*i:multi_channels_tca*(i+1)] = checkpoint['state_dict'][cur_data][256*i:256*(i+1),list(range(multi_channels_tca))].unsqueeze(0).repeat(num_class-1,1,1,1,1)
                    elif 'main_bias' in name:
                        cur_data = name.replace('.main_bias','')+'.bias'
                        new_checkpoint['state_dict'][name] = checkpoint['state_dict'][cur_data].unsqueeze(0)
                    elif 'multi_bias' in name:
                        cur_data = name.replace('.multi_bias','')+'.bias'
                        new_checkpoint['state_dict'][name] = checkpoint['state_dict'][cur_data].unsqueeze(0).repeat(num_class-1,1)
                else:
                    print(name)
            elif 'fat' in name: # output
                if 'conv' in name:
                    if 'main_weight' in name:
                        cur_data = name.replace('.main_weight','')+'.weight'
                        new_checkpoint['state_dict'][name] = checkpoint['state_dict'][cur_data].unsqueeze(0)
                    elif 'multi_weight' in name:
                        cur_data = name.replace('.multi_weights','')+'.weight'
                        new_checkpoint['state_dict'][name] = checkpoint['state_dict'][cur_data][:,list(range(multi_channels))].unsqueeze(0).repeat(num_class-1,1,1,1,1)
                    elif 'main_bias' in name:
                        cur_data = name.replace('.main_bias','')+'.bias'
                        new_checkpoint['state_dict'][name] = checkpoint['state_dict'][cur_data].unsqueeze(0)
                    elif 'multi_bias' in name:
                        cur_data = name.replace('.multi_bias','')+'.bias'
                        new_checkpoint['state_dict'][name] = checkpoint['state_dict'][cur_data].unsqueeze(0).repeat(num_class-1,1)  
                else:
                    if 'main_weight' in name:
                        cur_data = name.replace('.main_weight','')+'.weight'
                        new_checkpoint['state_dict'][name] = checkpoint['state_dict'][cur_data].unsqueeze(0)
                    elif 'multi_weight' in name:
                        cur_data = name.replace('.multi_weights','')+'.weight'
                        new_checkpoint['state_dict'][name] = checkpoint['state_dict'][cur_data][:,list(range(multi_channels))].unsqueeze(0).repeat(num_class-1,1,1)
                    elif 'main_bias' in name:
                        cur_data = name.replace('.main_bias','')+'.bias'
                        new_checkpoint['state_dict'][name] = checkpoint['state_dict'][cur_data].unsqueeze(0)
                    elif 'multi_bias' in name:
                        cur_data = name.replace('.multi_bias','')+'.bias'
                        new_checkpoint['state_dict'][name] = checkpoint['state_dict'][cur_data].unsqueeze(0).repeat(num_class-1,1) 
            else:
                print(name)
                
    torch.save(new_checkpoint, save_path)
            