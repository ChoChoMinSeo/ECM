def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
def cal_size(num_params,dtype):
    if dtype == "float32":
        return (num_params/1024**2) * 4
    elif dtype == "float16":
        return (num_params/1024**2) * 2
    elif dtype == "int8":
        return (num_params/1024**2) * 1
    else:
        return -1
def model_analysis(model):
    print('# Trainable Parameters:',format(count_trainable_parameters(model),','))
    print('# Parameters:',format(count_parameters(model),','))
    print('Model Size {:.2f} MB'.format(cal_size(count_trainable_parameters(model),'float32')))
    
    
if __name__ == '__main__':
    from src.models.flic import FrequencyAwareTransFormer
    # from src.models.flic_bms import FrequencyAwareTransFormer
    from build_dataset import onehot_convert
    import torch
    
    model = FrequencyAwareTransFormer(num_class=6,multi_channels=90, multi_channels_tca=30)
    model.eval()
    model_analysis(model)
    inputs = torch.randn(1, 3, 256, 256)
    mode_label = onehot_convert([0], max_cls = 6, eta = 0.0)
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        with_flops=True,
        record_shapes=True
    ) as prof:
        with torch.no_grad():
            model(inputs,mode_label)

    # 프로파일 결과 테이블
    table = prof.key_averages().table(sort_by="flops", row_limit=10)
    print("=== FLOPs Summary (Top 10) ===")
    print(table)

    # 총 FLOPs 계산
    total_flops = sum(e.flops for e in prof.key_averages() if e.flops is not None)
    print(f"\nTotal FLOPs: {total_flops / 1e9:.2f} GFLOPs")