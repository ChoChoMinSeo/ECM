import argparse
import math
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from src.datasets.datasets import onehot_convert
from src.utils import crop, pad, str2bool


def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    if not math.isfinite(mse):
        raise ValueError('Cannot compute PSNR for non-finite image values')
    return float('inf') if mse == 0 else -10 * math.log10(mse)


def compute_msssim(a, b):
    score = ms_ssim(a, b, data_range=1.).item()
    if not math.isfinite(score):
        raise ValueError('Cannot compute MS-SSIM for non-finite image values')
    error = max(0.0, 1 - score)
    return float('inf') if error == 0 else -10 * math.log10(error)


def compute_bpp(out_net, num_pixels=None):
    if num_pixels is None:
        size = out_net['x_hat'].size()
        num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods.clamp_min(1e-9)).sum()
               / (-math.log(2) * num_pixels)
               for likelihoods in out_net['likelihoods'].values()).item()


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Evaluate IMS checkpoints.')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=str2bool)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--data', default=None)
    parser.add_argument('--mode', default=None, type=int, choices=range(6), help='IMS coding mode (0-5); omit for RD search')
    parser.add_argument('--lmbda', default=None, type=float, help='RD-search lambda; inferred for qs0-qs5 checkpoints')
    parser.add_argument('--real', default=True, type=str2bool, help='Use actual entropy coding instead of likelihood estimates')
    parser.add_argument('--output', default=None, help='Output directory for reconstructions and y bitstreams')
    parser.add_argument('--max_images', default=None, type=int, help='Evaluate only the first N images')
    args = parser.parse_args(argv)
    if args.checkpoint is None:
        args.checkpoint = '../exp_results/pretrained_models/v016_unfreeze/qs5.pth.tar'
    if args.data is None:
        args.data = './data/test/00_Kodak'
    if args.lmbda is None:
        quality = Path(args.checkpoint).name.split('.')[0]
        lambdas = dict(zip(['qs0', 'qs1', 'qs2', 'qs3', 'qs4', 'qs5'],
                           [0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483]))
        args.lmbda = lambdas.get(quality)
        if args.mode is None and args.lmbda is None:
            parser.error('Specify --lmbda for RD search with a custom checkpoint filename')
    if args.max_images is not None and args.max_images < 1:
        parser.error('--max_images must be positive')
    if args.lmbda is not None and args.lmbda <= 0:
        parser.error('--lmbda must be positive')
    return args


@torch.no_grad()
def select_mode(net, x_padded, x, padding, lmbda):
    """Choose the image mode minimizing estimated bpp + lambda * 255^2 * MSE."""
    best_mode, best_cost = None, float('inf')
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    for mode in range(net.num_class):
        idx_cls = onehot_convert([mode], max_cls=net.num_class).to(x.device)
        x_hat, y_likelihoods, z_likelihoods = net(x_padded, idx_cls)
        bpp = compute_bpp({'x_hat': x_hat, 'likelihoods': {
            'y': y_likelihoods, 'z': z_likelihoods}}, num_pixels)
        mse = (crop(x_hat.clamp(0, 1), padding) - x).square().mean().item()
        cost = bpp + lmbda * 255 ** 2 * mse
        print(f'  mode {mode}: estimated RD cost {cost:.6f}', flush=True)
        if math.isfinite(cost) and cost < best_cost:
            best_mode, best_cost = mode, cost
    if best_mode is None:
        raise RuntimeError('IMS RD search found no finite candidate.')
    return best_mode


def main(argv):
    args = parse_args(argv)
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Run with --cuda False for CPU evaluation.')
    device = torch.device('cuda:0' if args.cuda else 'cpu')
    if device.type == 'cpu':
        # Dynamic grouped convolutions require the native CPU path for stable evaluation.
        torch.backends.mkldnn.enabled = False
        torch.set_flush_denormal(True)
    data_path = Path(args.data)
    img_list = sorted(p for p in data_path.iterdir()
                      if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
    if args.max_images is not None:
        img_list = img_list[:args.max_images]
    if not img_list:
        raise ValueError(f'No images found in {data_path}')
    output_dir = Path(args.output) if args.output else (
        Path('output') / 'ims' / Path(args.checkpoint).name / data_path.name
        / ('real' if args.real else 'estimated'))
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.models.flic import FrequencyAwareTransFormer
    net = FrequencyAwareTransFormer(num_class=6, multi_channels=90, multi_channels_tca=30)
    print(f'Loading {args.checkpoint} (IMS, {device})', flush=True)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}
    net.load_state_dict(state_dict, strict=True)
    del checkpoint, state_dict
    net = net.to(device).eval()
    if args.real:
        net.update()
    totals = {'psnr': 0., 'msssim': 0., 'bpp': 0., 'time': 0.}

    for img_path in tqdm(img_list):
        with Image.open(img_path) as img:
            x = transforms.ToTensor()(img.convert('RGB')).unsqueeze(0).to(device)
        x_padded, padding = pad(x, 256)
        num_pixels = x.size(0) * x.size(2) * x.size(3)
        with torch.no_grad():
            if args.cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            mode = args.mode if args.mode is not None else select_mode(net, x_padded, x, padding, args.lmbda)
            idx_cls = onehot_convert([mode], max_cls=net.num_class).to(device)
            print(f'{img_path.name}: selected IMS mode {mode}', flush=True)
            if args.real:
                bitstream_path = str(output_dir / f'{img_path.name}_y.bin')
                encoded = net.compress(x_padded, bitstream_path, idx_cls)
                decoded = net.decompress(encoded['z_strings'], encoded['minmax'],
                                         encoded['z_shape'], bitstream_path)
                x_hat = decoded['x_hat']
                bpp = (sum(len(s) for s in encoded['z_strings']) + encoded['y_size']) * 8 / num_pixels
            else:
                x_hat, y_likelihoods, z_likelihoods = net(x_padded, idx_cls)
                bpp = compute_bpp({'x_hat': x_hat, 'likelihoods': {
                    'y': y_likelihoods, 'z': z_likelihoods}}, num_pixels)
            if args.cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            x_hat = crop(x_hat.clamp(0, 1), padding)
            if not torch.isfinite(x_hat).all() or not math.isfinite(bpp):
                raise RuntimeError(f'Non-finite evaluation output for {img_path}; check the checkpoint and model.')
            psnr = compute_psnr(x, x_hat)
            msssim = compute_msssim(x, x_hat)
            save_image(x_hat, str(output_dir / f'{img_path.name}.png'))
        totals['psnr'] += psnr
        totals['msssim'] += msssim
        totals['bpp'] += bpp
        totals['time'] += elapsed
        print(f'{img_path.name}: PSNR {psnr:.2f} dB, MS-SSIM {msssim:.2f} dB, {bpp:.3f} bpp', flush=True)

    count = len(img_list)
    print(f'images: {count}')
    print(f'average_PSNR: {totals["psnr"] / count:.2f} dB')
    print(f'average_MS-SSIM: {totals["msssim"] / count:.4f} dB')
    print(f'average_Bit-rate: {totals["bpp"] / count:.3f} bpp')
    print(f'average_time: {totals["time"] / count * 1000:.3f} ms')
    print('Bitrate includes y/z streams and coded mode information; container headers are excluded.'
          if args.real else 'Bitrate is likelihood-based; mode signaling overhead is excluded.')


if __name__ == '__main__':
    main(sys.argv[1:])
