import argparse
import math
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from src.utils import crop, pad, str2bool
from eval import compute_bpp, compute_msssim, compute_psnr


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Evaluate BMS checkpoints with manually selected block modes.')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=str2bool)
    parser.add_argument('--checkpoint', default='../exp_results/pretrained_models/v312_6modes/qs5.pth.tar')
    parser.add_argument('--data', default='./data/BMS/images')
    parser.add_argument('--mode_map_path', default=None, help='Directory containing one <image stem>.pt mode map per image')
    parser.add_argument('--lmbda', default=None, type=float, help='RD-search lambda; inferred for qs0-qs5 checkpoints')
    parser.add_argument('--real', default=True, type=str2bool)
    parser.add_argument('--output', default=None)
    parser.add_argument('--max_images', default=None, type=int)
    args = parser.parse_args(argv)
    if args.lmbda is None:
        quality = Path(args.checkpoint).name.split('.')[0]
        lambdas = dict(zip(['qs0', 'qs1', 'qs2', 'qs3', 'qs4', 'qs5'],
                           [0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483]))
        args.lmbda = lambdas.get(quality)
        if args.lmbda is None and args.mode_map_path is None:
            parser.error('Specify --lmbda for RD search with a custom checkpoint filename')
    if args.max_images is not None and args.max_images < 1:
        parser.error('--max_images must be positive')
    if args.lmbda is not None and args.lmbda <= 0:
        parser.error('--lmbda must be positive')
    return args


def load_mode_map(path, image_shape, device):
    """Load selector maps (natural/document/prediction) or native 6/7-channel maps."""
    modes = torch.as_tensor(torch.load(path, map_location='cpu', weights_only=True))
    height, width = image_shape
    if height % 256 or width % 256:
        raise ValueError('Pad images to multiples of 256 before creating their mode maps.')
    if modes.ndim == 3:
        if modes.shape[0] not in (2, 3, 6, 7) or tuple(modes.shape[1:]) != (height, width):
            raise ValueError(f'{path}: expected a 2/3/6/7-channel one-hot map matching the image')
        if not torch.all((modes == 0) | (modes == 1)) or not torch.all(modes.sum(dim=0) == 1):
            raise ValueError(f'{path}: every pixel must have exactly one selected mode')
        labels = modes.argmax(dim=0).long()
        if modes.shape[0] == 3:
            # The selector's third label is prediction, not the Residue coding mode.
            labels[labels == 2] = 6
    elif modes.ndim == 2 and tuple(modes.shape) == (height, width):
        if not torch.isfinite(modes).all() or not torch.equal(modes, modes.long()):
            raise ValueError(f'{path}: native mode IDs must be integers')
        labels = modes.long()
    else:
        raise ValueError(f'{path}: mode map dimensions must match the image')
    if labels.min() < 0 or labels.max() > 6:
        raise ValueError(f'{path}: native mode IDs must be in 0-6')
    blocks = labels[::16, ::16]
    if not torch.equal(blocks.repeat_interleave(16, 0).repeat_interleave(16, 1), labels):
        raise ValueError(f'{path}: each 16x16 block must have a single mode')
    return blocks.unsqueeze(0).unsqueeze(0).to(device).contiguous()


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
        Path('output') / 'bms' / Path(args.checkpoint).name / data_path.name
        / ('real' if args.real else 'estimated'))
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.models.flic_bms import FrequencyAwareTransFormer
    net = FrequencyAwareTransFormer(num_class=6, multi_channels=90, multi_channels_tca=30)
    print(f'Loading {args.checkpoint} (BMS, {device})', flush=True)
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
        selected_map = None
        if args.mode_map_path is not None:
            map_path = Path(args.mode_map_path) / f'{img_path.stem}.pt'
            selected_map = load_mode_map(map_path, tuple(x.shape[-2:]), device)
            print(f'{img_path.name}: using {map_path}', flush=True)
        with torch.no_grad():
            if args.cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            if args.real:
                bitstream_path = str(output_dir / f'{img_path.name}_y.bin')
                encoded = net.compress(x_padded, bitstream_path, lmbda=args.lmbda, mode_map=selected_map)
                decoded = net.decompress(encoded['z_strings'], encoded['minmax'],
                                         encoded['z_shape'], bitstream_path)
                x_hat = decoded['x_hat']
                bpp = (sum(len(s) for s in encoded['z_strings']) + encoded['y_size']) * 8 / num_pixels
            else:
                mode_map = selected_map
                if mode_map is None:
                    mode_map = net.forward_rd_search_compress(x_padded, lmbda=args.lmbda)
                x_hat, y_likelihoods, z_likelihoods = net(x_padded, mode_map)
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
