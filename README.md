# Beyond One-Size-Fits-All: Learned Image Compression with Explicit Coding Modes

**Min Seo Cho\*** ([minseocho@mcl.korea.ac.kr](mailto:minseocho@mcl.korea.ac.kr)), **Seungmin Jeon\*** ([seungminjeon@mcl.korea.ac.kr](mailto:seungminjeon@mcl.korea.ac.kr))

\* Equal contribution.

**Official implementation · IEEE Access**

This repository contains the official implementation of **“Beyond One-Size-Fits-All: Learned Image Compression with Explicit Coding Modes”**

[Training](#training) · [Testing](#testing) · [Datasets](#datasets) · [Pretrained Checkpoints](#pretrained-checkpoints)

## Environment

Please refer to [environment.yaml](environment.yaml) for the environment and dependency versions used in this project.

## Datasets

Please refer to [dataset_links/original_dataset_links.txt](dataset_links/original_dataset_links.txt) for dataset download links and source information.

- Test image filename lists are provided in [dataset_links/](dataset_links/).
- For BMS image preparation, manual block-mode selection, and mode-map validation, follow the [BMS dataset guide](dataset_links/BMS_dataset/README.md).
- Update the local dataset paths in [build_dataset.py](build_dataset.py) and [build_dataset_bms.py](build_dataset_bms.py) to match your directory structure before running the scripts.

## Training

Please refer to [run_train.sh](run_train.sh) for the training commands and stages for IMS and BMS. Follow the initialization and training order below.

### 1. Prepare IMS initialization checkpoints

Download the pretrained FTIC checkpoints from the [official FTIC repository](https://github.com/qingshi9974/ICLR2024-FTIC). Update `checkpoint_path` and `save_path` in [checkpoint_converter.py](checkpoint_converter.py) to match your downloaded checkpoints and output locations, then run:

```bash
python checkpoint_converter.py
```

Use the converted checkpoints to initialize IMS training.

### 2. Train IMS

Update the dataset paths, checkpoint paths, output directories, and GPU settings for your environment, then run the **Train IMS** commands in [run_train.sh](run_train.sh) in order.

### 3. Prepare BMS initialization checkpoints

After completing IMS training, convert the trained IMS checkpoints into BMS initialization checkpoints using [checkpoint_converter_bms.py](checkpoint_converter_bms.py). Update `checkpoint_path`, `save_path`, and the quality-level selection in the converter for the IMS checkpoints you want to convert, then run:

```bash
python checkpoint_converter_bms.py
```

### 4. Train BMS

Set the BMS initialization checkpoint paths in [run_train.sh](run_train.sh) to the converted checkpoints, then run the **Train BMS** commands in order. Make sure checkpoint filenames and extensions match the converter outputs.

## Testing

Please refer to [run_test.sh](run_test.sh) for the IMS and BMS evaluation commands:

- **IMS:** [eval.py](eval.py). Omit `--mode` to search all six coding modes for each image and select the minimum estimated RD cost (`bpp + λ × 255² × MSE`). Set `--mode 0` through `--mode 5` to use a fixed mode.
- **BMS:** Prepare an image, annotate its block modes with `selector.py`, and inspect the saved map with `check.py`. Pass the checked maps to [eval_bms.py](eval_bms.py) using `--mode_map_path`. Follow the [step-by-step BMS guide](dataset_links/BMS_dataset/README.md) for commands and label definitions.

Update the checkpoint paths, dataset paths, output directories, and CUDA settings in the script, then run:

```bash
bash run_test.sh
```

For IMS RD search with checkpoints named `qs0`–`qs5`, λ is inferred automatically: 0.0018, 0.0035, 0.0067, 0.0130, 0.0250, and 0.0483, respectively. For other IMS checkpoint filenames, supply `--lmbda` explicitly when using RD search.

Dataset defaults use project-relative paths under `./data/`. Place your datasets there or update the paths in the builders and evaluation commands.

Both evaluators default to actual entropy encoding and decoding (`--real True`). Use `--real False` for likelihood-based bitrate estimation, which excludes mode signaling overhead. For a quick CPU check, run either command with `--cuda False --max_images 1`. Reconstructions and y bitstreams are saved in the directory specified by `--output`.

## Pretrained Checkpoints

Pretrained checkpoint download links will be added here.

| λ | IMS checkpoint | BMS checkpoint |
| :--- | :--- | :--- |
| 0.0483 | [qs5.pth.tar](https://drive.google.com/file/d/1fh5oO0Ht8dnSH0C48RxJKwIs9TgK3aYt/view?usp=drive_link) | [qs5.pth.tar](https://drive.google.com/file/d/1j_-26_lz0kjQUHeFesq0XASkxfyxmXqW/view?usp=drive_link) |
| 0.0250 | [qs4.pth.tar](https://drive.google.com/file/d/1ttz2zJP9dFCgqFHZCKPZcEE3YyKyavY_/view?usp=drive_link) | [qs4.pth.tar](https://drive.google.com/file/d/1w1yPDotk72Q23ljptJ2x3TAKv6dnaCSq/view?usp=drive_link) |
| 0.0130 | [qs3.pth.tar](https://drive.google.com/file/d/1KK9krmRpxkiYnXxjFJ8-4JuAt6ROqyKV/view?usp=drive_link) | [qs3.pth.tar](https://drive.google.com/file/d/10TWZhCp_FI3wo_DX2kSWFCQbjeC9g4Iq/view?usp=drive_link) |
| 0.0067 | [qs2.pth.tar](https://drive.google.com/file/d/1MeykqkJs5RWzYdUaE1PMP9DqxQSpz4lP/view?usp=drive_link) | [qs2.pth.tar](https://drive.google.com/file/d/1o7WU7b1zDgXC7oWIyUVfIAlDC-zqkdtS/view?usp=drive_link) |
| 0.0035 | [qs1.pth.tar](https://drive.google.com/file/d/1QG4OeJIioHuxPYYpMk-gURYPBNgjiMRA/view?usp=drive_link) | [qs1.pth.tar](https://drive.google.com/file/d/1JDVAITmLG8apT6HP4dyYHysmpFS1JL0S/view?usp=drive_link) |
| 0.0018 | [qs0.pth.tar](https://drive.google.com/file/d/1Q8irSvJb12IRX9c2jVycvvpkznUZASh9/view?usp=drive_link) | [qs0.pth.tar](https://drive.google.com/file/d/14s6hxGcCHLk8SHUtnEfUNxI9iU6TG98c/view?usp=drive_link) |

## Acknowledgments

This implementation builds on the following projects. We thank their authors for sharing their code:

- [FTIC (ICLR 2024)](https://github.com/qingshi9974/ICLR2024-FTIC)
- [LIC_TCM (CVPR 2023)](https://github.com/jmliu206/LIC_TCM)

## License

Copyright 2026 MinSeo Cho

The original code in `src/models/multi_modules.py`, `src/models/multi_modules_bms.py`, and all Python files under `dataset_links/BMS_dataset/mode_selector/` was written by MinSeo Cho and is licensed under the MIT License. See [LICENSE.txt](LICENSE.txt) for the full license text.

The MIT License granted by MinSeo Cho applies only to code authored by MinSeo Cho. Third-party code and portions derived from other projects are excluded from this grant and remain subject to their original copyright notices, license terms, and any required permissions. Acknowledgment of a project does not grant permission to redistribute its code.
