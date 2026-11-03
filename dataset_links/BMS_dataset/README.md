# Preparing BMS Images and Manual Mode Maps

Use the tools in [mode_selector/](mode_selector/) to prepare an image, select its block modes manually, inspect the saved map, and evaluate BMS with that map. Run the commands below from the repository root.

## 1. Prepare the image

Use PNG images with height and width divisible by 256. Annotate the exact image that will be passed to the evaluator. If necessary, pad your input first:

```bash
python dataset_links/BMS_dataset/mode_selector/preprocess.py \
  --image ./data/raw/example.png \
  --output ./data/BMS/images/example.png
```

This adds centered white padding to reach multiples of 256 and saves a lossless PNG. If the image already has the required dimensions, place it directly in `./data/BMS/images/`.

## 2. Select block modes manually

Open the prepared image with [selector.py](mode_selector/selector.py):

```bash
python dataset_links/BMS_dataset/mode_selector/selector.py \
  --image ./data/BMS/images/example.png \
  --output ./data/BMS/modes/example.pt
```

The selector uses a 16×16 pixel grid and the following labels:

| Selector label | Meaning | Overlay |
| :--- | :--- | :--- |
| 0 | Natural-image coding mode | Dark |
| 1 | Document coding mode | Original image |
| 2 | Predict the mode at this location | Yellow |

- Click a cell or drag over a rectangle to cycle the selected cells through the three labels. New maps start with label 1.
- Press **S** to save the mode map, then **ESC** to close the window.
- Reopen the same image and output path to continue editing an existing map.
- For large images, add `--scale 0.5` to reduce the displayed size; annotations still use the original image coordinates.

Each saved `.pt` file contains a three-channel, pixel-resolution one-hot tensor of shape `(3, H, W)`. Every 16×16 block has one label. The evaluator maps selector label 2 to the six-mode model's prediction marker, ID 6. The prediction module is used at locations marked for prediction; the other locations use the manually selected coding mode.

The interactive tools require OpenCV GUI support and a desktop display.

## 3. Check the saved map

Inspect the overlay with [check.py](mode_selector/check.py):

```bash
python dataset_links/BMS_dataset/mode_selector/check.py \
  --image ./data/BMS/images/example.png \
  --mode_map ./data/BMS/modes/example.pt
```

Confirm that the natural-image, document, and prediction regions match your intended selection. Press **ESC** to close the window, or **S** to save the overlay.

For validation and an overlay PNG without opening a window:

```bash
python dataset_links/BMS_dataset/mode_selector/check.py \
  --image ./data/BMS/images/example.png \
  --mode_map ./data/BMS/modes/example.pt \
  --output ./output/example_modes.png --no_gui
```

## 4. Evaluate BMS with the map

Keep each image and its map under the same filename stem:

```text
data/BMS/
├── images/
│   └── example.png
└── modes/
    └── example.pt
```

Pass both directories to [eval_bms.py](../../eval_bms.py):

```bash
python eval_bms.py --cuda True --real True \
  --checkpoint ../exp_results/pretrained_models/v312_6modes/qs5.pth.tar \
  --data ./data/BMS/images \
  --mode_map_path ./data/BMS/modes \
  --output ./output/BMS_qs5
```

The evaluator loads `<image stem>.pt` for each image and uses the supplied block choices during encoding. Those choices are encoded in the bitstream, so decoding uses the recorded choices. Keep the image and mode-map dimensions aligned; after changing the image size or padding, create a matching map again.

Use `--real False` for likelihood-based evaluation or `--max_images 1` for a one-image check. Checkpoint download links and general setup instructions are provided in the [main README](../../README.md).
