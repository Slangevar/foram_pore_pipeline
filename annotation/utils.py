import glob
import shutil
import json
import numpy as np
from pathlib import Path
from skimage.io import imsave, imread

import volumedata


# Brush palette. The index is the BRUSH NUMBER in the UI, not the training
# class id -- index 0 is the unannotated "ignore" colour, and brushes 1..N
# follow. For this dataset that means brush 1 = red, 2 = green, 3 = yellow.
#
# The contract with training is the RGB VALUE, never the position here:
# analysis/utils.py decodes masks through its own CLASS_COLORS, where
# red -> background, yellow -> chamber, green -> pores. The two lists are
# deliberately in different orders. If you change an RGB value in one,
# change it in the other.
BRUSH_COLORS = np.array(
    [
        [0, 0, 0],          # unannotated / ignore
        [230, 25, 75],      # red
        [60, 180, 75],      # green
        [255, 225, 25],     # yellow
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [170, 255, 195],
    ]
)

IGNORE_COLOR = BRUSH_COLORS[0]


def load_dataset():

    image_volume_files = np.sort(glob.glob("data/image_volumes/*.npy"))

    dataset = []
    if len(image_volume_files) > 0:
        dataset = [
            volumedata.VolumeData(f)
            for f in image_volume_files
        ]

    return dataset


def get_input_size():

    input_size = 512

    train_masks = glob.glob("data/train/masks/*.tiff")

    if len(train_masks) > 0:
        mask = imread(train_masks[0])
        input_size = mask.shape[0]

    return input_size


def get_num_classes():

    num_classes = 2

    train_masks = glob.glob("data/train/masks/*.tiff")

    if len(train_masks) > 0:
        mask = imread(train_masks[0])
        num_classes = np.unique(mask.reshape(-1, mask.shape[-1]), axis=0).shape[0] - 1

    return num_classes


def save_sample(
    image_slice,
    mask_slice,
    slice_data,
    num_classes=None,
    slice_idx=None,
):

    # Stamp one pixel of every colour into the top-left corner so that each
    # class is guaranteed to be present, which keeps per-class metrics from
    # dividing by zero on slices where a class was not painted. These pixels
    # are then given zero weight below, so they never affect the loss.
    if num_classes is not None:
        for i in range(num_classes + 1):
            mask_slice[0, i, :] = BRUSH_COLORS[i]

    weight_slice = compute_weight_map(mask_slice)
    weight_slice[0, : num_classes + 1] = 0

    image_slice = np.round(image_slice).astype("uint8")
    mask_slice = np.round(mask_slice).astype("uint8")
    weight_slice = np.round(weight_slice).astype("uint8")
    input_size = image_slice.shape[0]

    config_dict = {
        "InputSize": input_size,
        "Volume": slice_data["volume"],
        "Origin": slice_data["slicer"]["Origin"],
        "RotationVector": slice_data["slicer"]["RotationVector"],
    }

    # Save training sample
    if slice_idx is None:
        slice_idx = len(glob.glob("data/train/images/*.tiff"))

    imsave(f"data/train/images/{slice_idx:04d}.tiff", image_slice)
    imsave(f"data/train/masks/{slice_idx:04d}.tiff", mask_slice)
    imsave(f"data/train/weights/{slice_idx:04d}.tiff", weight_slice)
    np.save(f"data/train/slices/{slice_idx:04d}.npy", slice_data)
    with open(f"data/train/configs/{slice_idx:04d}.json", "w") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)


# Folder and data functions -------------------------------------------------------------------------------------


def create_directories():

    Path("data/image_volumes").mkdir(parents=True, exist_ok=True)

    Path("data/train/images").mkdir(parents=True, exist_ok=True)
    Path("data/train/masks").mkdir(parents=True, exist_ok=True)
    Path("data/train/weights").mkdir(parents=True, exist_ok=True)
    Path("data/train/slices").mkdir(parents=True, exist_ok=True)
    Path("data/train/configs").mkdir(parents=True, exist_ok=True)

    if len(glob.glob("data/image_volumes/*")) == 0:
        print(
            "No volumetric data found. Place one or more 3-D .npy volumes in "
            "data/image_volumes/ and restart."
        )


def clear_annotations():

    shutil.rmtree("./data/train", ignore_errors=True)
    create_directories()


def reset_all():

    clear_annotations()


# Data representation functions -------------------------------------------------------------------------------------


def compute_weight_map(colored_mask):
    """
    Build the loss weight map for a colour-coded annotation.

    Returns 255 where any class colour was painted and 0 over the unannotated
    ignore colour, so that unpainted pixels contribute no gradient. Partial
    annotation is the intended way to use this tool -- roughly a third of the
    pixels in the current dataset are deliberately left unpainted.

    Note this deliberately does not decode class channels. The annotator only
    ever needs annotated-vs-unannotated; training decodes the classes itself
    via analysis/utils.py's CLASS_COLORS, which is ordered by class rather than by
    brush.
    """

    annotated = ~np.all(colored_mask == IGNORE_COLOR, axis=-1)

    return (annotated.astype(np.uint8)) * 255


