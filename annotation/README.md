# Foram Annotator

Browser-based tool for annotating arbitrarily-oriented 2-D slices cut from 3-D
micro-CT volumes. This is what produced the labelled slices in `data/train/`
and `data/val/`.

Adapted from the upstream `interactive_unet` tool. **The model training,
inference and live-suggestion code has been removed** — that functionality is
provided by `src/` in this repository, and keeping a second, divergent copy
inside the annotator was a source of confusion. What remains is the annotation
tool alone, with no `torch` dependency.

## Install and run

```bash
pip install -r requirements.txt

# from a working directory containing data/image_volumes/*.npy
python annotation/app.py
```

The annotator itself needs only `nicegui`, `opencv-python`, `numpy`, `scipy`,
`scikit-image` and `Pillow` — no `torch`. Those are listed in the repository's
single `requirements.txt`, which also covers the training and analysis code, so
installing it pulls more than the annotator strictly needs.

Open <http://localhost:9546>.

The tool reads 3-D `.npy` volumes from `data/image_volumes/` relative to the
directory you launch it from, and writes annotated slices to `data/train/`.

## Workflow

1. A random slice is cut from a random volume at a random orientation.
2. Paint each class with the brush; leave everything you are unsure about
   **unpainted** (black) — those pixels are excluded from the loss.
3. **Save Annotation** writes the image, mask and weight map.
4. **Resample** for the next slice.

Sampling can be Random, Axially-aligned, Custom (explicit origin and rotation
vector) or Replicate (re-cut a previously saved slice configuration).

## Controls

| Input | Action |
|---|---|
| Left drag | Paint with the current class |
| Right drag | Paint with class 0 (background) |
| Shift + left drag | Pan the view |
| Mouse wheel | Brush size |
| Shift + mouse wheel | Zoom in / out |
| `c` / `v` | Next / previous class colour |
| `q` / `a` | Step the slice forwards / backwards through the volume |
| `Ctrl+Z` / `Ctrl+Y` | Undo / redo |
| `Ctrl+S` | Save annotation |

## Output format

Written to `data/train/`, paired by sorted filename:

| Path | Contents |
|---|---|
| `images/NNNN.tiff` | `uint8` grayscale slice |
| `masks/NNNN.tiff` | `uint8` RGB colour-coded annotation |
| `weights/NNNN.tiff` | `uint8`, 255 = annotated, 0 = ignore |
| `slices/NNNN.npy` | Slice geometry, for exact re-cutting |
| `configs/NNNN.json` | Human-readable geometry record |

### Colour encoding

| Colour | RGB | Class |
|---|---|---|
| Red | `(230, 25, 75)` | 0 — Background |
| Yellow | `(255, 225, 25)` | 1 — Chamber (shell wall) |
| Green | `(60, 180, 75)` | 2 — Pores |
| Black | `(0, 0, 0)` | Unannotated — excluded from the loss |

These RGB values are the interchange format between this tool and training:
`src/utils.py` decodes them via `CLASS_COLORS`, which is the single source of
truth for the encoding. If you change the palette, change it there too.

Note that the **brush order in this tool is red, green, yellow** — so the second
brush is pores and the third is chamber. Only the RGB value written into the
mask matters downstream, not the brush index.

Roughly a third of the pixels in the current dataset are deliberately left
unannotated; partial annotation is the intended way to use this tool.

## Relationship to the rest of the repository

```
annotation/  ──writes──>  data/train/{images,masks,weights}
                                │
                                └──read by──>  src/loader.py  ──>  training
```
