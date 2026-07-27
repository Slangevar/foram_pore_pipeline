# Annotation interface

*Placeholder — the annotation tool will be added here.*

This is the interface used to produce the annotated slices in `data/train/` and
`data/val/`. It is currently maintained outside this repository and will be
copied in.

## Expected output format

Whatever is added here should keep producing the format the training loader
already expects, so that `src/loader.py` works unchanged:

- `images/<name>.tiff` — `768 × 768` `uint8` grayscale CT slice
- `masks/<name>.tiff`  — `768 × 768 × 3` `uint8` RGB colour-coded annotation
- `weights/<name>.tiff` — `768 × 768` `uint8`, `255` = annotated, `0` = ignore

Filenames must sort identically across the three folders — the loader pairs
them by sorted order, not by name lookup.

### Colour encoding

| Colour | RGB | Meaning |
|---|---|---|
| Red | `(230, 25, 75)` | Background |
| Yellow | `(255, 225, 25)` | Chamber (shell wall) |
| Green | `(60, 180, 75)` | Pores |
| Black | `(0, 0, 0)` | Unannotated — excluded from the loss |

Only part of each slice needs annotating; leave the rest black and set the
corresponding `weights/` pixels to `0`. In the current dataset roughly 34% of
pixels are left unannotated this way.

Save TIFFs with deflate compression (`tifffile.imwrite(..., compression="zlib")`)
to keep the repository small — it is lossless and `skimage.io.imread` reads it
transparently.
