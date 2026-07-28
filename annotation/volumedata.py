import numpy as np
from pathlib import Path

from slicer import Slicer

"""
A VolumeData object pairs a 3-D image volume with the Slicer that cuts
arbitrarily-oriented 2-D slices out of it for annotation.
"""


class VolumeData(object):

    def __init__(self, file):

        self.filename = Path(file).stem
        self.image_volume = np.load(f"data/image_volumes/{self.filename}.npy")

        self.slicer = Slicer(self.image_volume.shape)

    # Slicer functions-----------------------------------------------------------------------------------------------------------------------

    def randomize(
        self,
        candidates=None,
        class_weights=None,
        origin_shift_range=0.3,
        sampling_mode="random",
        sampling_axis="random",
    ):
        self.slicer.randomize(
            candidates=candidates,
            class_weights=class_weights,
            origin_shift_range=origin_shift_range,
            sampling_mode=sampling_mode,
            sampling_axis=sampling_axis,
        )

    def shift_origin(self, shift_amount=[0, 0, 0]):
        self.slicer.shift_origin(shift_amount=shift_amount)

    def get_slice(self, axis=0, slice_width=256, order=0):
        return self.slicer.get_slice(
            self.image_volume, axis=axis, slice_width=slice_width, order=order
        )

    # ---------------------------------------------------------------------------------------------------------------------------------------
