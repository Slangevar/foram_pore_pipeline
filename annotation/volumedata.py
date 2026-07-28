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
        self.path = f"data/image_volumes/{self.filename}.npy"

        self._image_volume = None
        self.slicer = Slicer(self._peek_shape())

    def _peek_shape(self):
        """Read the volume's shape from the .npy header without loading data."""
        header = np.load(self.path, mmap_mode="r")
        shape = header.shape
        del header
        return shape

    @property
    def image_volume(self):
        """The volume, memory-mapped and opened on first use.

        Loading is deferred because the dataset holds every volume in the
        folder, while a session slices only the ones actually selected. It is
        also memory-mapped rather than read into RAM, so only the pages a slice
        touches are faulted in.

        Both halves matter because nicegui re-executes this program on every
        page request. Constructing the dataset eagerly meant re-reading the
        whole collection each time -- about 24 GB for the 123-volume set, none
        of it released afterwards.
        """
        if self._image_volume is None:
            self._image_volume = np.load(self.path, mmap_mode="r")
        return self._image_volume

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
