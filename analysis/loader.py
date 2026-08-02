import glob
import numpy as np
from skimage import io
from scipy import ndimage
import albumentations as A
from . import utils


from torch.utils.data import Dataset, DataLoader

def get_data_loader(data_folder, batch_size, shuffle=True, augment=False, num_workers=4):

    dataset = SegmentationDataset(data_folder, augment=augment)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        pin_memory=True,
                        persistent_workers=(num_workers > 0))

    return loader

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=0.1, rotate=(-45, 45), p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        # Reverted ElasticTransform to heavy setting
        A.ElasticTransform(alpha=100, sigma=10, p=0.3),
        A.GridDistortion(p=0.3),
    ], additional_targets={'weight': 'mask'})


class SegmentationDataset(Dataset):

    def __init__(self, data_folder, augment=False):
        
        image_filenames = np.sort(glob.glob(f'{data_folder}/images/*'))    
        mask_filenames = np.sort(glob.glob(f'{data_folder}/masks/*'))    
        weight_filenames = np.sort(glob.glob(f'{data_folder}/weights/*'))

        self.data = []
        for i in range(len(image_filenames)):
            self.data.append([io.imread(image_filenames[i]),
                              io.imread(mask_filenames[i]),
                              io.imread(weight_filenames[i])])
        np.random.shuffle(self.data)

        self.augment = augment
        if self.augment:
            self.transform = get_train_transform()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image_slice, mask_slice, weight_slice = self.data[idx]

        # Normalize image
        image_slice = image_slice / 255
        
        # Colour-coded RGB annotation -> one-hot (H, W, 3) in CLASS_NAMES order:
        # [background, chamber, pores]. utils.CLASS_COLORS is already in class
        # order, so the channels need no reordering here.
        # The second return value (a weight map derived from the unannotated
        # colour) is discarded: the explicit weights/*.tiff files are used instead.
        mask_slice, _ = utils.colored_to_categorical(mask_slice, num_classes=3)
        mask_slice = mask_slice.astype(np.float32)

        # Weight map is binary in the file (255 = annotated, 0 = ignore)
        weight_slice = weight_slice / 255
	            
        # Augment sample
        if self.augment:
            augmented = self.transform(image=image_slice, mask=mask_slice, weight=weight_slice)
            image_slice = augmented['image']
            mask_slice = augmented['mask']
            weight_slice = augmented['weight']

        image_slice = (np.expand_dims(image_slice, 0)).astype(np.float32)

        mask_slice = (np.moveaxis(mask_slice, -1, 0)).astype(np.float32)
        weight_slice = (np.expand_dims(weight_slice, 0)).astype(np.float32)

        # Make weight into multi-class class for softmax
        weight_slice = np.array([weight_slice[0] for i in range(mask_slice.shape[0])])

        # print(image_slice.shape, np.min(image_slice), np.max(image_slice))
        # print(mask_slice.shape, np.min(mask_slice), np.max(mask_slice))
        # print(weight_slice.shape, np.min(weight_slice), np.max(weight_slice))

        return (image_slice, mask_slice, weight_slice)
