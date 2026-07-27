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
        
        # Use utils to convert colored mask to categorical one-hot
        # utils.colored_to_categorical returns (mask_slice, weight_slice)
        # The mask_slice has num_classes channels (3 in this case)
        # It maps:
        # Index 1 -> Red [230, 25, 75] -> Background
        # Index 2 -> Green [60, 180, 75] -> Pores 
        # Index 3 -> Yellow [255, 225, 25] -> Chamber
        
        # We need 3 classes: [Background, Chamber, Pores] -> [Red, Yellow, Green]
        # So we request num_classes=3
        cat_mask, _ = utils.colored_to_categorical(mask_slice, num_classes=3)
        
        # cat_mask is (H, W, 3) corresponding to indices 1, 2, 3 of the colors list
        # Index 0 is Red (Background)
        # Index 1 is Green (Pores)
        # Index 2 is Yellow (Chamber)
        # ... Wait, checking utils.py line 242:
        # colors[1] = Red
        # colors[2] = Green
        # colors[3] = Yellow
        
        # So cat_mask[..., 0] is Red (Background)
        # cat_mask[..., 1] is Green (Pores)
        # cat_mask[..., 2] is Yellow (Chamber)
        
        # WE WANT: [Background, Chamber, Pores] -> [Red, Yellow, Green]
        # So we need to reorder: [0, 2, 1]
        
        final_mask = np.zeros_like(cat_mask)
        final_mask[..., 0] = cat_mask[..., 0] # Red -> Background
        final_mask[..., 1] = cat_mask[..., 2] # Yellow -> Chamber
        final_mask[..., 2] = cat_mask[..., 1] # Green -> Pores
        
        mask_slice = final_mask.astype(np.float32)
        
        # Handle weights (assuming binary weight slice from file is correct)
        # Or we can derive it from the image itself as done in utils
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
