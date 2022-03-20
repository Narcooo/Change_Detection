import os
import glob
# import rasterio
import numpy as np
import pandas as pd
from typing import Optional
import torch
from torch.utils.data import Dataset
from . import transforms
# import transforms
import cv2
import warnings
import random
from torchvision import transforms as pytorchtrans
import albumentations as A

warnings.simplefilter("ignore")

class ChangeDataset_HW_FDA(Dataset):

    def __init__(
            self,
            root_dir: str,
            ids: Optional[list] = None,
            transform_name: Optional[str] = None,
            transform_color_name: Optional[str] = None,
            model: str='train',
            exchange: bool=True,
            FDA_limit: float=0.1
    ):
        super().__init__()

        self.images1_dir = os.path.join(root_dir, 'A')
        self.images2_dir = os.path.join(root_dir, 'B')
        self.gt_dir = os.path.join(root_dir, 'label')

        self.names = ids
        self.FDA_limit = FDA_limit
        self.names = self.names
        self.model = model
        self.exchange = exchange

        self.transform_color = transforms.__dict__[transform_color_name] if transform_color_name else None
        self.transform = transforms.__dict__[transform_name] if transform_name else None

        self.tfms = pytorchtrans.Compose([pytorchtrans.ToTensor(),
                                          pytorchtrans.Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                                                 [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), ])

    def __len__(self):
        return len(self.names)

    def normimg(self, img):
        temp = img.astype(np.float32)
        temp2 = temp.T
        temp2 -= self._mean
        temp2 /= self._std
        temp = temp2.T
        return temp

    def __getitem__(self, i):
        name = self.names[i]

        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        img1 = cv2.imread(os.path.join(self.images1_dir, name + '.tif'), -1)
        img2 = cv2.imread(os.path.join(self.images2_dir, name + '.tif'), -1)
        aug = A.Compose([A.FDA([img2], p=1, beta_limit=self.FDA_limit, read_fn=lambda x: x)])
        img1 = aug(image=img1)['image']
        gt_change = cv2.imread(os.path.join(self.gt_dir, name + '.png'), -1)

        img = np.concatenate([img1, img2], -1)
        gt_change = np.where(gt_change > 0, 1, 0).astype('uint8')

        # read data sample
        sample = dict(
            id=name + '.png',
            image=img,
            mask=gt_change,
        )

        # apply augmentations
        if self.transform is not None:
            sample = self.transform(**sample)

        if self.transform_color is not None:
            sample['image'] = self.transform_color(image=sample['image'])['image']

        sample["mask"] = sample["mask"][None].astype("float32")
        sample["deep_mask"] = sample["mask"]
        sample['image'] = self.tfms(np.ascontiguousarray(sample['image']).astype("float32")/255.0).float()

        return sample


if __name__ == "__main__":

    df = pd.read_csv('/config_data/dataset/Hawei_2021/HUWEI_5folds.csv', dtype={'id': object})
    train_ids = df[df['fold'] != 1]['id'].tolist()
    valid_ids = df[df['fold'] == 1]['id'].tolist()

    dategen = ChangeDataset_HW(root_dir=r'/config_data/dataset/Hawei_2021/train',
                               ids=train_ids)

    for i, data in enumerate(dategen):
        print(np.min(data['mask']), np.max(data['mask']))

        cv2.imwrite(os.path.join('/storage/Huawei2021/scs', data['id']), (data['mask']*255).astype('uint8'))
        print(i, len(dategen), data['id'], data['image'].shape, data['mask'].shape)



# break