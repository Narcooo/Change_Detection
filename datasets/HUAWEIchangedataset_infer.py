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

warnings.simplefilter("ignore")


def get_map(Hist):
    # 计算概率分布Pr
    sum_Hist = sum(Hist)
    Pr = Hist / sum_Hist
    # 计算累计概率Sk
    Sk = []
    temp_sum = 0
    for n in Pr:
        temp_sum = temp_sum + n
        Sk.append(temp_sum)
    Sk = np.array(Sk)
    # 计算映射关系img_map
    img_map = []
    for m in range(256):
        temp_map = int(255 * Sk[m] + 0.5)
        img_map.append(temp_map)
    img_map = np.array(img_map)
    return img_map


def get_off_map(map_):  # 计算反向映射，寻找最小期望
    map_2 = list(map_)
    off_map = []
    temp_pre = 0  # 如果循环开始就找不到映射时，默认映射为0
    for n in range(256):
        try:
            temp1 = map_2.index(n)
            temp_pre = temp1
        except BaseException:
            temp1 = temp_pre  # 找不到映射关系时，近似取向前最近的有效映射值
        off_map.append(temp1)
    off_map = np.array(off_map)
    return off_map


def get_infer_map(infer_img):
    infer_Hist_b = cv2.calcHist([infer_img], [0], None, [256], [0, 255])
    infer_Hist_g = cv2.calcHist([infer_img], [1], None, [256], [0, 255])
    infer_Hist_r = cv2.calcHist([infer_img], [2], None, [256], [0, 255])
    infer_b_map = get_map(infer_Hist_b)
    infer_g_map = get_map(infer_Hist_g)
    infer_r_map = get_map(infer_Hist_r)
    infer_b_off_map = get_off_map(infer_b_map)
    infer_g_off_map = get_off_map(infer_g_map)
    infer_r_off_map = get_off_map(infer_r_map)
    return [infer_b_off_map, infer_g_off_map, infer_r_off_map]


def get_finalmap(org_map, infer_off_map):  # 计算原始图像到最终输出图像的映射关系
    org_map = list(org_map)
    infer_off_map = list(infer_off_map)
    final_map = []
    for n in range(256):
        temp1 = org_map[n]
        temp2 = infer_off_map[temp1]
        final_map.append(temp2)
    final_map = np.array(final_map)
    return final_map


def get_newimg(img_org, org2infer_maps):
    w, h, _ = img_org.shape
    b, g, r = cv2.split(img_org)
    for i in range(w):
        for j in range(h):
            temp1 = b[i, j]
            b[i, j] = org2infer_maps[0][temp1]
    for i in range(w):
        for j in range(h):
            temp1 = g[i, j]
            g[i, j] = org2infer_maps[1][temp1]
    for i in range(w):
        for j in range(h):
            temp1 = r[i, j]
            r[i, j] = org2infer_maps[2][temp1]
    newimg = cv2.merge([b, g, r])
    return newimg


def get_new_img(img_org, infer_map):
    org_Hist_b = cv2.calcHist([img_org], [0], None, [256], [0, 255])
    org_Hist_g = cv2.calcHist([img_org], [1], None, [256], [0, 255])
    org_Hist_r = cv2.calcHist([img_org], [2], None, [256], [0, 255])
    org_b_map = get_map(org_Hist_b)
    org_g_map = get_map(org_Hist_g)
    org_r_map = get_map(org_Hist_r)
    org2infer_map_b = get_finalmap(org_b_map, infer_map[0])
    org2infer_map_g = get_finalmap(org_g_map, infer_map[1])
    org2infer_map_r = get_finalmap(org_r_map, infer_map[2])
    return get_newimg(img_org, [org2infer_map_b, org2infer_map_g, org2infer_map_r])


class ChangeDataset_HW_Infer(Dataset):
    def __init__(
            self,
            root_dir: str,
            ids: Optional[list] = None,
            transform_name: Optional[str] = None,
            transform_color_name: Optional[str] = None,
            model: str='train',
            exchange: bool=True
    ):
        super().__init__()

        self.images1_dir = os.path.join(root_dir, 'A')
        self.images2_dir = os.path.join(root_dir, 'B')
        self.gt_dir = os.path.join(root_dir, 'label')

        self.names = ids

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
        infer_map = get_infer_map(img2)  # 计算参考映射关系
        img1 = get_new_img(img1, infer_map)
        gt_change = cv2.imread(os.path.join(self.gt_dir, name + '.png'), -1)


        # if self.model == 'train' and self.exchange:
        #     # 随机数据交换
        #     if random.random() < 0.5:
        #         img1, img2 = img2, img1
        #         gt1, gt2 = gt2, gt1
        #     # 随机平移

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
        sample['image'] = self.tfms(np.ascontiguousarray(sample['image']).astype("float32")).float()

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