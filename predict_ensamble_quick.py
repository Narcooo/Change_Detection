import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ttach
import addict
import getters
import cv2
import time
from torchvision import transforms as pytorchtrans
from torch.utils.data import Dataset
from typing import Optional
from training.config import parse_config
from ttach.base import Merger, Compose
from utile import Bigdata_change, get_model_config

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ChangeDataset_HW(Dataset):

    def __init__(self, root_dir,):
        super().__init__()

        self.images1_dir = os.path.join(root_dir, 'A')
        self.images2_dir = os.path.join(root_dir, 'B')
        self.names = os.listdir(self.images1_dir)
        self.tfms = pytorchtrans.Compose([pytorchtrans.ToTensor(),
                                          pytorchtrans.Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                                                 [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), ])

    def __len__(self):
        return len(self.names)


    def __getitem__(self, i):
        name = self.names[i].replace('.tif', '')

        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        img1 = cv2.imread(os.path.join(self.images1_dir, name + '.tif'), -1)
        img2 = cv2.imread(os.path.join(self.images2_dir, name + '.tif'), -1)
        img = np.concatenate([img1, img2], -1)
        # 如果不位5k*5k
        if img1.shape != (5000, 5000, 3) or img2.shape != (5000, 5000, 3):
            pass

        # read data sample
        sample = dict(
            id=name + '.png',
            image=img,
        )
        sample['image'] = self.tfms(np.ascontiguousarray(sample['image']).astype("float32")/255.0).float()
        sample['image_1'] = sample['image'][:, :2688, :2688]
        sample['image_2'] = sample['image'][:, :2688, 2312:]
        sample['image_3'] = sample['image'][:, 2312:, 2312:]
        sample['image_4'] = sample['image'][:, 2312:, :2688]
        return sample

    def getfullimg(self, result):
        outzeors = np.zeros((5000, 5000), 'uint8')
        outzeors[:2500, :2500] = result[0][:2500, :2500]
        outzeors[:2500, 2500:] = result[1][:2500, 188:]
        outzeors[2500:, 2500:] = result[2][188:, 188:]
        outzeors[2500:, :2500] = result[3][188:, :2500]
        return outzeors




class EnsembleModel_max(torch.nn.Module):
    """Ensemble of torch models, pass tensor through all models and average results"""

    def __init__(self, models: list, change_weights: list):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        changeall = None
        for index, model in enumerate(self.models):
            change = model(x)
            if isinstance(change, (list, tuple)):
                change = change[0]
            if changeall is None:
                changeall = change
            else:
                changeall = torch.max(changeall, change)
        return changeall


class SegmentationScalelist(nn.Module):
    def __init__(
            self,
            model: nn.Module,
            scales: list
    ):
        super().__init__()
        self.model = model
        self.scales = scales

    def forward(self, x_org):
        oldsize = x_org.shape[-1]
        changeall = None
        for scale in self.scales:
            x = F.interpolate(x_org, scale_factor=scale)
            x = self.model(x)
            x = F.interpolate(x, size=[oldsize, oldsize], mode="bilinear")
            if changeall is None:
                changeall = x
            else:
                changeall = torch.max(changeall, x)
        return changeall


class SegmentationTTAWrapper(nn.Module):
    def __init__(
            self,
            model: nn.Module,
            transforms: Compose,
            merge_mode: str = "mean",
            output_mask_key: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.merge_mode = merge_mode
        self.output_key = output_mask_key

    def forward(self, image: torch.Tensor):
        mc = Merger(type=self.merge_mode, n=len(self.transforms))

        for transformer in self.transforms:
            augmented_image = transformer.augment_image(image)
            change = self.model(augmented_image)
            if isinstance(change, (list, tuple)):
                change = change[0]
            change_output = transformer.deaugment_mask(change)
            mc.append(change_output)
        change = mc.result
        return change


def model_from_config(path: str, checkpoint_path: str):
    """Create model from configuration specified in config file and load checkpoint weights"""
    cfg = addict.Dict(parse_config(config=path))  # read and parse config file
    init_params = cfg.model.init_params  # extract model initialization parameters
    if "encoder_weights" in init_params:
        init_params["encoder_weights"] = None  # because we will load pretrained weights for whole model

    model = getters.get_model(architecture=cfg.model.architecture, init_params=init_params)

    state_dict = torch.load(checkpoint_path)["state_dict"]

    state_dict = {k.replace('module.model.', ''): v for k, v in state_dict.items()}
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.encoder.set_swish(memory_efficient=False)
    return model


def test_model(input_path, output_path, configpath, modelpath, tta=None, change_weights=None, scales=None):
    print("Available devices:", device)

    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    assert len(configpath) == len(modelpath)

    models = []
    for index, path in enumerate(modelpath):
        model = model_from_config(configpath[index], modelpath[index])
        if scales != None:
            model = SegmentationScalelist(model, scales)
        models.append(model)
    model = EnsembleModel_max(models, change_weights)

    if tta != None:
        model = SegmentationTTAWrapper(model, tta, merge_mode='mean')
    model = model.to(device)
    model.eval()
    print("prepar data")
    test_dataset = ChangeDataset_HW(input_path)
    test_load = torch.utils.data.DataLoader(test_dataset, batch_size=1, pin_memory=True, num_workers=4)
    print("begin predict")
    with torch.no_grad():
        for index, data in enumerate(test_load):
            im1 = data['image_1'].to(device).float()
            im2 = data['image_2'].to(device).float()
            im3 = data['image_3'].to(device).float()
            im4 = data['image_4'].to(device).float()
            imname = data['id']
            print(index, '/', len(test_load), imname, im1.shape, im2.shape, im3.shape ,im4.shape)

            resule = []
            resule.append(model(im1))
            resule.append(model(im2))
            resule.append(model(im3))
            resule.append(model(im4))
            for index, res in enumerate(resule):
                res = res.cpu().numpy()[0, 0, :, :]
                res = np.where(res > 0.5, 1, 0).astype('uint8')
                resule[index] = res

            res_full = test_dataset.getfullimg(resule)

            cv2.imwrite(os.path.join(output_path, imname[0]), res_full)




def predict_ensamble(config_lists):
    input_path = '/input_path'
    output_path = r'/output_path'

    model_path = config_lists
    print(model_path)

    config_path, model_path = get_model_config(model_path)

    tta = None
    scales = [1.5]
    start = time.time()
    test_model(input_path, output_path, config_path, model_path, tta, scales=scales)
    print('run out use time ', time.time() - start)


if __name__ == "__main__":
    input_path = '/opt/nvidia/nsight-systems-cli/d54c9c38-89c5-4863-99e3-f4b9fa26707e'
    output_path = r'/storage/Radiaipytorch_studio_change_fs/models/stage_HW_Final/effb1_change_cad_s15_FDA_alltrain/output_pre'

    model_path = ['/storage/Radiaipytorch_studio_change_fs/models/stage_HW_Final/effb1_change_cad_s15_FDA_alltrain/checkpoints/k-ep[5]-0.6408.pth']

    config_path, model_path = get_model_config(model_path)

    tta = None
    scales = [1.5]
    start = time.time()
    test_model(input_path, output_path, config_path, model_path, tta, scales=scales)
    print('run out use time ', time.time() - start)

