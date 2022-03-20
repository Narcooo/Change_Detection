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
    model.load_state_dict({k.replace('module.model.', ''): v for k, v in state_dict.items()})
    return model


def test_model(input_path, output_path, configpath, modelpath, tta=None, change_weights=None, scales=None):
    print("Available devices:", device)

    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    assert len(configpath)==len(modelpath)

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

    with torch.no_grad():
        imlist = os.listdir(os.path.join(input_path, 'A'))
        for imname in imlist:
            print('Process -> ', imname)
            img_A_path = os.path.join(input_path, 'A', imname)
            img_B_path = os.path.join(input_path, 'B', imname)
            imname = imname.split(".")[0] + '.png'
            save_path = os.path.join(output_path, imname)
            test_dataset = Bigdata_change(img_A_path, img_B_path, save_path, cut_size=2560, edge_padding=256,
                                          overlap=0.01, max_batch=1)

            test_load = torch.utils.data.DataLoader(test_dataset, batch_size=test_dataset.getbatchsize(),
                                                    pin_memory=True, num_workers=4)
            for inedx, data in enumerate(test_load):
                print(inedx, '/', len(test_load))
                input_img = data['image'].to(device).float()
                im_xs = data['im_x']
                im_ys = data['im_y']
                
                O_change = model(input_img)
                if isinstance(O_change, (list, tuple)):
                    O_change = O_change[0]
                O_change = O_change.cpu().numpy()[:, 0, :, :]
                O_change = np.where(O_change > 0.5, 1, 0).astype('uint8')

                for i in range(O_change.shape[0]):
                    test_dataset.union_res(im_xs[i], im_ys[i], O_change[i])
            test_dataset.writeimg()
    
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

