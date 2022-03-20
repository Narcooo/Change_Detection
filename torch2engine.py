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
import onnx
from onnxsim import simplify
import tensorrt as trt
from convert_trt.onnx2engine import TensorRTEngine, tensorrt_init

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


class Alinmodel(nn.Module):
    def __init__(
            self,
            model: nn.Module
    ):
        super().__init__()
        self.model = model
        self.tfms = pytorchtrans.Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    def forward(self, x):
        x = self.model(self.tfms(x/255.0))
        return x


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


def pytorch2onnx(model,
                 convert_shape,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False):
    """Export Pytorch model to ONNX model and verify the outputs are same
    between Pytorch and ONNX.

    Args:
        model (nn.Module): Pytorch model we want to export.
        input_shape (tuple): Use this input shape to construct
            the corresponding dummy input and execute the model.
        opset_version (int): The onnx op version. Default: 11.
        show (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the output ONNX model.
            Default: `tmp.onnx`.
        verify (bool): Whether compare the outputs between Pytorch and ONNX.
            Default: False.
    """

    example_tensor = torch.randn(convert_shape[0], convert_shape[1], convert_shape[2], convert_shape[3], device='cuda')

    with torch.no_grad():
        torch.onnx.export(
            model, example_tensor,
            output_file,
            export_params=True,
            keep_initializers_as_inputs=True,
            verbose=show,
            opset_version=opset_version)
        print(f'Successfully exported ONNX model: {output_file}')


def get_model(configpath, modelpath, tta=None, change_weights=None, scales=None):
    print("Available devices:", device)

    assert len(configpath) == len(modelpath)

    models = []
    for index, path in enumerate(modelpath):
        model = model_from_config(configpath[index], modelpath[index])
        if scales != None:
            model = SegmentationScalelist(model, scales)
        models.append(model)
    model = EnsembleModel_max(models, change_weights)
    model = Alinmodel(model)
    if tta != None:
        model = SegmentationTTAWrapper(model, tta, merge_mode='mean')
    model = model.to(device)
    model.eval()
    return model


def convert_ensamble(config_lists, output_path):
    model_path = config_lists
    print(model_path)

    config_path, model_path = get_model_config(model_path)

    tta = None
    scales = [1.5]
    start = time.time()
    model = get_model(config_path, model_path, tta, scales=scales)
    pytorch2onnx(model, output_file=output_path)
    print('run out use time ', time.time() - start)


if __name__ == "__main__":
    mroot = '/storage/Radiaipytorch_studio_change_fs/models/stage_HW_Final/effb1_change_cad_s15_FDA_alltrain'
    
    output_onnxpath = os.path.join(mroot, 'k-ep[5]-0.6408_2688.onnx')
    output_trtpath = os.path.join(mroot, 'k-ep[5]-0.6408_2688.trt')
    model_path = [os.path.join(mroot, 'checkpoints/k-ep[5]-0.6408.pth')]

    batch_size = 1
    use_fp16 = True
    convert_shape = [1, 6, 2688, 2688]
    tta = None
    scales = [1.5]

    config_path, model_path = get_model_config(model_path)
    print("============================== 1/3 convert onnx")
    start = time.time()
    model = get_model(config_path, model_path, tta, scales=scales)
    pytorch2onnx(model, output_file=output_onnxpath, convert_shape=convert_shape)
    print('convert onnx use time ', time.time() - start)

    # simple
    print("============================== 2/3 onnx simple")
    start = time.time()
    onnx_model = onnx.load(output_onnxpath)  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, output_onnxpath)
    print('onnx simple use time ', time.time() - start)


    # convert to engine
    print("============================== 3/3 onnx to trt")
    start = time.time()
    tensorrt_init()  # 进程起始位置初始化cuda driver
    infer_engine = TensorRTEngine(output_onnxpath, output_trtpath, batch_size, use_fp16)
    print('onnx to trt use time ', time.time() - start)
    print("finish, engine save to ", output_trtpath)

