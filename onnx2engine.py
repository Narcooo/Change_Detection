import pycuda.driver as cuda
import numpy as np
import tensorrt as trt
import cv2
import os


def tensorrt_init():  # 1. 子进程开始初始化cuda driver
    cuda.init()


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


TRT_LOGGER = trt.Logger()


class TensorRTEngine(object):
    def __init__(self, onnx_file, save_path, batch_size=1, use_fp16=False):
        self.use_fp16 = use_fp16
        self.save_path = save_path
        self.cfx = cuda.Device(0).make_context()  # 2. trt engine创建前首先初始化cuda上下文
        self.engine, self.network = self.load_engine(onnx_file, batch_size)


    def load_engine(self, onnx_file, batch_size=1):
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(
                network, TRT_LOGGER) as parser:
            builder.max_batch_size = batch_size
            builder.max_workspace_size = 1 << 30
            builder.fp16_mode = self.use_fp16
            with open(onnx_file, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))

            engine = builder.build_cuda_engine(network)
        print("Load onnx sucessful!")
        with open(self.save_path, 'wb') as f:
            f.write(engine.serialize())

        return engine, network


if __name__ == "__main__":

    onnx_path = '/storage/Huawei2021/Radiaipytorch_studio_change_onnx/models/stage_HW_Final_11_8/ensamble_max_train11_nodrop.onnx'
    simonnx_path = '/storage/Huawei2021/Radiaipytorch_studio_change_onnx/models/stage_HW_Final_11_8/ensamble_max_train11_nodrop_sim.onnx'
    save_path = '/storage/Huawei2021/Radiaipytorch_studio_change_onnx/models/stage_HW_Final_11_8/ensamble_max_train11_nodrop.trt'
    # 简化
    # import os
    # os.system('python -m onnxsim ' + onnx_path + '  ' + simonnx_path)

    batch_size = 1
    use_fp16 = True

    tensorrt_init()  # 进程起始位置初始化cuda driver
    infer_engine = TensorRTEngine(simonnx_path, save_path, batch_size, use_fp16)