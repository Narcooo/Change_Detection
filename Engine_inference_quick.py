import pycuda.driver as cuda
import numpy as np
import tensorrt as trt
import cv2
import time
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms as pytorchtrans

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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
    def __init__(self, onnx_file, batch_size=1, use_fp16=False):
        self.use_fp16 = use_fp16
        self.cfx = cuda.Device(0).make_context()  # 2. trt engine创建前首先初始化cuda上下文
        self.engine = self.load_engine(onnx_file)
        self.input_shape, self.output_shape = self.infer_shape()
        print(self.output_shape)
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

        print(self.inputs)

    def __del__(self):
        del self.inputs
        del self.outputs
        del self.stream
        self.cfx.detach()  # 2. 实例释放时需要detech cuda上下文

    def load_engine(self, engine_file):
        with open(engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())  # 反序列化

    def infer_shape(self):
        for binding in self.engine:
            print(len(self.engine), binding)
            if self.engine.binding_is_input(binding):
                input_shape = self.engine.get_binding_shape(binding)
            else:
                output_shape = self.engine.get_binding_shape(binding)
        return input_shape, output_shape

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) # * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def do_inference(self, context, bindings, inputs, outputs, stream):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    def preprocess(self, data):
        return data

    def postprocess(self, data):
        return data

    def inference(self, data):
        self.inputs[0].host = self.preprocess(data)
        self.cfx.push()  # 3. 推理前执行cfx.push()
        trt_outputs = self.do_inference(self.context, bindings=self.bindings,
                                        inputs=self.inputs,
                                        outputs=self.outputs,
                                        stream=self.stream)

        output = self.postprocess(trt_outputs)
        self.cfx.pop()  # 3. 推理后执行cfx.pop()
        return output


class ChangeDataset_HW(Dataset):

    def __init__(self, root_dir,):
        super().__init__()

        self.images1_dir = os.path.join(root_dir, 'A')
        self.images2_dir = os.path.join(root_dir, 'B')
        self.names = os.listdir(self.images1_dir)
        self.tfms = pytorchtrans.Compose([pytorchtrans.ToTensor()])

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
        sample['image'] = self.tfms(np.ascontiguousarray(sample['image']).astype("float32")).float()
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


def test_model(input_path, output_path,  model_path):
    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    tensorrt_init()  # 进程起始位置初始化cuda driver
    infer_engine = TensorRTEngine(model_path)

    print("prepar data")
    test_dataset = ChangeDataset_HW(input_path)
    test_load = torch.utils.data.DataLoader(test_dataset, batch_size=1, pin_memory=True, num_workers=2)
    print("begin predict")
    with torch.no_grad():
        for index, data in enumerate(test_load):
            im1 = data['image_1']
            im2 = data['image_2']
            im3 = data['image_3']
            im4 = data['image_4']
            imname = data['id']
            print(index, '/', len(test_load), imname, im1.shape, im2.shape, im3.shape, im4.shape)
            result = []
            result.append(infer_engine.inference(np.ascontiguousarray(im1.numpy()).astype("float32"))[0])
            result.append(infer_engine.inference(np.ascontiguousarray(im2.numpy()).astype("float32"))[0])
            result.append(infer_engine.inference(np.ascontiguousarray(im3.numpy()).astype("float32"))[0])
            result.append(infer_engine.inference(np.ascontiguousarray(im4.numpy()).astype("float32"))[0])
            for index, res in enumerate(result):
                res = res.reshape(2688, 2688).astype("float32")
                res = np.where(res > 0.5, 1, 0).astype('uint8')
                result[index] = res
            res_full = test_dataset.getfullimg(result)
            cv2.imwrite(os.path.join(output_path, imname[0]), res_full)


if __name__ == "__main__":
    input_path = '/opt/nvidia/nsight-systems-cli/d54c9c38-89c5-4863-99e3-f4b9fa26707e'
    output_path = r'/storage/Radiaipytorch_studio_change_fs/models/stage_HW_Final/effb1_change_cad_s15_FDA_alltrain/output_pre_trt'
    model_path = '/storage/Radiaipytorch_studio_change_fs/models/stage_HW_Final/effb1_change_cad_s15_FDA_alltrain/k-ep[5]-0.6408_2688.trt'
    start = time.time()
    test_model(input_path, output_path, model_path)
    print('run out use time ', time.time() - start)