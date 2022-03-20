import os
from predict_ensamble import predict_ensamble
from training.config import parse_config
import addict

config_dir = 'config/Temp_test'
data_dir = '/train'
config_list = os.listdir(config_dir)

for config_name in config_list:
    print("CUDA_VISIBLE_DEVICES=0 python train.py --configs " + os.path.join(config_dir, config_name) + " --data_path " + data_dir)
    os.system("CUDA_VISIBLE_DEVICES=0 python train.py --configs " + os.path.join(config_dir, config_name) + " --data_path " + data_dir)



model_pathes = []
for config_name in config_list:
    cfg = addict.Dict(parse_config(config=os.path.join(config_dir, config_name)))
    model_pathes.append(os.path.join(cfg.logdir, 'checkpoints/last.pth'))
                        
predict_ensamble(model_pathes)
