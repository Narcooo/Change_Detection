import os
import torch
from core.mmodel.ChangeNet_CAD import ChangeModel_CAD
from core.mmodel.ChangeNet_CADN import ChangeModel_CADN


def get_mymodel(architecture, **init_params):
    print(init_params)
    model = config_get[architecture](**init_params)
    return model


def mymodel_contain(architecture):
    if architecture in config_get:
        return True
    return False


config_get = {
    'ChangeModel_CAD': ChangeModel_CAD,
    'ChangeModel_CADN': ChangeModel_CADN
}


def loadpretrain(model, path):
    if os.path.exists(path) is False:
        print('No recover model weight!')
        return 0
    checkpoint = torch.load(path, map_location='cpu')

    module_model_state_dict = dict()
    # load network weights
    for item, value in checkpoint['encoder'].items():
        if (item in model.encoder.state_dict()) and (value.shape == model.encoder.state_dict()[item].shape):
            module_model_state_dict[item] = value
        else:
            print(item)
    model.encoder.load_state_dict(module_model_state_dict)

    module_model_state_dict = dict()
    # load network weights
    for item, value in checkpoint['decoder'].items():
        if (item in model.landcover_decoder.state_dict()) and (
                value.shape == model.landcover_decoder.state_dict()[item].shape):
            module_model_state_dict[item] = value
        else:
            print(item)
    model.landcover_decoder.load_state_dict(module_model_state_dict)
    return model


def loadhrnetpretrain(model, pretrainpath):
   checkpoint = torch.load(pretrainpath, map_location='cpu')
   module_model_state_dict = dict()
   model_dict = model.state_dict()
   errcount = 0
   succcount = 0
   incasecount = 0
   for item, value in checkpoint.items():
       if item in model_dict.keys():
           incasecount += 1

       if item in model_dict.keys() and value.shape == model_dict[item].shape:
           module_model_state_dict[item] = value
           succcount += 1
       else:
           errcount += 1
   print(len(model_dict), 'incasecount = ', incasecount, 'successcount = ', succcount, '  errorcount = ', errcount)
   model_dict.update(module_model_state_dict)
   model.load_state_dict(model_dict)