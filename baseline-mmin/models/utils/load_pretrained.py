import os
import json
from .config import OptConfig

def load_from_opt_record(file_path):
    opt_content = json.load(open(file_path, 'r'))
    opt = OptConfig()
    opt.load(opt_content)
    return opt

def load_pretrained_model(model_class, checkpoints_dir, cv, gpu_ids):
    path = os.path.join(checkpoints_dir, str(cv))
    config_path = os.path.join(checkpoints_dir, 'train_opt.conf')
    config = load_from_opt_record(config_path)
    config.isTrain = False                             # teacher model should be in test mode
    config.gpu_ids = gpu_ids                       # set gpu to the same
    model = model_class(config)
    model.cuda()
    model.load_networks_cv(path)
    model.eval()
    return model
