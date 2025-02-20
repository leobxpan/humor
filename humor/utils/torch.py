
import sys, os, time
import torch
import torch.nn as nn
import numpy as np

def get_device_from_gpu_arg(gpu_arg):
    '''
    Given the --gpu flag argument list, parses it and 
    returns the correct device.
    -1 : all available gpus
    2 : the gpu at idx 2
    0 2 4 : the gpus at indexes 0, 2, and 4
    Returns (pytorch_device, is_parallel=True if using multiple GPUs)
    '''
    gpu_idx = None
    gpu_arg = [gpu_arg]
    if len(gpu_arg) == 1:
        gpu_idx = gpu_arg[0]
        if gpu_idx != -1:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
            print('Attempting to use GPU: ' + str(gpu_idx))
            gpu_idx = 0
    else:
        # using multiple GPUs, make them visible and set to use as many as possible
        gpu_list_str = ','.join([str(cur_gpu) for cur_gpu in gpu_arg])
        print('Attempting to use GPUs: ' + gpu_list_str)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list_str
        gpu_idx = -1
    
    device = None
    is_parallel = False
    if gpu_idx == -1:
        device = get_device(0)
        is_parallel = True
    else:
        device = get_device(gpu_idx)

    return device, is_parallel

def get_device(gpu_idx=0):
    '''
    Returns the pytorch device for the given gpu index.
    '''
    gpu_device_str = 'cuda:%d' % (gpu_idx)
    device_str = gpu_device_str if torch.cuda.is_available() else 'cpu'
    if device_str == gpu_device_str:
        print('Using detected GPU...')
        device_str = 'cuda:0'
    else:
        print('No detected GPU...using CPU.')
    device = torch.device(device_str)
    return device

def torch_to_numpy(tensor_list):
    return [x.to('cpu').data.numpy() for x in tensor_list]

def torch_to_scalar(tensor_list):
    return [x.to('cpu').item() for x in tensor_list]

copy2cpu = lambda tensor: tensor.detach().cpu().numpy()

def save_state(file_out, model, optimizer, cur_epoch=0, min_val_loss=float('Inf'), min_train_loss=float('Inf'), ignore_keys=None):
    model_state_dict = model.state_dict()
    if ignore_keys is not None:
        model_state_dict = {k: v for k, v in model_state_dict.items() if k.split('.')[0] not in ignore_keys}

    full_checkpoint_dict = {
        'model' : model_state_dict,
        'optim' : optimizer.state_dict(),
        'epoch' : cur_epoch,
        'min_val_loss' : min_val_loss,
        'min_train_loss' : min_train_loss
    }
    torch.save(full_checkpoint_dict, file_out)

def load_state(load_path, model, optimizer=None, is_parallel=False, map_location=None, ignore_keys=None):
    if not os.path.exists(load_path):
        print('Could not find checkpoint at path ' + load_path)

    full_checkpoint_dict = torch.load(load_path, map_location=map_location)
    model_state_dict = full_checkpoint_dict['model']
    optim_state_dict = full_checkpoint_dict['optim']

    # load model weights
    for k, v in model_state_dict.items():
        if k.split('.')[0] == 'module' and not is_parallel:
            # then it was trained with Data parallel
            print('Loading weights trained with DataParallel...')
            model_state_dict = {'.'.join(k.split('.')[1:]) : v for k, v in model_state_dict.items() if k.split('.')[0] == 'module'}
        break
    
    if ignore_keys is not None:
        model_state_dict = {k: v for k, v in model_state_dict.items() if k.split('.')[0] not in ignore_keys}
        
    # overwrite entries in the existing state dict
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    if ignore_keys is not None:
        missing_keys = [k for k in missing_keys if k.split('.')[0] not in ignore_keys]
        unexpected_keys = [k for k in unexpected_keys if k.split('.')[0] not in ignore_keys]
    if len(missing_keys) > 0:
        print('WARNING: The following keys could not be found in the given state dict - ignoring...')
        print(missing_keys)
    if len(unexpected_keys) > 0:
        print('WARNING: The following keys were found in the given state dict but not in the current model - ignoring...')
        print(unexpected_keys)

    # load optimizer weights
    if optimizer is not None:
        optimizer.load_state_dict(optim_state_dict)

    min_train_loss = float('Inf')
    if 'min_train_loss' in full_checkpoint_dict.keys():
        min_train_loss = full_checkpoint_dict['min_train_loss']

    return full_checkpoint_dict['epoch'], full_checkpoint_dict['min_val_loss'], min_train_loss

class DataParallelPassthrough(nn.DataParallel):
    ''' Wrapper to allow passthrough of attributes'''
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def save_state(file_out, model, optimizer, cur_epoch=0, min_val_loss=float('Inf'), min_train_loss=float('Inf'), ignore_keys=None):
    model_state_dict = model.state_dict()
    if ignore_keys is not None:
        model_state_dict = {k: v for k, v in model_state_dict.items() if k.split('.')[0] not in ignore_keys}

    full_checkpoint_dict = {
        'model' : model_state_dict,
        'optim' : optimizer.state_dict(),
        'epoch' : cur_epoch,
        'min_val_loss' : min_val_loss,
        'min_train_loss' : min_train_loss
    }
    torch.save(full_checkpoint_dict, file_out)

def load_state(load_path, model, optimizer=None, is_parallel=False, map_location=None, ignore_keys=None):
    if not os.path.exists(load_path):
        print('Could not find checkpoint at path ' + load_path)

    full_checkpoint_dict = torch.load(load_path, map_location=map_location)
    model_state_dict = full_checkpoint_dict['model']
    optim_state_dict = full_checkpoint_dict['optim']

    # load model weights
    for k, v in model_state_dict.items():
        if k.split('.')[0] == 'module' and not is_parallel:
            # then it was trained with Data parallel
            print('Loading weights trained with DataParallel...')
            model_state_dict = {'.'.join(k.split('.')[1:]) : v for k, v in model_state_dict.items() if k.split('.')[0] == 'module'}
        break
    
    if ignore_keys is not None:
        model_state_dict = {k: v for k, v in model_state_dict.items() if k.split('.')[0] not in ignore_keys}
        
    # overwrite entries in the existing state dict
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    if ignore_keys is not None:
        missing_keys = [k for k in missing_keys if k.split('.')[0] not in ignore_keys]
        unexpected_keys = [k for k in unexpected_keys if k.split('.')[0] not in ignore_keys]
    if len(missing_keys) > 0:
        print('WARNING: The following keys could not be found in the given state dict - ignoring...')
        print(missing_keys)
    if len(unexpected_keys) > 0:
        print('WARNING: The following keys were found in the given state dict but not in the current model - ignoring...')
        print(unexpected_keys)

    # load optimizer weights
    if optimizer is not None:
        optimizer.load_state_dict(optim_state_dict)

    min_train_loss = float('Inf')
    if 'min_train_loss' in full_checkpoint_dict.keys():
        min_train_loss = full_checkpoint_dict['min_train_loss']

    return full_checkpoint_dict['epoch'], full_checkpoint_dict['min_val_loss'], min_train_loss
