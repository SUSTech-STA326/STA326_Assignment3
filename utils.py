"""
    Some handy functions for pytroch model training ...
"""
import torch
import os


# import os
# import torch

def save_checkpoint(model, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    try:
        checkpoint_path = os.path.join(model_dir, 'model_checkpoint.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved successfully at: {checkpoint_path}")
    except Exception as e:
        print(f"Error saving model checkpoint: {e}")

# Call save_checkpoint(model, 'checkpoints') to save the model



def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), 
                                                          lr=params['adam_lr'],
                                                          weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer