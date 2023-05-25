import torch
import yaml
import os


def save_model(model, save_path, epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist, optimizer=None):
  model_to_save = model.module if hasattr(model, 'module') else model
  state_dict = {key: value for key, value in model_to_save.state_dict().items() if 'embedder' not in key}
  if optimizer is not None:
    optimizer_state_dict= optimizer.state_dict()
    checkpoint = {'epochs': epochs, \
                  'lowest_eval_loss': lowest_eval_loss,\
                  'state_dict': state_dict,\
                  'optimizer_state_dict': optimizer_state_dict,\
                  'train_loss_hist': train_loss_hist,\
                  'valid_loss_hist': valid_loss_hist
                 }
  else: 
    checkpoint = {'epochs': epochs, \
              'lowest_eval_loss': lowest_eval_loss,\
              'state_dict': state_dict,\
              'train_loss_hist': train_loss_hist,\
              'valid_loss_hist': valid_loss_hist
             }
  del state_dict
  try:
    os.makedirs("/".join(save_path.split("/")[:-1]))
  except:
    pass
  torch.save(checkpoint, save_path)
  print("Saving model at epoch {} with validation loss of {}".format(epochs,\
                                                                     lowest_eval_loss))
  return

def load_model(model, save_path, optimizer= None):
  checkpoint = torch.load(save_path)
  model_state_dict = checkpoint['state_dict']

  fine_tuning_log = f"### Loading pretrained model from {save_path}\n"
  loaded=[]
  non_load=[]
  freeze=[]
  for name, param in model.named_parameters():
    try:
      if param.requires_grad == False:  # Freeze
        freeze.append(name)
      param.data.copy_(
          model_state_dict[name].data
      )  # load from pretrained model
      loaded.append(name)
    except:
      non_load.append(name)
  fine_tuning_log += f"loaded layer: {loaded}\n"
  fine_tuning_log += f"non-loaded layer: {non_load}\n"
  fine_tuning_log += f"freezed layer: {freeze}\n"

  epochs = checkpoint["epochs"]
  lowest_eval_loss = checkpoint["lowest_eval_loss"]
  train_loss_hist = checkpoint["train_loss_hist"]
  valid_loss_hist = checkpoint["valid_loss_hist"]

  print(fine_tuning_log)
  if optimizer is not None:
    try:
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      print("### Loading optimizer state dict\n")
    except:
      pass
    return model, epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist, optimizer
  return model, epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist