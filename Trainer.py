import os
import math
import yaml
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
import torch
import torch.nn as nn
from models import *
from dataset import *
from model_utils import *

def fit(model, num_epochs,\
          optimizer,\
          train_dataloader, valid_dataloader,\
          model_save_path,\
          train_loss_set=[], valid_loss_set = [],\
          lowest_eval_loss=None, start_epoch=0,\
          device="cuda"
          ):
  model.to(device)

  # trange is a tqdm wrapper around the normal python range
  for i in trange(num_epochs, desc="Epoch"):
    # if continue training from saved model
    actual_epoch = start_epoch + i
    """
    Training
    """
    # Set our model to training mode (as opposed to evaluation mode)
    model.train()
    # Tracking variables
    tr_loss = 0
    as_loss= 0
    se_loss=0
    num_train_samples = 0

    # Train the data for one epoch
    for step, batch in enumerate(train_dataloader):
      # Add batch to GPU
      batch = tuple(t.to(device) for t in batch)
      # Unpack the inputs from our dataloader
      b_input_ids, b_input_mask, b_labels = batch
      # Clear out the gradients (by default they accumulate)
      optimizer.zero_grad()
      # Forward pass
      output = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
      # store train loss
      tr_loss += output['loss'].item()
      as_loss += output['aspect_loss'].item()
      se_loss += output['sent_loss'].item()
      num_train_samples += b_labels.size(0)

      # Backward pass
      output['loss'].backward()

      #for name, param in model.sentiment_fcs.named_parameters():
          #if param.requires_grad:
            #print(param.register_hook(hook_fn))

      #clipping_value = 1 # arbitrary value of your choosing
      #torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)
      # Update parameters and take a step using the computed gradient
      optimizer.step()
      # scheduler.step()

    # Update tracking variables
    epoch_train_loss = tr_loss/num_train_samples
    epoch_aspect_loss = as_loss/num_train_samples
    epoch_sentiment_loss = se_loss/num_train_samples
    train_loss_set.append([epoch_train_loss, epoch_aspect_loss, epoch_sentiment_loss])
    print("Train loss: total - {}, classifier - {}, sentiment - {}".format(epoch_train_loss, epoch_aspect_loss, epoch_sentiment_loss))

    """
    Validation
    """
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()
    # Tracking variables 
    eval_loss = 0
    as_loss = 0
    se_loss = 0
    num_eval_samples = 0

    # Evaluate data for one epoch
    for batch in valid_dataloader:
      # Add batch to GPU
      batch = tuple(t.to(device) for t in batch)
      # Unpack the inputs from our dataloader
      b_input_ids, b_input_mask, b_labels = batch
      # Telling the model not to compute or store gradients,
      # saving memory and speeding up validation
      with torch.no_grad():
        # Forward pass, calculate validation loss
        output = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        # store valid loss
        eval_loss += output['loss'].item()
        as_loss += output['aspect_loss'].item()
        se_loss += output['sent_loss'].item()
        num_eval_samples += b_labels.size(0)

    # Update tracking variables
    epoch_eval_loss = eval_loss/num_eval_samples
    epoch_aspect_loss = as_loss/num_eval_samples
    epoch_sentiment_loss = se_loss/num_eval_samples
    valid_loss_set.append([epoch_eval_loss,epoch_aspect_loss, epoch_sentiment_loss])
    print("Validation loss: total - {}, classifier - {}, sentiment - {}".format(epoch_eval_loss, epoch_aspect_loss, epoch_sentiment_loss))

    if lowest_eval_loss == None:
      lowest_eval_loss = epoch_eval_loss
      # save model
      save_model(model, model_save_path, actual_epoch,\
                 lowest_eval_loss, train_loss_set, valid_loss_set, optimizer)
    else:
      if epoch_eval_loss < lowest_eval_loss:
        lowest_eval_loss = epoch_eval_loss
        # save model
        save_model(model, model_save_path, actual_epoch,\
                   lowest_eval_loss, train_loss_set, valid_loss_set, optimizer)
    print("\n")

  return model, train_loss_set, valid_loss_set

def hook_fn(grad):
    print(grad)  # print grad value

if __name__ == "__main__":
  with open('config/absa_model.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)
    cfg['device']= torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print(f"### Loading config {cfg}")

  tokenizer = AutoTokenizer.from_pretrained(cfg['pretrained'])
  train = load_dataset_by_filepath(cfg, cfg['train_file'], tokenizer= tokenizer)
  val = load_dataset_by_filepath(cfg, cfg['val_file'], tokenizer= tokenizer)

  train_dataloader = create_dataloader(cfg, train)
  val_dataloader = create_dataloader(cfg, val)

  model= AsMil(cfg).to(cfg['device'])

  optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

  try:
    model, start_epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist, optimizer= load_model(model, cfg['saved_model'], optimizer)
  except:
    start_epochs = 0
    lowest_eval_loss = None
    train_loss_hist = []
    valid_loss_hist = []
    
  if cfg['freeze_embedder']:
    print('Freeze embedder layer (set requires_grad=False)!!\n')
    model.embedder.freeze_PhoBert_encoder()

  if cfg['acd_only'] & cfg['acsc_only']:
    print('Warning!! No layer requires grad!!\n')
    sys.exit()
  if cfg['acd_warmup']:
    optimizer_warmup = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    model.set_grad_for_acsc_parameter(requires_grad= False)
    model, train_loss_set, valid_loss_set = fit(model=model,\
                                                num_epochs=cfg['acd_warmup'],\
                                                optimizer=optimizer_warmup,\
                                                train_dataloader=train_dataloader,\
                                                valid_dataloader=val_dataloader,\
                                                model_save_path=cfg['model_save_path'],\
                                                train_loss_set= train_loss_hist, valid_loss_set= valid_loss_hist,\
                                                lowest_eval_loss= lowest_eval_loss,\
                                                start_epoch= 0,\
                                                device=cfg['device'])
    #load best model warmup phase
    model, _, _, train_loss_hist, valid_loss_hist= load_model(model, cfg['saved_model'])
    model.set_grad_for_acsc_parameter(requires_grad= True)
    cfg['acd_warmup']= False
  lowest_eval_loss= None
  if cfg['acd_only']:
    print('Freeze sentiment layers (set requires_grad=False)!!\n')
    model.set_grad_for_acd_parameter(requires_grad= True)
    model.set_grad_for_acsc_parameter(requires_grad= False)
  if cfg['acsc_only']:
    print('Freeze categorical_detection layers (set requires_grad=False)!!\n')
    model.set_grad_for_acsc_parameter(requires_grad= True)
    model.set_grad_for_acd_parameter(requires_grad= False)

  model, train_loss_set, valid_loss_set = fit(model=model,\
                                              num_epochs=cfg['num_epochs'],\
                                              optimizer=optimizer,\
                                              train_dataloader=train_dataloader,\
                                              valid_dataloader=val_dataloader,\
                                              model_save_path=cfg['model_save_path'],\
                                              train_loss_set= train_loss_hist, valid_loss_set= valid_loss_hist,\
                                              lowest_eval_loss= lowest_eval_loss,\
                                              start_epoch= start_epochs,\
                                              device=cfg['device'])

