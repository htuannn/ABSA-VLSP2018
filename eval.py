import os
import math
import yaml
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import pandas as pd

from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from models import *
from dataset import *
from model_utils import *

label_map= {'not_exist': -1,
            'negative':0,
            'neutral':1,
            'positive':2}

replacements={-1: 'None',
              0: 'negative',
              1: 'neutral',
              2: 'positive'}
target_names = list(map(str, replacements.values()))

def aspect_detection_eval(y_test, y_pred):
  """
  y_test: grouth_true test, DataFrame
  y_pred: grouth_true predict, DataFrame
  """
  categories= y_pred.columns
  y_test= y_test.fillna('not_exist').replace(label_map).values.tolist()
  y_pred= y_pred.fillna('not_exist').replace(label_map).values.tolist()

  aspect_test = []
  aspect_pred = []

  for row_test, row_pred in zip(y_test, y_pred):
      for index, (col_test, col_pred) in enumerate(zip(row_test, row_pred)):
          aspect_test.append(bool(col_test) * categories[index])
          aspect_pred.append(bool(col_pred) * categories[index])

  aspect_report = classification_report(aspect_test, aspect_pred, digits=4, zero_division=1, output_dict=True)
  print("## Aspect Detection Evaluate ##")
  print(classification_report(aspect_test, aspect_pred, digits=4, zero_division=1))
  
def sentiment_classification_eval(y_test, y_pred):
  """
  y_test: grouth_true test, DataFrame
  y_pred: grouth_true predict, DataFrame
  """
  categories= y_pred.columns
  y_test= y_test.fillna('not_exist').replace(label_map).values.tolist()
  y_pred= y_pred.fillna('not_exist').replace(label_map).values.tolist()

  y_test_flat = np.array(y_test).flatten()
  y_pred_flat = np.array(y_pred).flatten()
  target_names = list(map(str, replacements.values()))

  polarity_report = classification_report(y_test_flat, y_pred_flat, digits=4, output_dict=True)
  print("## Sentiment Classification Evaluate ##")
  print(classification_report(y_test_flat, y_pred_flat, target_names=target_names, digits=4))
  
def combination_eval(y_test, y_pred):
  """
  y_test: grouth_true test, DataFrame
  y_pred: grouth_true predict, DataFrame
  """
  categories= y_pred.columns
  y_test= y_test.fillna('not_exist').replace(label_map).values.tolist()
  y_pred= y_pred.fillna('not_exist').replace(label_map).values.tolist()

  aspect_polarity_test = []
  aspect_polarity_pred = []

  for row_test, row_pred in zip(y_test, y_pred):
      for index, (col_test, col_pred) in enumerate(zip(row_test, row_pred)):
          aspect_polarity_test.append(f'{categories[index]},{replacements[col_test]}')
          aspect_polarity_pred.append(f'{categories[index]},{replacements[col_pred]}')

  aspect_polarity_report = classification_report(aspect_polarity_test, aspect_polarity_pred, digits=4, zero_division=1, output_dict=True)
  print("## Combination Evaluate (Aspect Detection + Sentiment Classification) ##")
  print(classification_report(aspect_polarity_test, aspect_polarity_pred, digits=4, zero_division=1))
  
if __name__ == "__main__":
  with open('config/absa_model.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)
    cfg['device']= torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print(f"### Loading config {cfg}")
    
  tokenizer = AutoTokenizer.from_pretrained(cfg['pretrained'])
  
  model= AsMil(cfg).to(cfg['device'])
  model, _ , _ , _ , _ = load_model(model, cfg['saved_model'])
  model.eval()
  
  test = load_dataset_by_filepath(cfg, cfg['test_file'], tokenizer= tokenizer)
  
  X = test["features"].values.tolist()
	masks = test["masks"].values.tolist()
	label_cols =test.columns.values.tolist()[1:-2]
	y = label_encoder_df(test[label_cols].applymap(lambda x: x.lower() if pd.notnull(x) else x)).values.tolist()
  
	X = torch.tensor(X)

	y = torch.tensor(np.array([onehot_enconder(lb, len(cfg['labels'])) for lb in y]))

	masks = torch.tensor(masks, dtype=torch.long)

	test_set = TensorDataset(X, masks, y)
  
  pred= []
  for i in tqdm(range(len(test_set))):
    batch =  tuple(t.to(cfg['device']) for t in test_set[i])
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
      output = model(b_input_ids.unsqueeze(0), attention_mask=b_input_mask.unsqueeze(0), labels=None)
    pred.append(output['predict'][0])
    
y_pred=pd.DataFrame(pred)
y_pred=y_pred.applymap(lambda x: x.lower() if pd.notnull(x) else x)
categories= y_pred.columns

y_test=test.loc[:,categories]
y_test=y_test.applymap(lambda x: x.lower() if pd.notnull(x) else x)

aspect_detection_eval(y_test, y_pred)

sentiment_classification_eval(y_test, y_pred)

combination_eval(y_test, y_pred)
