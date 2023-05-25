import numpy as np
import re
import pandas as pd
import torch
import underthesea
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
import yaml
import os


def label_encoder(label, aspects):
    y = [np.nan] * len(aspects)
    ap_stm = re.findall('{(.+?), ([a-z]+)}', label)

    for aspect, sentiment in ap_stm:
        idx = aspects.index(aspect)
        y[idx] = sentiment

    return y

def label_encoder_df(df):
  return df.replace({'negative': 1, 
            'neutral': 2, 
            'positive': 3})

def onehot_enconder(labels, num_classes):
  # Khởi tạo mảng zero-filled với kích thước (length, num_classes)
  onehot = np.zeros((len(labels), num_classes))
  
  for i, value in enumerate(labels):
      # Kiểm tra nếu giá trị là NaN
      if np.isnan(value):
          onehot[i] = np.nan
      else:
          # Chuyển giá trị thành số nguyên
          value_int = int(value)-1
          
          # Thiết lập vị trí tương ứng thành 1
          onehot[i, value_int] = 1.
  
  return onehot

def txt2df(filepath, aspect):
    with open(filepath, 'r', encoding='utf-8-sig') as txt:
        data = txt.read().split('\n')

    df = pd.DataFrame()
    df['review'] = [review for review in data[1::4]]
    df[aspect] = [label_encoder(label, aspect) for label in data[2::4]]

    return df

def tokenize_inputs(text_list, tokenizer, num_embeddings=512):
    # Tokenize the text, then truncate sequence to the desired length minus 2 for the 2 special characters
    tokenized_texts = list(map(lambda t: tokenizer.tokenize(t)[:num_embeddings-2], text_list))
    # Convert tokenized text into numeric ids for the appropriate LM
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # Append special tokens [CLS] and [SEP] to the end of each sentence
    input_ids = [tokenizer.build_inputs_with_special_tokens(x) for x in input_ids]
    # Pad sequences
    input_ids = pad_sequence([torch.tensor(x) for x in input_ids], batch_first=True, padding_value=0)
    return input_ids

def create_attn_masks(input_ids):
    # Create attention masks
    attention_masks = []
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks
