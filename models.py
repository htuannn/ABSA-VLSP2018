import torch
import torch.nn as nn
import numpy as np
import yaml
import os
from transformers import AutoModel

class PhoBertFeatureExtraction(torch.nn.Module):
  def __init__(self, cfg) -> None:
    super(PhoBertFeatureExtraction, self).__init__()
    self.cfg= cfg
    self.phobert = AutoModel.from_pretrained(self.cfg['pretrained'])
    
  def forward(self, input_ids, token_type_ids=None,\
              attention_mask=None, labels= None, avg_pooling=False) -> torch.Tensor:
    # last hidden layer
    last_hidden_state = self.phobert(input_ids=input_ids,\
                                   attention_mask=attention_mask,\
                                   token_type_ids=token_type_ids)
    if avg_pooling: return self.pool_hidden_state(last_hidden_state)
    return last_hidden_state[0]
    
  def pool_hidden_state(self, last_hidden_state):
    """
    Pool the output vectors into a single mean vector 
    """
    last_hidden_state = last_hidden_state[0]
    mean_last_hidden_state = torch.mean(last_hidden_state, 1)
    return mean_last_hidden_state

  def freeze_PhoBert_decoder(self):
    """
    Freeze PhoBert weight parameters. They will not be updated during training.
    """
    for param in self.phobert.parameters():
      param.requires_grad = False
    
  def unfreeze_PhoBert_decoder(self):
    """
    Unfreeze PhoBert weight parameters. They will be updated during training.
    """
    for param in self.phobert.parameters():
      param.requires_grad = True
  def get_output_dim(self):
    output_dim= self.state_dict()['phobert.embeddings.word_embeddings.weight'].shape[1]
    return output_dim
    
class AsMil(torch.nn.Module):
  def __init__(self, cfg) -> None:
    super(AsMil, self).__init__()
    self.cfg= cfg
    self.device= self.cfg['device']
    self.embedder= PhoBertFeatureExtraction(self.cfg).to(self.device) 
    self.aspect = self.cfg['aspect']
    self.polarites = self.cfg['labels']
    self.aspect_num= len(self.aspect)
    self.polarity_num= len(self.polarites)
    self.word_embedding_dim=self.embedder.get_output_dim()
    self.aspect_loss= torch.nn.BCEWithLogitsLoss()
    self.sentiment_loss= torch.nn.CrossEntropyLoss(reduce=False)
    self.log_vars = nn.Parameter(torch.zeros((self.aspect_num*2)))

    lstm_input_size= self.embedder.get_output_dim()

    self.embedding_layer_fc = torch.nn.Linear(self.word_embedding_dim, self.word_embedding_dim, bias=True)
    self.lstm = torch.nn.LSTM(lstm_input_size, int(self.word_embedding_dim / 2), batch_first=True,
                          bidirectional=True, num_layers=self.cfg['num_layers_lstm'], dropout=0.5)
    torch.nn.init.xavier_normal_(self.embedding_layer_fc.weight)

    self.embedding_layer_aspect_attentions = nn.ModuleList()
    for AttentionInHtt_idx in range(self.aspect_num):
       self.embedding_layer_aspect_attentions.append(AttentionInHtt(self.word_embedding_dim,
                                                              self.word_embedding_dim))

    self.category_fcs = nn.ModuleList()
    for layer_idx in range(self.aspect_num):
      self.category_fcs.append(torch.nn.Linear(self.word_embedding_dim, 1))

    self.sentiment_fcs = nn.ModuleList()
    for layer_idx in range(self.aspect_num):
      self.sentiment_fcs.append(torch.nn.Sequential(torch.nn.Linear(self.word_embedding_dim, self.word_embedding_dim),\
                                                    torch.nn.GELU(),\
                                                    torch.nn.Linear(self.word_embedding_dim, self.polarity_num)))
    
  def forward(self, input_ids, token_type_ids=None,\
              attention_mask=None, labels=None) -> dict:
    # word_embedding
    if labels is None:
      self.embedder.eval()
    else: 
      self.embedder.train()

    embedding_feture = self.embedder(input_ids=input_ids,\
                                    attention_mask=attention_mask,\
                                    labels= labels)
    word_embeddings_fc= self.embedding_layer_fc(embedding_feture)
    if self.cfg['use_lstm']:
      lstm_result, _ = self.lstm(embedding_feture)
    embedding_layer_category_outputs = []
    embedding_layer_category_alphas = []
    embedding_layer_sentiment_outputs = []
    embedding_layer_sentiment_alphas = []
    for i in range(self.aspect_num):
        embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
        alpha = embedding_layer_aspect_attention(word_embeddings_fc, attention_mask)
        embedding_layer_category_alphas.append(alpha)
        
        if self.cfg['use_lstm']:
          lstm_embeddings_attention= element_wise_mul(lstm_result, alpha, return_not_sum_result=False)
          word_embeddings_attention= element_wise_mul(word_embeddings_fc, alpha, return_not_sum_result=False)
          
          #sentiment analysis
          #sentiment_alpha= embedding_layer_category_alphas[i]
          #sentiment_alpha = sentiment_alpha.unsqueeze(1)
          sentiment = self.sentiment_fcs[i](lstm_embeddings_attention)
        else:  
          word_embeddings_attention = element_wise_mul(word_embeddings_fc, alpha, return_not_sum_result=False)

          #sentiment_alpha= embedding_layer_category_alphas[i]
          #sentiment_alpha = sentiment_alpha.unsqueeze(1)

          sentiment = self.sentiment_fcs[i](word_embeddings_attention)
        embedding_layer_category_outputs.append(word_embeddings_attention)

        #entiment_output = torch.matmul(sentiment_alpha, sentiment).squeeze(1) 
        embedding_layer_sentiment_outputs.append(sentiment)

    final_category_outputs = []
    final_lstm_category_outputs = []
    final_sentiment_outputs = []    

    for i in range(self.aspect_num):
      category_output = embedding_layer_category_outputs[i]
      final_category_output=  self.category_fcs[i](category_output)
      final_category_outputs.append(final_category_output)

      sentiment_output= embedding_layer_sentiment_outputs[i]
      final_sentiment_outputs.append(sentiment_output)
      
    pred_categorys= np.array([torch.sigmoid(e).detach().cpu().numpy() for e in final_category_outputs]).transpose(1,0,2)
    pred_sentiments= np.array([torch.nn.functional.softmax(e, dim=-1).detach().cpu().numpy() for e in final_sentiment_outputs]).transpose(1,0,2)

    output={}
    if labels is None:
      predict=[]
      for (pred_category, pred_sentiment)  in zip(pred_categorys,pred_sentiments):
        pred={}
        for i in range(self.aspect_num):
          if pred_category[i] >= 0.5:
            pred[self.aspect[i]] = self.polarites[pred_sentiment[i].argmax()]
          else:
            pred[self.aspect[i]] = np.nan
        predict.append(pred)
      output['predict']= predict
    else:
      aspect_labels = (~torch.isnan(labels)).all(axis=-1).clone().detach().type(torch.float).transpose(0,1)
      polarity_labels= labels.clone().detach().transpose(0,1)

      aspect_loss_total = 0
      sent_loss_total = 0
      loss = 0 
      for i in range(self.aspect_num):
        #classfication loss
        aspect_loss_total += self.aspect_loss(final_category_outputs[i].squeeze(dim=-1), aspect_labels[i])
        
        #sentiment loss 
        if not(polarity_labels[i].isnan().all()):
          sent_loss_total += self.sentiment_loss(final_sentiment_outputs[i][aspect_labels[i].bool()],\
                                                 polarity_labels[i][aspect_labels[i].bool()]).mean()

        #sent_loss = torch.tensor([]).to(self.cfg['device'])
        #for ind,b in enumerate(polarity_labels[i]):
        #  if ~b.isnan().all(axis=0):
        #    sent_loss= torch.cat((sent_loss, self.sentiment_loss(final_sentiment_outputs[i][ind].squeeze(dim=-1), b).unsqueeze(0)), dim=0)
        #mean_sent_loss= sent_loss.mean()
        
        #if not(mean_sent_loss.isnan()):
        #  sent_loss_total+=mean_sent_loss


      output['loss']= aspect_loss_total+ sent_loss_total
      output['aspect_loss']= aspect_loss_total
      output['sent_loss']= sent_loss_total
    return output

class AttentionInHtt(torch.nn.Module):
    """
    2016-Hierarchical Attention Networks for Document Classification
    """

    def __init__(self, in_features, out_features, bias=True, softmax=True) -> None:
        super().__init__()
        self.W = torch.nn.Linear(in_features, out_features, bias)
        self.uw = torch.nn.Linear(out_features, 1, bias=False)
        if softmax:
          self.softmax = MaskedSoftmax()
        else:
          self.softmax = None

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        u = self.W(h)
        u = torch.tanh(u)
        similarities = self.uw(u)
        similarities = similarities.squeeze(dim=-1)
        if self.softmax is not None:
            alpha = self.softmax(similarities, mask)
            return alpha
        else:
            return similarities

class MaskedSoftmax(nn.Module):
    def __init__(self) -> None:
        super(MaskedSoftmax, self).__init__()
        self.softmax = nn.Softmax(1)

    def forward(self, x, mask=None) -> torch.Tensor:
        """
        Performs masked softmax, as simply masking post-softmax can be
        inaccurate
        :param x: [batch_size, num_items]
        :param mask: [batch_size, num_items]
        :return:
        """
        if mask is not None:
            mask = mask.float()
        if mask is not None:
            x_masked = x * mask + (1 - 1 / mask)
        else:
            x_masked = x
        x_max = x_masked.max(1)[0]
        x_exp = (x - x_max.unsqueeze(-1)).exp()
        if mask is not None:
            x_exp = x_exp * mask.float()
        return x_exp / x_exp.sum(1).unsqueeze(-1)


def element_wise_mul(input1, input2, return_not_sum_result=False):
    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1)
        feature_2 = feature_2.expand_as(feature_1)
        feature = feature_1 * feature_2
        feature = feature.unsqueeze(0)
        feature_list.append(feature)
    output = torch.cat(feature_list, 0)

    result = torch.sum(output, 1)
    if return_not_sum_result:
        return result, output
    else:
        return result