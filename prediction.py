#!/usr/bin/env python
# coding: utf-8

# In[20]:



import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.linalg import hadamard
import flair
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings
import re

# In[3]:


class LSTMTagger(nn.Module):

    def __init__(self, hidden_dim, tagset_size):
        ''' Initialize the layers of this model.'''
        super(LSTMTagger, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        #LAYER1 : LSTM
        # the LSTM takes embedded word vectors (of a specified size) as inputs and outputs hidden states of size hidden_dim
        self.lstm = nn.GRU(2048, hidden_dim, bidirectional=True)

        #LAYER2 : DENSE
        # the linear layer that maps the hidden state output dimension to the number of tags we want as output, tagset_size (in this case this is 3 tags)
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
        

    
    def forward(self, sentence):
               
        lstm_out, self.hidden = self.lstm(sentence.view(len(sentence), 1, -1))
        #print(lstm_out.shape)
        
        # get the scores for the most likely tag for a word
        tag_scores = self.hidden2tag(lstm_out.view(len(sentence), -1))
     
        return tag_scores


# In[5]:
class EntityExtraction():
    def __init__(self, gpu = False):
        
        if gpu:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        self.embedding = flair.embeddings.TransformerWordEmbeddings('roberta-large')
            
        #from trained model
        self.hidden_dim = 1024
        self.tag_size = 3
        
        # load model
        self.model = LSTMTagger(self.hidden_dim, self.tag_size)
        self.model.load_state_dict(torch.load("model_st.pth"))
        self.model.to(self.device)
        self.model.eval()
        

    def clean_sen(self, sen):
        sen = re.sub('[^a-zA-Z0-9 \n\.]', ' ', sen)
        sen = sen.replace('.', ' ')
        sen = sen.lower()
        sen = sen.split()
        return sen


    def prepare_sequence(self, seq):
        seq = self.clean_sen(seq)
        sen = Sentence(seq)
        self.embedding.embed(sen)
        word_embeds = torch.zeros(len(seq), 1024).to(self.device)
        for j in range(len(sen)):
            seq = sen[j].embedding
            word_embeds[j,:] = seq     

        wh = torch.Tensor(hadamard(1024)).to(self.device)
        ht_wh= torch.matmul(word_embeds, wh)
        final_ht_wh = torch.cat((word_embeds , ht_wh), 1)

        return final_ht_wh
    
    def get_embeddings(self,sen):
        emb_test = self.prepare_sequence(sen)
        return emb_test


    def get_scores(self,sen):
        emb_test = self.get_embeddings(sen)
        tag_scores = self.model(emb_test)
        _, predicted_tags = torch.max(tag_scores, 1)
        return predicted_tags
    
    def predict_tags(self,sen):   
        predicted_tags = self.get_scores(sen)
        output_tags = []
        for i in predicted_tags:
            if i==0:
                output_tags.append('Other')
            elif i==1:
                output_tags.append('Cause')
            else:
                output_tags.append('Effect')
        return output_tags







