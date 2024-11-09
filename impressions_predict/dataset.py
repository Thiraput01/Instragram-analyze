from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torchvision
import sys
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5")

class InstagramDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        
        # Get caption and hashtags
        caption = self.dataframe.iloc[idx]['Caption']
        hashtags = self.dataframe.iloc[idx]['Hashtags keywords']
        
        # tokenize the caption and hashtags
        caption_tokens = self.tokenizer(caption, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        hashtags_tokens = self.tokenizer(hashtags, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        
        # embed the caption and hashtags
        with torch.no_grad():
            caption_embedding = model(**caption_tokens).last_hidden_state.mean(dim=1)
            hashtags_embedding = model(**hashtags_tokens).last_hidden_state.mean(dim=1)
            
        # concatenate the embeddings
        embedding = torch.cat((caption_embedding, hashtags_embedding), dim=1).squeeze(0)
        
        label = torch.tensor(self.dataframe.iloc[idx]['Impressions'], dtype=torch.float32)
        
        return embedding, label # embedding dim = 768*2 = 1536