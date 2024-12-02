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
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        
        # Retrieve the caption, hashtags, and label for the given index
        caption = self.dataframe.iloc[idx]['Caption']
        hashtags = self.dataframe.iloc[idx]['Hashtags keywords']
        label = torch.tensor(self.dataframe.iloc[idx]['Impressions'], dtype=torch.float32)

        # Tokenize and return the raw tokenized inputs
        caption_tokens = self.tokenizer(caption, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        hashtags_tokens = self.tokenizer(hashtags, return_tensors='pt', padding='max_length', truncation=True, max_length=128)

        # Embed the caption and hashtags using the model
        with torch.no_grad():
            caption_embeddings = model(**caption_tokens).last_hidden_state.mean(dim=1)
            hashtags_embeddings = model(**hashtags_tokens).last_hidden_state.mean(dim=1)
            
        # Concatenate the embeddings
        embeddings = torch.cat([caption_embeddings, hashtags_embeddings], dim=1)
        
        return embeddings, label