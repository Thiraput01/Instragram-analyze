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
        
        # Combine caption and hashtags
        combined_text = caption + " " + hashtags
        
        # Tokenize and get embeddings
        inputs = self.tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True)
        embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze()
        
        # Get the label (impression count)
        label = torch.tensor(self.dataframe.iloc[idx]['Impressions'], dtype=torch.float)
        
        return embedding, label