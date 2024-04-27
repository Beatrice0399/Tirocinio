#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:03:10 2024

@author: beatriceippoliti
"""

import pandas as pd
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Carica i dati dalla riga idx del dataframe
        sample = self.data.iloc[idx]
        
        # Esempio: se il CSV ha colonne 'feature1', 'feature2', 'label'
        features = torch.tensor(sample[['feature1', 'feature2']].values, dtype=torch.float32)
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        # Applica la trasformazione, se presente
        if self.transform:
            features = self.transform(features)
        
        return features, label
