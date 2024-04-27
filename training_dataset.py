#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:55:38 2024

@author: beatriceippoliti
"""


from torch.utils.data import DataLoader
from torchvision import transforms
from CustomDataset import CustomDataset


# Definizione della trasformazione per i dati (se necessario)
transform = transforms.Compose([
    # Aggiungi eventuali trasformazioni necessarie qui
])

# Caricamento del dataset CSV utilizzando la classe CustomDataset
dataset = CustomDataset(csv_file='/Users/beatriceippoliti/Desktop/Uni/Tirocinio/dataset/datafile.csv', transform=transform)

# Creazione del DataLoader per iterare sui dati durante l'addestramento
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(dataloader.)