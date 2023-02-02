import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from torch.utils.data import DataLoader, Dataset

#%% DataSet
class Lipophilicity(Dataset):
    def __init__(self,path):

        self.df = pd.read_csv(path)

        self.input_vectors= self.df[self.df.columns[0:-1]].values
        
        self.targets = self.df[self.df.columns[-1]].values

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index): 
        input_vector = self.input_vectors[index]
        target = self.targets[index]
        
        return torch.tensor(input_vector, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

data_set = Lipophilicity('lipo_fp_processed.csv') 

#%% DataLoader


print('hola mundo')
