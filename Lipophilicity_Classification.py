#%% Libraries
import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryConfusionMatrix

#%% Device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("\n ///////// Running on the GPU /////////")
else:
    device = torch.device("cpu")
    print("\n //////// Running on the cpu /////////")



#%% DataSet
class Lipophilicity(Dataset):
    def __init__(self,path):

        self.df = pd.read_csv(path)
        self.input_vectors= self.df[self.df.columns[0:-1]].values
        self.targets = self.df[self.df.columns[-1]].values
        self.mean_lipo = self.df['lipo'].mean()

        self.classify = []
        for i in self.targets:
            if i <= self.mean_lipo:
                self.classify.append([1, 0])
            else:
                self.classify.append([0, 1])      

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index): 
        input_vector = self.input_vectors[index]
        target = self.classify[index]

        return   torch.tensor(input_vector, dtype=torch.float32, device=device), torch.tensor(target, dtype=torch.float32, device=device)



#%% Model  
class Linear_Net(nn.Module): 
    

    def __init__(self,input_size, hidden_size1, output_size): 
        super(Linear_Net, self).__init__() 

        self.fl1 = nn.Linear(input_size, hidden_size1)
        self.fl2 = nn.Linear(hidden_size1, output_size) 
        
    def forward(self, x):
        out = self.fl1(x)
        out = F.relu(out) 
        out = self.fl2(out)
        
        return out 

#%% Train routine

def train(model, device, train_dataloader, optim): 
    model.train()

    loss_func = nn.CrossEntropyLoss()
    loss_collect = 0

    
    for b_i, (input_vectors, target) in enumerate(train_dataloader):

        input_vectors, target = input_vectors.to(device), target.to(device)

        optim.zero_grad() 
        pred_prob = model(input_vectors)

        loss = loss_func(pred_prob, target.view(-1,2))
        loss.backward() 

        optim.step() 
        loss_collect += loss.item() 
    

        if b_i % 10 == 0:
            print('epoch: {} [{}/{} ({:.0f}%)]\t  training loss: {:.1f}'.format(
                epoch, b_i * len(input_vectors), len(train_dataloader.dataset),
                100 * b_i * len(input_vectors) / len(train_dataloader.dataset),
                loss.item()))
    loss_collect /= len(train_dataloader.dataset)          

    return loss_collect


#%% Validation Routine

def validate(model, device, val_dataloader, epoch): 

    model.eval()
    loss_collect = 0 
    loss_func = nn.CrossEntropyLoss()
    
    with torch.no_grad(): 
        for input_vectors, target in val_dataloader:

            input_vectors, target = input_vectors.to(device), target.to(device) 
            pred_prob = model(input_vectors)
            loss_collect += loss_func(pred_prob, target).item()
  
    loss_collect /= len(val_dataloader.dataset)
    
    print('\nTest dataset: Overall Loss: {:.1f}, ({:.2f}%)\n'.format(
        len(val_dataloader.dataset)*loss_collect,loss_collect))

    return loss_collect


#%% Prediction 
def predict(model, device, dataloader):

    model.eval()

    input_vectors_all = []
    targets_all = []
    pred_prob_all = []

    # Remove gradients:
    with torch.no_grad():

        # Looping over the dataloader allows us to pull out input/output data:
        for input_vector, target in dataloader:

            # Assign X and y to the appropriate device:
            input_vector, target = input_vector.to(device), target.to(device)

            # Make a prediction:
            pred_prob = model(input_vector)

            input_vectors_all.append(input_vector)
            targets_all.append(target)
            pred_prob_all.append(pred_prob)
    return (
        torch.concat(input_vectors_all), 
        torch.concat(targets_all), 
        torch.concat(pred_prob_all).view(-1))



#%% Data Loader
data_set = Lipophilicity('lipo_processed.csv') 
classify_split = data_set.mean_lipo
k = data_set.classify

bat_size = 60

# important to use split for test data and validation data
size_train = int(len(data_set) * 0.6) 
size_val = len(data_set) - size_train

train_set, val_set = torch.utils.data.random_split(data_set, [size_train, size_val], generator=torch.Generator().manual_seed(0))

train_loader = DataLoader(dataset = train_set, batch_size=bat_size, shuffle=True)
val_loader = DataLoader(dataset = val_set, batch_size=bat_size, shuffle=True)


#%% Run training loop

learning_rate = 0.0001
torch.manual_seed(0)


model = Linear_Net(input_size=1024, hidden_size1=10, output_size=2).to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate) 

losses_train = []
losses_val = []

for epoch in range(1, 100): 
    train_loss = train(model, device, train_loader, optimizer)
    losses_train.append(train_loss)

    val_loss = validate(model, device, val_loader, epoch)
    losses_val.append(val_loss)

plt.plot(losses_train, label ='Train losses')
plt.plot(losses_val, label ='Validation losses')

plt.legend()
plt.xlabel('Time')
plt.ylabel('Losses')

plt.show()


#%% Model Statistics
input_all, target_all, pred_prob_all = predict(model, device, val_loader)

target_all = target_all.cpu()
pred_prob_all = pred_prob_all.cpu() 




print("//////////////// hola mundo ////////////////")

