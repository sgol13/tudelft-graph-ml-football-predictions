import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear, Softmax, ELU, RNN, CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm

from dataloader_paired import CumulativeSoccerDataset, SequentialSoccerDataset

class Classifier(torch.nn.Module):
    def __init__(self, input_size = 64, hidden_size = 16, num_classes = 3):
        super().__init__()
        self.lin1 = Linear(input_size, hidden_size)
        self.lin2 = Linear(hidden_size, num_classes)
        self.softmax = Softmax(dim = 1)
        
    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.softmax(x)
        
        return x
    
class GAT(torch.nn.Module):
    def __init__(self, input_size = 7, N1 = 128, N2 = 128, N3 = 64, N4 = 64, L = 16):
        super().__init__()
        self.conv1 = GATConv(input_size, N1)
        self.conv2 = GATConv(N1, N2)
        self.conv3 = GATConv(N2, N3)
        
        self.lin = Linear(N3+L, N4)
        self.elu = ELU()
        
    def forward(self, x1, x2, edge_index1, edge_index2, batch, half_y, x_norm2_1, x_norm2_2, edge_col1 = None, edge_col2 = None):
        # 1. Obtain node embeddings 
        x1 = self.elu(self.conv1(x1, edge_index1, edge_col1))
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x1 = self.elu(self.conv2(x1, edge_index1, edge_col1))
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x1 = self.elu(self.conv3(x1, edge_index1, edge_col1))
        x1 = F.dropout(x1, p = 0.5, training = self.training)

        x2 = self.elu(self.conv1(x2, edge_index2, edge_col2))
        x2 = F.dropout(x2, p=0.5, training=self.training)

        x2 = self.elu(self.conv2(x2, edge_index2, edge_col2))
        x2 = F.dropout(x2, p=0.5, training=self.training)

        x2 = self.elu(self.conv3(x2, edge_index2, edge_col2))
        x2 = F.dropout(x2, p = 0.5, training = self.training)

        # 2. Readout layer
        x1 = global_mean_pool(x1, batch) # This can be changed (Experiment?)
        x2 = global_mean_pool(x2, batch) # This can be changed (Experiment?)

        if x_norm2_1 is not None:
            x1 = torch.cat((x1, x_norm2_1), dim = 1) # I assume x_norm2 are the global match features (?)
        if x_norm2_2 is not None:
            x2 = torch.cat((x2, x_norm2_2), dim = 1)

        x1 = self.lin(x1) # These are optional
        x2 = self.lin(x2) # according to the paper (different methods)
        
        return x1, x2
    
class SpatialModel(torch.nn.Module):
    def __init__(self, input_size = 7, N1 = 128, N2 = 128, N3 = 64, N4 = 64, N5 = 16, L = 16, num_classes = 3):
        super().__init__()
        self.gat = GAT(input_size, N1, N2, N3, N4, L)
        self.classifier = Classifier(2*N4, N5, num_classes)
        
    def forward(self, x1, x2, edge_index1, edge_index2, batch, half_y, x_norm2_1, x_norm2_2, edge_col1 = None, edge_col2 = None):
        x1, x2 = self.gat(x1, x2, edge_index1, edge_index2, batch, half_y, x_norm2_1, x_norm2_2, edge_col1, edge_col2)
        x = torch.cat((x1, x2), dim = 1) # x has both graph embeddings
        # For the Disjoint Model, we want this x as output of each GAT
        
        x = self.classifier(x)
        return x
    
# 1 GAT Shared across all windows or 1 GAT per window?
# Concatenate x1 and x2 and feed that to an RNN, or have 2 RNNs (one for x1 and one for x2) and then concatenate their outputs?
class DisjointModel(torch.nn.Module):
    def __init__(self, num_windows=5, hidden_dim = 32, input_size = 7, N1 = 128, N2 = 128, N3 = 64, N4 = 64, N5 = 16, L = 16, num_classes = 3):
        super().__init__()
        self.num_windows = num_windows
        self.N4 = N4
        
        self.gat = GAT(input_size, N1, N2, N3, N4, L)
        self.rnn = RNN(2*N4, hidden_dim, batch_first=True) # Can change num_layers, dropout, nonlinearity
        self.classifier = Classifier(hidden_dim, N5, num_classes)
    
    # All inputs are expected in shape (num_windows, L), with L being the original length of each parameter
    def forward(self, x1, x2, edge_index1, edge_index2, batch, half_y, x_norm2_1, x_norm2_2, edge_col1 = None, edge_col2 = None):
        assert x1.shape[0] == self.num_windows
        assert x2.shape[0] == self.num_windows
        
        outputs = np.zeros(shape=(1, self.num_windows, self.N4))
        for i in range(self.num_windows):
            x1, x2 = self.gat(x1[i,:], x2[i,:], edge_index1[i, :], edge_index2[i,:], batch[i,:], half_y[i,:], x_norm2_1[i,:], x_norm2_2[i,:], edge_col1[i,:], edge_col2[i,:])
            outputs[0, i, :] = torch.cat((x1,x2), dim=1)
            
        rnn_outputs, x = self.rnn(outputs)
        
        x = self.classifier(x)
        
        return x

dataset = CumulativeSoccerDataset(root="data", starting_year=2015, ending_year=2024, time_interval=30)
model = SpatialModel(input_size=1, L=0)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

model.train()
num_epochs = 100

for epoch in tqdm(range(num_epochs)):
    epoch_loss = 0.0
    for i in tqdm(range(200)):
        data = dataset[i]
        if data.home_x.size(0) == data.away_x.size(0):
            batch = torch.zeros(data.home_x.size(0), dtype=torch.long, device=data.home_x.device)
            optimizer.zero_grad()
            x = model(
                data.home_x, data.away_x,
                data.home_edge_index, data.away_edge_index,
                batch, None, None, None
            )

            loss = criterion(x.squeeze(), data.final_result)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(dataset):.4f}")



"""
model = DisjointModel(num_windows=3, input_size=1, L=0)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

model.train()
num_epochs = 100
losses = []

for epoch in tqdm(range(num_epochs)):
    epoch_loss = 0.0
    for i in tqdm(range(200)):
        count = 0
        j = i
        data = []
        while count < 3 and j < 1000:
            if dataset[j].home_x.size(0) == 11 and dataset[j].away_x.size(0) == 11:
                count += 1
                data.append(dataset[j])
            j += 1
        
        # batch = all nodes belong to a single graph
        batch = torch.zeros(data.home_x.size(0), dtype=torch.long, device=data.home_x.device)

        # batch_home = torch.zeros(data.home_x.size(0), dtype=torch.long, device=data.home_x.device)
        # batch_away = torch.zeros(data.away_x.size(0), dtype=torch.long, device=data.away_x.device)

        optimizer.zero_grad()
        x = model(
            data.home_x, data.away_x,
            data.home_edge_index, data.away_edge_index,
            batch, None, None, None
        )

        loss = criterion(x.squeeze(), data.final_result)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    losses.append(epoch_loss / len(dataset))
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(dataset):.4f}")

    # TO DO: Update implementation of disjoint model to use "batch" rather than tensors of size "num_windows" for the different time windows.
"""