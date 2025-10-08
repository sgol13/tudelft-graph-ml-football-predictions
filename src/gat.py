import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear, Softmax, ELU

class GAT(torch.nn.Module):
    def __init__(self, input_size = 7, N1 = 128, N2 = 128, N3 = 64, N4 = 64, N5 = 16, L = 16, num_classes = 3):
        super(GAT,self).__init__()
        self.conv1 = GATConv(input_size, N1)
        self.conv2 = GATConv(N1, N2)
        self.conv3 = GATConv(N2, N3)
        
        self.lin1 = Linear(N3+L, N4)
        self.lin2 = Linear(N4*2, N5)
        self.lin3 = Linear(N5, num_classes)
        
        self.elu = ELU()
        self.softmax = Softmax(dim = 1)
        
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

        x1 = torch.cat((x1, x_norm2_1), dim = 1) # I assume x_norm2 are the global match features (?)
        x2 = torch.cat((x2, x_norm2_2), dim = 1)

        x1 = self.lin1(x1) # These are optional
        x2 = self.lin1(x2) # according to the paper (different methods)

        x = torch.cat((x1, x2), dim = 1) # x has both graph embeddings

        # 3. Apply a final classifier
        x = self.lin2(x)
        x = self.lin3(x)
        x = self.softmax(x)
    
        return x
    
model = GAT()
print(model)