import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ManyToOneRNN(torch.nn.Module):
    def __init__(self, num_players=22, num_teams=2, num_event_types=10,
                 embed_dim=64, hidden_dim=128, num_layers=1, num_classes=3):
        super(ManyToOneRNN, self).__init__()
        
        # Embeddings for categorical inputs
        self.player_embed = nn.Embedding(num_players, embed_dim)
        self.team_embed   = nn.Embedding(num_teams, embed_dim // 2)
        self.event_embed  = nn.Embedding(num_event_types, embed_dim // 2)
        
        # Vanilla RNN
        input_dim = embed_dim*2 + embed_dim//2 + embed_dim//2
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Classifier head
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
    
    def forward(self, player_from, player_to, team, event_type):
        # Embed categorical features
        from_emb = self.player_embed(player_from)   # [B, L, D]
        to_emb   = self.player_embed(player_to)
        team_emb = self.team_embed(team)
        event_emb= self.event_embed(event_type)
        
        # Concatenate embeddings
        x = torch.cat([from_emb, to_emb, team_emb, event_emb], dim=-1)  # [B, L, input_dim]
        
        # RNN forward
        _, h = self.rnn(x)  # h: [num_layers, B, hidden_dim]
        h = h[-1]           # Take last layer hidden state
        
        # Classifier
        z = F.relu(self.fc1(h))
        out = self.fc2(z)   # [B, num_classes]
        return F.log_softmax(out, dim=-1)

model = ManyToOneRNN()
print(model)