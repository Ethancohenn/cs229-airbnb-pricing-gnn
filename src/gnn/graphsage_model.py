# Import libraries required
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Build the GraphSAGE model
class GraphSAGE_Model(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
        """
        GraphSAGE model for regression.
        Args:
            in_channels: number of node features
            hidden_channels: hidden layer size
            out_channels: output dimension (1 for price prediction)
        """
        super(GraphSAGE_Model, self).__init__()

        # 1st GraphSAGE layer
        self.conv1 = SAGEConv(in_channels, hidden_channels)

        # 2nd GraphSAGE layer
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        # Final MLP to predict price
        self.fc = nn.Linear(hidden_channels, out_channels)

        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass.
        x: node features [num_nodes, in_channels]
        edge_index: graph edges [2, num_edges]
        """

        # ---- GraphSAGE layer 1 ----
        x = self.conv1(x, edge_index)  
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ---- GraphSAGE layer 2 ----
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ---- Final regression layer ----
        out = self.fc(x)  # [num_nodes, 1]

        return out.squeeze()  # flatten to [num_nodes]

# From your friend:
# x           -> node feature tensor
# edge_index  -> graph edges
# y           -> price labels
# train_mask, val_mask, test_mask -> boolean masks

data = Data(
    x=x,
    edge_index=edge_index,
    y=y,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask
)

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)  # predictions
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def evaluate(model, data, mask, criterion):
    model.eval()
    out = model(data.x, data.edge_index)
    loss = criterion(out[mask], data.y[mask])
    mae = torch.mean(torch.abs(out[mask] - data.y[mask]))
    return loss.item(), mae.item()

# Hyperparameters
in_channels  = data.x.size(1)
hidden_channels = 64
out_channels = 1

model = GraphSAGE_Model(in_channels, hidden_channels, out_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

EPOCHS = 200

for epoch in range(1, EPOCHS + 1):
    train_loss = train(model, data, optimizer, criterion)
    val_loss, val_mae = evaluate(model, data, data.val_mask, criterion)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")
        
test_loss, test_mae = evaluate(model, data, data.test_mask, criterion)
print("Test MSE:", test_loss)
print("Test MAE:", test_mae)
