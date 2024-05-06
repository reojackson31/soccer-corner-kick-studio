import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Data, DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.model_selection import StratifiedShuffleSplit

import os
import warnings
warnings.filterwarnings("ignore")

def create_graph_data(row):
    players = row['freeze_frame']
    attack_rating = row['attack']
    defense_rating = row['defence']

    features = [
        [player['location'][0], player['location'][1],
         attack_rating if player['teammate'] else defense_rating]
        for player in players
    ]
    features_tensor = torch.tensor(features, dtype=torch.float)
    
    edges = []
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            if players[i]['teammate'] == players[j]['teammate']:
                edges.append([i, j])
                edges.append([j, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    target = torch.tensor([int(row['pass_shot_assist'])], dtype=torch.long)
    
    graph_data = Data(x=features_tensor, edge_index=edge_index, y=target)
    return graph_data


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
    
def get_prediction(attack_locations, defense_locations, attack_rating, defense_rating):
    graph_input = pd.DataFrame({
        'freeze_frame': [
            {'location': loc, 'teammate': True}
            for loc in attack_locations
        ] + [
            {'location': loc, 'teammate': False}
            for loc in defense_locations
        ],
        'attack': 0,
        'defence': 0,
        'pass_shot_assist': 0
    })
    graph_input['attack'] = attack_rating
    graph_input['defence'] = defense_rating
    data = create_graph_data(graph_input.iloc[0])
    model = GCN(hidden_channels=64, num_node_features=3, num_classes=2)
    model.load_state_dict(torch.load('gnn_model.pth'))
    model.eval()
    with torch.no_grad():
        out = model(data.x.unsqueeze(0), data.edge_index, torch.zeros(data.x.size(0), dtype=torch.long))
        prob = F.softmax(out.squeeze(), dim=0)
        return prob[1].item()
    