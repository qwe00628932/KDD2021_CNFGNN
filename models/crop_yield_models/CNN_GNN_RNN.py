import torch
import models
from models import FeatureCNN
import torch.nn.functional as F


class CNN_GNN(torch.nn.Module):
    def __init__(self, device, num_layers=1, hidden_size=40,
               conv_name="GATConv", dropout=0.0, heads=1, aggr='max', **kwargs):
        super(CNN_GNN, self).__init__()
        self.feature_cnn = FeatureCNN()
        conv_class = getattr(models, conv_name)
        self.conv1 = conv_class(80, hidden_size, dropout=dropout, heads=heads, aggr=aggr)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(conv_class(hidden_size, hidden_size, dropout=dropout, aggr=aggr))
        self.device = device

    def forward(self, data):
        x = data.x.to(self.device)
        # avg_y = x[:, :, 0].reshape(-1, 1)
        x = x[:, :, 3:].reshape(-1, 1, 392)
        edge_index = data.edge_index.to(self.device)
        out = self.feature_cnn(x)  # out: 80
        # out = torch.cat([out, avg_y], 1)
        # 5作为图神经网络中的batch_size
        out = out.reshape(-1, 5, 80).transpose(0, 1)
        out = F.relu(self.conv1(out, edge_index))
        for conv in self.convs:
            out = conv(out, edge_index)
        return out # 5, N, hidden_size


class CNN_GNN_RNN(torch.nn.Module):
    def __init__(self, device, num_layers=1, hidden_size=40,
               conv_name="GATConv", dropout=0.0, heads=1, aggr='max', **kwargs):
        super(CNN_GNN_RNN, self).__init__()
        self.cnn_gnn = CNN_GNN(device=device, num_layers=num_layers,
                               hidden_size=hidden_size, conv_name=conv_name,
                               dropout=dropout, heads=heads, aggr=aggr)
        self.gru = torch.nn.GRU(input_size=hidden_size+1, hidden_size=64, num_layers=1)
        self.lin1 = torch.nn.Linear(64, 40)
        self.lin2 = torch.nn.Linear(40, 1)
        self.device = device

    def forward(self, data):
        g_out = self.cnn_gnn(data) # 5,N, hidden_size
        y = data.x[:, :4, 2:3].transpose(0, 1).to(self.device)
        g_out = torch.cat((F.relu(g_out[:4, :, :]), y), 2)
        _, out = self.gru(g_out)  # N, 64
        # todo: 这里可以拼接上当年的环境信息，再进行最终的预测
        out = F.relu(self.lin1(out))
        return self.lin2(out)


class Single_CNN_GNN(torch.nn.Module):
    def __init__(self, device, num_layers=1, hidden_size=40,
                 conv_name="GATConv", dropout=0.0, heads=1, aggr='max', **kwargs):
        super(Single_CNN_GNN, self).__init__()
        self.feature_cnn = FeatureCNN()
        conv_class = getattr(models, conv_name)
        self.conv1 = conv_class(81, hidden_size, dropout=dropout, heads=heads, aggr=aggr)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(conv_class(hidden_size, hidden_size, dropout=dropout, aggr=aggr))
        self.lin = torch.nn.Linear(20, 1)
        self.device = device

    def forward(self, data):
        x = data.x[:, -1, :].to(self.device)
        avg_y = x[:, 0:1].reshape(-1, 1)
        x = x[:, 3:].reshape(-1, 1, 392)
        edge_index = data.edge_index.to(self.device)
        out = self.feature_cnn(x)  # out: 80
        out = torch.cat([out, avg_y], 1)
        out = out.reshape(-1, 1, 81).transpose(0, 1)
        out = F.relu(self.conv1(out, edge_index))
        for conv in self.convs:
            out = conv(out, edge_index)
        return self.lin(out)
