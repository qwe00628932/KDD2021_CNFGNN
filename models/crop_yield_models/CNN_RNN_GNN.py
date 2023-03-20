import torch
import torch.nn.functional as F
import torch.nn as nn
import models
from torch_geometric.nn import Sequential, JumpingKnowledge
from models.tcn import TemporalConvNet




class CNN_GNN(torch.nn.Module):
    def __init__(self):
        super(CNN_GNN, self).__init__()
        self.feature_cnn = FeatureCNN()
        self.lin1 = nn.Linear(80, 40)

    def forward(self, data):
        batch_size = data.shape[0]
        y = data[:, :, 2].reshape(-1, 1)
        x = data[:, :, 3:].reshape(-1, 1, 392)
        out = self.feature_cnn(x)  # out: 80


class CNN_GRU_vanilla(torch.nn.Module):
    def __init__(self):
        super(CNN_GRU_vanilla, self).__init__()
        self.feature_cnn = FeatureCNN()
        self.gru1 = nn.GRU(input_size=41, hidden_size=64, num_layers=1, batch_first=True)
        self.lin1 = nn.Linear(80, 40)

    def forward(self, data):
        batch_size = data.shape[0]
        # 这里将1改为0了
        y = data[:, :, 0].reshape(-1, 1)
        x = data[:, :, 3:].reshape(-1, 1, 392)
        out = self.feature_cnn(x)  # out: 80
        out = self.lin1(out)
        out = F.relu(out)
        x = torch.cat([out, y], 1)  # out: 41/81
        x = x.reshape(batch_size, 5, -1)
        _, hn = self.gru1(x[:, :, :]) # 64
        out = torch.cat([x[:, 4, :-1], hn.squeeze(0)], 1)
        return out # 104/80+64



class CNN_GRU(torch.nn.Module):
    def __init__(self):
        super(CNN_GRU, self).__init__()
        self.feature_cnn = FeatureCNN()
        self.gru1 = nn.GRU(input_size=41, hidden_size=64, num_layers=1, batch_first=True)
        self.lin1 = nn.Linear(80, 40)

    def forward(self, data):
        batch_size = data.shape[0]
        # print(f"batch_size: {batch_size}")
        y = data[:, :, 2].reshape(-1, 1)
        x = data[:, :, 3:].reshape(-1, 1, 392)
        out = self.feature_cnn(x)  # out: 80
        out = self.lin1(out)
        out = F.relu(out)
        x = torch.cat([out, y], 1)  # out: 41/81
        x = x.reshape(batch_size, 5, -1)
        _, hn = self.gru1(x[:, :4, :]) # 64
        out = torch.cat([x[:, 4, :-1], hn.squeeze(0)], 1)
        return out # 104/80+64


class CNN_GNN_GRU(torch.nn.Module):
    def __init__(self, device, num_layers=1, hidden_size=40,
                 conv_name="GATConv", dropout=0.0, heads=1, **kwargs):
        super(CNN_GNN_GRU, self).__init__()
        self.cnn = FeatureCNN()
        # self.gat = GATConv(40, 64)
        self.gat = models.ClusterGCNConv(40, 64)
        self.lin1 = nn.Linear(104, 40)
        self.lin2 = nn.Linear(64, 1)
        self.device = device

    def forward(self, data):
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        # batch_size = x.shape[0]
        # y = x[:, :, 2].reshape(-1, 1)
        x = x[:, :, 3:].reshape(-1, 1, 392)
        # out = self.feature_cnn(x)  # out: 80

        x = self.cnn_gru(x)
        x = self.lin1(x)
        x = F.relu(x)
        x_g = self.gat(x, edge_index)
        # x_g = F.relu(x_g)
        # x_g = self.gat2(x_g, edge_index)
        out = self.lin2(x_g)
        return out
        batch_size = data.shape[0]
        y = data[:, :, 2].reshape(-1, 1)
        x = data[:, :, 3:].reshape(-1, 1, 392)
        out = self.feature_cnn(x)  # out: 80
        out = self.lin1(out)
        out = F.relu(out)
        x = torch.cat([out, y], 1)  # out: 41

        x = x.reshape(batch_size, 5, -1)


        _, hn = self.gru1(x[:, :4, :])  # 64
        out = torch.cat([x[:, 4, :-1], hn.squeeze(0)], 1)



class CNN_RNN_GNN(torch.nn.Module):
    def __init__(self, device, num_layers=1, hidden_size=40,
                 conv_name="GATConv", dropout=0.0, mode='cat',
                 add_self_loop=False, **kwargs):
        super(CNN_RNN_GNN, self).__init__()
        self.device = device
        self.cnn_gru = CNN_GRU()
        conv_class = getattr(models, conv_name)
        self.conv1 = conv_class(104, hidden_size, dropout=dropout)
        self.convs = nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(conv_class(hidden_size, hidden_size, dropout=dropout, add_self_loop=add_self_loop))
        self.jump = JumpingKnowledge(mode)
        self.lin1 = nn.Linear(104, hidden_size)
        if mode == 'cat':
            self.lin2 = nn.Linear(num_layers * hidden_size, hidden_size)
        else:
            self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
        self.prelu = nn.PReLU()


    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        x = self.cnn_gru(x)
        x_g = self.conv1(x, edge_index)
        xs = [x_g]
        for conv in self.convs:
            x_g = conv(x_g, edge_index)
            x_g = self.prelu(x_g)
            xs += [x_g]
        x_g = self.jump(xs)
        x_g = F.relu(self.lin2(x_g))
        return self.lin3(x_g)


class RNN_GNN(torch.nn.Module):
    def __init__(self, device, num_layers=1, hidden_size=40,
                 conv_name="GATConv", dropout=0.0, heads=1, aggr='max', **kwargs):
        super(RNN_GNN, self).__init__()
        self.cnn_gru = CNN_GRU()
        conv_class = getattr(models, conv_name)
        self.conv1 = conv_class(104, hidden_size, dropout=dropout, heads=heads, aggr=aggr)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(conv_class(hidden_size, hidden_size, dropout=dropout, aggr=aggr))
        # self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size*heads, 1)
        self.device = device

    def forward(self, data):
        x = data.x.to(self.device)
        edge_index = data.feat_edge_index.to(self.device)
        x = self.cnn_gru(x)
        x_g = self.conv1(x, edge_index)
        for conv in self.convs:
            x_g = conv(x_g, edge_index)
        out = self.lin3(x_g)
        return out



class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class SFGCN(nn.Module):
    def __init__(self, device, num_layers=1,
                 hidden_size=40, conv_name="GCNConv",
                 dropout=0.5, heads=1, aggr="max", **kwargs):
        super(SFGCN, self).__init__()
        conv_class = getattr(models, conv_name)
        self.device = device
        self.dropout = dropout
        self.cnn_gru = CNN_GRU()
        self.sgcn1 = conv_class(104, hidden_size, dropout=dropout, heads=heads, aggr=aggr)
        self.sgcn2 = conv_class(104, hidden_size, dropout=dropout, heads=heads, aggr=aggr)
        self.cgcn = conv_class(104, hidden_size, dropout=dropout, heads=heads, aggr=aggr)
        self.sgcn1s = torch.nn.ModuleList()
        self.sgcn2s = torch.nn.ModuleList()
        self.cgcns = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.sgcn1s.append(conv_class(hidden_size, hidden_size, dropout=dropout, aggr=aggr))
            self.sgcn2s.append(conv_class(hidden_size, hidden_size, dropout=dropout, aggr=aggr))
            self.cgcns.append(conv_class(hidden_size, hidden_size, dropout=dropout, aggr=aggr))
        self.mlp = nn.Linear(hidden_size, 1)
        self.attention = Attention(hidden_size)

    def forward(self, data):
        x = data.x.to(self.device)
        sadj = data.edge_index.to(self.device)
        fadj = data.feat_edge_index.to(self.device)
        x = self.cnn_gru(x)
        emb1 = F.relu(self.sgcn1(x, sadj))
        # emb1 = self.sgcn1(x, sadj)
        emb1 = F.dropout(emb1, self.dropout, training=self.training)
        com1 = F.relu(self.cgcn(x, sadj))
        # com1 = self.cgcn(x, sadj)
        com1 = F.dropout(com1, self.dropout, training=self.training)
        com2 = F.relu(self.cgcn(x, fadj))
        # com2 = self.cgcn(x, fadj)
        com2 = F.dropout(com2, self.dropout, training=self.training)
        emb2 = F.relu(self.sgcn2(x, fadj))
        # emb2 = self.sgcn2(x, fadj)
        emb2 = F.dropout(emb2, self.dropout, training=self.training)
        for conv in self.sgcn1s:
            emb1 = conv(emb1, sadj)
        for conv in self.cgcns:
            com1 = conv(com1, sadj)
        for conv in self.cgcns:
            com2 = conv(com2, sadj)
        for conv in self.sgcn2s:
            emb2 = conv(emb2, sadj)
        Xcom = (com1 + com2) / 2
        emb = torch.stack([emb1, emb2, Xcom], dim=1)
        emb, att = self.attention(emb)
        output = self.mlp(emb)
        return output, att, emb1, com1, com2, emb2, emb

    # def reset_parameters(self):
    #     self.cnn_gru.reset_parameters()
    #     self.sgcn1.reset_parameters()
    #     self.sgcn2.reset_parameters()
    #     self.cgcn.reset_parameters()
    #     for conv in self.sgcn1s:
    #         conv.reset_parameters()
    #     for conv in self.sgcn2s:
    #         conv.reset_parameters()
    #     for conv in self.cgcns:
    #         conv.reset_parameters()
    #     self.mlp.reset_parameters()


class FixSFGCN(SFGCN):
    def __init__(self, device, num_layers=1,
                 hidden_size=40, conv_name="GCNConv",
                 dropout=0.5, heads=1, aggr="max", **kwargs):
        super(FixSFGCN, self).__init__(
            device=device, num_layers=num_layers,
            hidden_size=hidden_size, conv_name=conv_name,
            dropout=dropout, heads=heads, aggr=aggr
        )
        self.mlp = nn.Linear(3*hidden_size, 1)

    def forward(self, data):
        x = data.x.to(self.device)
        sadj = data.edge_index.to(self.device)
        fadj = data.feat_edge_index.to(self.device)
        x = self.cnn_gru(x)
        emb1 = F.relu(self.sgcn1(x, sadj))
        # emb1 = self.sgcn1(x, sadj)
        emb1 = F.dropout(emb1, self.dropout, training=self.training)
        com1 = F.relu(self.cgcn(x, sadj))
        # com1 = self.cgcn(x, sadj)
        com1 = F.dropout(com1, self.dropout, training=self.training)
        com2 = F.relu(self.cgcn(x, fadj))
        # com2 = self.cgcn(x, fadj)
        com2 = F.dropout(com2, self.dropout, training=self.training)
        emb2 = F.relu(self.sgcn2(x, fadj))
        # emb2 = self.sgcn2(x, fadj)
        emb2 = F.dropout(emb2, self.dropout, training=self.training)
        for conv in self.sgcn1s:
            emb1 = conv(emb1, sadj)
        for conv in self.cgcns:
            com1 = conv(com1, sadj)
        for conv in self.cgcns:
            com2 = conv(com2, fadj)
        for conv in self.sgcn2s:
            emb2 = conv(emb2, fadj)
        Xcom = (com1 + com2) / 2
        # emb = torch.stack([emb1, emb2, Xcom], dim=1)
        # emb, att = self.attention(emb)

        emb = torch.cat([emb1, emb2, Xcom], dim=1)
        output = self.mlp(emb)
        return output, 0, emb1, com1, com2, emb2, emb


class SFGCN_v2(nn.Module):
    def __init__(self, device, num_layers=1,
                 hidden_size=40, conv_name="GCNConv",
                 dropout=0.0, heads=1, aggr="max", **kwargs):
        super(SFGCN_v2, self).__init__()
        conv_class = getattr(models, conv_name)
        self.device = device
        self.dropout = dropout
        self.cnn_gru = CNN_GRU()
        self.sgcn1 = conv_class(104, hidden_size, dropout=dropout, heads=heads, aggr=aggr)
        self.sgcn2 = conv_class(104, hidden_size, dropout=dropout, heads=heads, aggr=aggr)
        self.cgcn = conv_class(104, hidden_size, dropout=dropout, heads=heads, aggr=aggr)
        self.sgcn1s = torch.nn.ModuleList()
        self.sgcn2s = torch.nn.ModuleList()
        self.cgcns = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.sgcn1s.append(conv_class(hidden_size, hidden_size, dropout=dropout, aggr=aggr))
            self.sgcn2s.append(conv_class(hidden_size, hidden_size, dropout=dropout, aggr=aggr))
            self.cgcns.append(conv_class(hidden_size, hidden_size, dropout=dropout, aggr=aggr))
        self.mlp = nn.Linear(hidden_size, 1)
        self.attention = Attention(hidden_size)

    def forward(self, data):
        x = data.x.to(self.device)
        sadj = data.edge_index.to(self.device)
        fadj = data.feat_edge_index.to(self.device)
        cadj = data.merged_edge_index.to(self.device)
        x = self.cnn_gru(x)
        emb1 = F.relu(self.sgcn1(x, sadj))
        emb1 = F.dropout(emb1, self.dropout, training=self.training)
        com = F.relu(self.cgcn(x, cadj))
        com = F.dropout(com, self.dropout, training=self.training)
        emb2 = F.relu(self.sgcn2(x, fadj))
        emb2 = F.dropout(emb2, self.dropout, training=self.training)
        for conv in self.sgcn1s:
            emb1 = conv(emb1, sadj)
        for conv in self.cgcns:
            com = conv(com, cadj)
        for conv in self.sgcn2s:
            emb2 = conv(emb2, fadj)
        emb = torch.stack([emb1, emb2, com], dim=1)
        emb, att = self.attention(emb)
        output = self.mlp(emb)
        return output, att, emb1, com, com, emb2, emb

    def reset_parameters(self):
        self.cnn_gru.reset_parameters()
        self.sgcn1.reset_parameters()
        self.sgcn2.reset_parameters()
        self.cgcn.reset_parameters()
        for conv in self.sgcn1s:
            conv.reset_parameters()
        for conv in self.sgcn2s:
            conv.reset_parameters()
        for conv in self.cgcns:
            conv.reset_parameters()
        self.mlp.reset_parameters()


class SFGCN_v3(FixSFGCN):
    def __init__(self, device, num_layers=1,
                 hidden_size=40, conv_name="GCNConv",
                 dropout=0.0, heads=1, aggr="max", **kwargs):
        super(SFGCN_v3, self).__init__(
            device=device, num_layers=num_layers,
            hidden_size=hidden_size, conv_name=conv_name,
            dropout=dropout, heads=heads, aggr=aggr
        )


class SFGCN_v4(FixSFGCN):
    def __init__(self, device, num_layers=1,
                 hidden_size=40, conv_name="GCNConv",
                 dropout=0.0, heads=1, aggr="max", **kwargs):
        super(SFGCN_v4, self).__init__(
            device=device, num_layers=num_layers,
            hidden_size=hidden_size, conv_name=conv_name,
            dropout=dropout, heads=heads, aggr=aggr
        )


class TFSF_TF(nn.Module):
    def __init__(self, device, num_layers=1,
                 hidden_size=40, conv_name="GCNConv",
                 dropout=0.0, heads=1, aggr="max", **kwargs):
        super(TFSF_TF, self).__init__()
        conv_class = getattr(models, conv_name)
        self.device = device
        self.dropout = dropout
        self.cnn_gru = CNN_GRU()
        self.sgcn1 = conv_class(104, hidden_size, dropout=dropout, heads=heads, aggr=aggr)
        self.sgcn2 = conv_class(104, hidden_size, dropout=dropout, heads=heads, aggr=aggr)
        self.sgcn1s = torch.nn.ModuleList()
        self.sgcn2s = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.sgcn1s.append(conv_class(hidden_size, hidden_size, dropout=dropout, aggr=aggr))
            self.sgcn2s.append(conv_class(hidden_size, hidden_size, dropout=dropout, aggr=aggr))
        self.mlp = nn.Linear(hidden_size*2, 1)

    def forward(self, data):
        x = data.x.to(self.device)
        sadj = data.edge_index.to(self.device)
        fadj = data.feat_edge_index.to(self.device)
        x = self.cnn_gru(x)
        emb1 = F.relu(self.sgcn1(x, sadj))
        # emb1 = self.sgcn1(x, sadj)
        emb1 = F.dropout(emb1, self.dropout, training=self.training)
        emb2 = F.relu(self.sgcn2(x, fadj))
        # emb2 = self.sgcn2(x, fadj)
        emb2 = F.dropout(emb2, self.dropout, training=self.training)
        for conv in self.sgcn1s:
            emb1 = conv(emb1, sadj)
        for conv in self.sgcn2s:
            emb2 = conv(emb2, fadj)

        emb = torch.cat([emb1, emb2], dim=1)
        output = self.mlp(emb)
        return output
