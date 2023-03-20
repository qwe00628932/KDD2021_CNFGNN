import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureCNN(torch.nn.Module):
    def __init__(self):
        self.layers_w_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=8, kernel_size=9),
                nn.ReLU(),
                nn.AvgPool1d(2, stride=2),
                nn.Conv1d(in_channels=8, out_channels=12, kernel_size=3),
                nn.ReLU(),
                nn.AvgPool1d(2, stride=2),
                nn.Conv1d(in_channels=12, out_channels=16, kernel_size=3),
                nn.ReLU(),
                nn.AvgPool1d(2, stride=2),
                nn.Conv1d(in_channels=16, out_channels=20, kernel_size=3),
                nn.ReLU(),
                nn.AvgPool1d(2, stride=2)
            ) for _ in range(6)
        ])
        self.layers_s_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3),
                nn.ReLU(),
                nn.AvgPool1d(2, stride=2),
                nn.Conv1d(in_channels=3, out_channels=6, kernel_size=2),
                nn.ReLU(),
            ) for _ in range(11)
        ])

    def forward(self, data):
        """
        data: shape-(batch_size, time, n_feature)
        """
        # x_j = data.unsqueeze(1)
        x_j = data
        W1 = x_j[:, :, :52]
        W2 = x_j[:, :, 52:104]
        W3 = x_j[:, :, 104:156]
        W4 = x_j[:, :, 156:208]
        W5 = x_j[:, :, 208:260]
        W6 = x_j[:, :, 260:312]
        bdod = x_j[:, :, 312:318]
        cec = x_j[:, :, 318:324]
        cfvo = x_j[:, :, 324:330]
        clay = x_j[:, :, 330:336]
        nitrogen = x_j[:, :, 336:342]
        ocd = x_j[:, :, 342:348]
        ocs = x_j[:, :, 348:354]
        phh2o = x_j[:, :, 354:360]
        sand = x_j[:, :, 360:366]
        silt = x_j[:, :, 366:372]
        soc = x_j[:, :, 372:378]
        w_features = [W1, W2, W3, W4, W5, W6]
        s_features = [bdod, cec, cfvo, clay, nitrogen, ocd, ocs, phh2o, sand, silt, soc]
        w_list = [self.layers_w_list[idx](w_feature.to(self.mydevice)) for idx, w_feature in enumerate(w_features)]
        s_list = [self.layers_s_list[idx](s_feature.to(self.mydevice)) for idx, s_feature in enumerate(s_features)]
        w_out = torch.cat(w_list, 1).squeeze(2)
        w_out = self.lin_w(w_out)
        w_out = F.relu(w_out)
        s_out = torch.cat(s_list, 1).squeeze(2)
        s_out = self.lin_s(s_out)
        s_out == F.relu(s_out)
        out = torch.cat([w_out, s_out], 1)
        return out

