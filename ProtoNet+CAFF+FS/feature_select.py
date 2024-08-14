import torch
from torch import nn
from torch.nn import functional as F


class Chanal(nn.Module):
    def __init__(self, shots):
        super(Chanal, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (shots, 1), padding=(shots // 2, 0))

        self.conv2 = nn.Conv2d(32, 64, (shots, 1), padding=(shots // 2, 0))
        self.conv_final = nn.Conv2d(64, 1, (shots, 1), stride=(shots, 1))
        self.drop = nn.Dropout()
        self.conv3 = nn.Conv2d(1, 32, (3, 1), padding=(3 // 2, 0))
        self.conv4 = nn.Conv2d(32, 1, (3, 1), stride=(3, 1))
    def forward(self, protot):
        protot = protot.to(torch.device("cuda"))
        protot_2 = protot.squeeze()
        class_number = protot.shape[0]
        protot = F.relu(self.conv1(protot))  # (B * N, 32, K, D)
        protot = F.relu(self.conv2(protot))  # (B * N, 64, K, D)
        protot = self.drop(protot)
        protot_2 = torch.stack([protot_2[index, :].mean(0) for index in range(class_number)], dim=0)
        protot_mean = protot_2.mean(0)
        protot_min,_ = protot_2.min(0)
        protot_max,_ = protot_2.max(0)
        protot_2=torch.cat([protot_mean.unsqueeze(0),protot_min.unsqueeze(0),protot_max.unsqueeze(0)],dim=0)
        protot_2=F.relu(self.conv3(protot_2.unsqueeze(0)))
        protot_2=F.sigmoid(self.conv4(protot_2))    #
        fea_att_score = self.conv_final(protot)

        fea_att_score = torch.mul(protot_2, fea_att_score)
        fea_att_score = F.relu(fea_att_score)
        fea_att_score = fea_att_score.squeeze()
        return fea_att_score


if __name__ == '__main__':
    model = Chanal(shots=5)
    model.train()
    sar = torch.randn(20, 1, 5, 32)
    opt = torch.randn(20, 1, 5, 32)
    print(model)
    print("input:", sar.shape, opt.shape)
    print("output:", model(sar).shape)