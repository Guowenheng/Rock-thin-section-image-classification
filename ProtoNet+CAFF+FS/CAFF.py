import torch
from torch import nn
from torch.nn import functional as F

class CAFF(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 dimension=1,
                 sub_sample=False,
                 bn_layer=False):
        super(CAFF, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g_sar = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        self.g_opt = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta_sar = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.theta_opt = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi_sar = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.phi_opt = conv_nd(in_channels=self.in_channels,
                               out_channels=self.inter_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.norm = nn.LayerNorm(768)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(768, 256)
        nn.init.trunc_normal_(self.head.weight, std=.02)
        nn.init.constant_(self.head.bias, 0)
        self.gamma_sar = nn.Parameter(torch.zeros(1))
        self.gamma_opt = nn.Parameter(torch.ones(1))
        self.gamma_att = nn.Parameter(torch.ones(1))
        if sub_sample:
            self.g_sar = nn.Sequential(self.g_sar, max_pool_layer)
            self.g_opt = nn.Sequential(self.g_opt, max_pool_layer)
            self.phi_sar = nn.Sequential(self.phi_sar, max_pool_layer)
            self.phi_opt = nn.Sequential(self.phi_opt, max_pool_layer)

    def forward(self, sar, opt):

        batch_size = sar.size(0)

        g_x = self.g_sar(sar).view(batch_size, self.inter_channels, -1)  # [bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta_sar(sar).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi_sar(sar).view(batch_size, self.inter_channels, -1)

        f_x = torch.matmul(theta_x, phi_x)
        f_div_C_x = F.softmax(f_x, dim=-1)
        del theta_x ,phi_x
        g_y = self.g_opt(opt).view(batch_size, self.inter_channels, -1)  # [bs, c, w*h]
        g_y = g_y.permute(0, 2, 1)

        theta_y = self.theta_opt(opt).view(batch_size, self.inter_channels, -1)
        theta_y = theta_y.permute(0, 2, 1)

        phi_y = self.phi_opt(opt).view(batch_size, self.inter_channels, -1)

        f_y = torch.matmul(theta_y, phi_y)
        del theta_y,phi_y
        f_div_C_y = F.softmax(f_y, dim=-1)
        # print(f_div_C_x.shape, f_div_C_y.shape)
        ##### [1, 4096, 1024]    [1, 4096, 1024]
        # print(f_div_C_x.shape, f_div_C_y.shape)
        y = torch.einsum('ijk,ijk->ijk', [f_div_C_x, f_div_C_y])
        # y = f_div_C_x * f_div_C_y

        # print(y.shape, g_x.shape, g_y.shape)
        y_x = torch.matmul(y, g_x)
        y_y = torch.matmul(y, g_y)
        del g_x,g_y
        # print(y_x.shape, y_y.shape)
        y = y_x * y_y
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *sar.size()[2:])
        y = self.W(y)#(200,49,768)
        # y=torch.cat([y,opt],dim=2)

        y = self.gamma_att*y+self.gamma_opt*opt+self.gamma_sar*sar
        # y = self.gamma_att*y+self.gamma_sar*sar
        y = self.avgpool(y.transpose(1, 2))  # [B, C, 1] (200,768,1)

        y = torch.flatten(y, 1)#(200,768)

        y = self.head(y)#(200,256)
        return y

if __name__ == '__main__':
    model = CAFF(in_channels=128)
    model.train()
    sar = torch.randn(2, 128, 64,64)
    opt = torch.randn(2, 128, 64,64)
    print(model)
    print("input:", sar.shape, opt.shape)
    print("output:", model(sar, opt).shape)