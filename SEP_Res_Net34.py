import torch
import torch.nn as nn
import time


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()
    def forward(self,x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += res
        x = self.relu2(x)
        return x




class ResBlockDown(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlockDown, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 2, 1, bias=False)  #因为用了下采样 所以缩小规模，减半
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()
        self.pool = nn.Conv2d(in_channel, out_channel, 1, 2, 0, bias=False)   #下带采样
    def forward(self, x):
        res = x
        res = self.pool(res)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += res
        x = self.relu2(x)
        return x




class CBS(nn.Module):
    def __init__(self, in_channel, out_channel, size, stride, padding=0):
        super(CBS, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, size, stride, padding)
        self.selu = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.selu(out)
        return out


class SE_Block(nn.Module):
    def __init__(self, in_channel, ratio):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.lin1 = nn.Linear(int(in_channel), int(in_channel / ratio))
        self.lin2 = nn.Linear(int(in_channel / ratio), int(in_channel))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        tmp = self.gap(x)
        tmp = tmp.view(x.size(0), -1)
        tmp = self.lin1(tmp)
        tmp = self.relu(tmp)
        tmp = self.lin2(tmp)
        tmp = self.sigmoid(tmp)
        tmp = tmp.view(x.size(0), x.size(1), 1, 1)
        return torch.mul(tmp, x)

class gap_conv2(nn.Module):
    def __init__(self, in_channel, out_channel, out_size):
        super(gap_conv2, self).__init__()
        self.conv1 = CBS(in_channel, in_channel, 3, 1, 1)
        self.conv2 = CBS(in_channel, out_channel, 3, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(out_size)
    def forward(self, x):
        out = self.gap(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return out


class up_conv2_bil(nn.Module):
    def __init__(self, out_size):
        super(up_conv2_bil, self).__init__()
        self.up = nn.UpsamplingBilinear2d(out_size)
    def forward(self, x):
        out = self.up(x)
        return out



class detail_attention(nn.Module):
    def __init__(self):
        super(detail_attention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(128)
        self.conv1 = nn.Conv2d(64, 3, 1, 1, 0)
        self.conv2 = up_conv2_bil(256)
        self.conv3 = gap_conv2(3, 3, 128)
        self.conv4 = gap_conv2(3, 3, 64)
        self.conv5 = nn.Conv2d(3, 1, 1, 1, 0)
    def forward(self, x):
        outx = self.pool(x)
        outx = self.conv1(outx)
        out2 = self.conv2(outx)
        out = self.conv3(out2)
        out = torch.add(out, outx)
        out = self.conv4(out)
        out = self.conv5(out)
        return out

class subject_attention1(nn.Module):
    def __init__(self):
        super(subject_attention1, self).__init__()
        self.one = nn.Conv2d(64, 2, 1, 1, 0)
        self.conv1 = nn.AdaptiveAvgPool2d(32)
        self.conv2 = gap_conv2(2, 2, 8)
        self.conv3 = up_conv2_bil(32)
        self.conv4 = gap_conv2(2, 1, 32)
    def forward(self, x):
        out1 = self.one(x)
        outx = self.conv1(out1)
        out = self.conv2(outx)
        out = self.conv3(out)
        out = torch.add(out, outx)
        out = self.conv4(out)
        return out

class subject_attention2(nn.Module):
    def __init__(self):
        super(subject_attention2, self).__init__()
        self.one = nn.Conv2d(64, 3, 1, 1, 0)
        self.conv1 = nn.AdaptiveAvgPool2d(32)
        self.conv2 = gap_conv2(3, 3, 8)
        self.conv3 = up_conv2_bil(32)
        self.conv4 = gap_conv2(3, 1, 32)
    def forward(self, x):
        out1 = self.one(x)
        outx = self.conv1(out1)
        out = self.conv2(outx)
        out = self.conv3(out)
        out = torch.add(out, outx)
        out = self.conv4(out)
        return out

class SEP_Res_Net(nn.Module):
    def __init__(self):
        super(SEP_Res_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d((3, 3), 2, 1)
        self.layer1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            ResBlockDown(64, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            ResBlockDown(128, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
        )
        self.layer4 = nn.Sequential(
            ResBlockDown(256, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512,29)


        self.SE1 = SE_Block(64, 12)
        self.SE2 = SE_Block(128, 18)

        self.sigmoid = nn.Sigmoid()
        self.detail_attention = detail_attention()
        self.lin = nn.Linear(1024, 64)
        self.maxpool = nn.AvgPool2d((3, 3), 2, 1)

        self.subject_attention1 = subject_attention1()
        self.subject_attention2 = subject_attention2()
        self.line = nn.Linear(256, 32)
        self.lin3 = nn.Linear(32, 1)
        self.lin4 = nn.Linear(32, 1)
        self.maxpool2 = nn.AvgPool2d((3, 3), 2, 1)

    def forward(self, x):
        time1 = time.time()
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        x2x = self.pool1(x1)
        x2x = self.layer1(x2x)

        # attention
        outx = self.detail_attention(x1)

        map1 = self.sigmoid(outx)
        x2 = torch.mul(map1, x2x)

        x2 = self.SE1(x2)

        x3x = self.layer2(x2)

        # attention
        branch1_outx = self.subject_attention1(x1)
        branch2_outx = self.subject_attention2(x2x)

        # select
        tmp1 = self.maxpool2(branch1_outx)
        tmp1 = tmp1.squeeze(1).flatten(1)
        tmp2 = self.maxpool2(branch2_outx)
        tmp2 = tmp2.squeeze(1).flatten(1)
        tmp_add = torch.add(tmp1, tmp2)
        tmp_add = self.line(tmp_add)
        tmp1 = self.lin3(tmp_add).unsqueeze(1).unsqueeze(1)
        tmp2 = self.lin4(tmp_add).unsqueeze(1).unsqueeze(1)
        branch1_out = torch.mul(branch1_outx, tmp1)
        branch2_out = torch.mul(branch2_outx, tmp2)


        map2 = torch.add(branch1_out, branch2_out)
        map2 = self.sigmoid(map2)
        x3 = torch.add(x3x, torch.mul(x3x, map2))
        x3 = self.SE2(x3)


        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        x5 = self.avgpool(x5)
        x5 = x5.view(x5.size(0), -1)
        x5 = self.fc(x5)
        time2 = time.time()

        return x5,x2

    def weight_init(self):    #初始化
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)


if __name__ == "__main__":
    device = torch.device('cpu')
    start = time.time()

    a = torch.ones(1, 3, 256, 256)
    model = SEP_Res_Net().to(device)
    model.eval()

    b = model(a)


    end = time.time()
    print(end-start)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))






