import torch
import torch.nn as nn
import time

# #### 先定义不下带采样的ResBlock

# In[2]:


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
        x += res         #残差连接
        x = self.relu2(x)
        return x


# #### 再定义下带采样的ResBlockDown

# In[3]:


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
        x += res         #残差连接
        x = self.relu2(x)
        return x


# #### 搭建模型 

# In[4]:


class Res_Net(nn.Module):
    def __init__(self):
        super(Res_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.conv2 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
        )
        self.conv3 = nn.Sequential(
            ResBlockDown(64, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
        )
        self.conv4 = nn.Sequential(
            ResBlockDown(128, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
        )
        self.conv5 = nn.Sequential(
            ResBlockDown(256, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512,29)
    def forward(self, x):
        time1 = time.time()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        xx = self.conv2(x)
        x = self.conv3(xx)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        time2 = time.time()
        return x, time2
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
    # result = model(input)
    # torch.cuda.synchronize()
    # end = time.time()
    a = torch.ones(1, 3, 256, 256)
    model = Res_Net().to(device)
    model.eval()

    # print(model)

    b = model(a)


    end = time.time()
    print(end-start)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    # print(b, b.shape)
    # print(model.out2, model.out2.shape)





