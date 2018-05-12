# coding=utf-8
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class CoarseNet(nn.Module):
    def __init__(self, init_weights=True):
        super(CoarseNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=4, padding=0)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(29568, 127*42)
        self.fc2 = nn.Linear(127*42, 127*42)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):

       ''' x = self.cnn(x)

        x = x.view(x.size(0), -1)'''
        x = F.max_pool2d(self.conv1(x), 3, stride=2)
        x = F.max_pool2d(self.conv2(x), 2, stride=1)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 1, 42, 127)
        #VariableSizeInspector()
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class FineNet(nn.Module):
    def __init__(self, init_weights=True):
        super(FineNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 63, kernel_size=9, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=1)
        if init_weights:
            self._initialize_weights()

    def forward(self, x, y):
        x = F.max_pool2d(self.conv1(x), 3, stride=2)
        x = torch.cat((x, y), 1)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()