import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.core.numeric import outer
from torchsummary import summary
from prepare import *

device = torch.device("cuda")

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 2)
        self.conv2 = nn.Conv2d(32, 192, 3, 2)
        self.dense1 = nn.Linear(5760, 120)   # dropout并没有使得神经元数目变少，而是使某些神经元失效而已
        self.dropout2 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(120, 72)
        self.dense3 = nn.Linear(72, 40)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 5760)       # 会出现输出维度与期望的不一致现象，一般是这里的展平没弄好
        x = F.relu(self.dense1(x))
        x = self.dropout2(x)
        x = F.relu(self.dense2(x))
        x = F.softmax(self.dense3(x))

        return x

if __name__ == '__main__':
    model = Model().to(device)
    summary(model, (1, 92, 112))
    input = torch.ones((1, 1, 92, 112)).to(device)
    output = model(input)
    print(output.shape)
