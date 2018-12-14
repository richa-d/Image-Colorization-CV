import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB has 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # default
        # self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(500, 50)
        # self.fc2 = nn.Linear(50, nclasses)

        # more conv layers
        # 1 x 128 x 128 -> 15 x 124 x 124
        self.conv1 = nn.Conv2d(1, 32, stride=2, kernel_size=2)
        self.batch1 = nn.BatchNorm2d(32)

        # 15 x 124 x 124 -> 50 x 12 x 12
        self.conv2 = nn.Conv2d(32, 64, stride=2, kernel_size=2)
        self.batch2 = nn.BatchNorm2d(64)

        # 50 x 12 x 12 -> 75 x 8 x 8
        self.conv3 = nn.Conv2d(64, 128, stride=2, kernel_size=2)
        # self.conv3_drop = nn.Dropout2d()
        self.batch3 = nn.BatchNorm2d(128)

        # 75 x 8 x 8 -> 100 x 2 x 2
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)

        self.conv4 = nn.Conv2d(128, 2, stride=1, kernel_size=1)
        # self.conv4_drop = nn.Dropout2d()
        # self.batch4 = nn.BatchNorm2d(128)

        # self.conv4_drop = nn.Dropout2d()
        # self.batch4 = nn.BatchNorm2d(100)
        #
        # # size = 100 * 2 * 2
        # self.fc1 = nn.Linear(400, 50)
        # self.fc2 = nn.Linear(50, nclasses)


    def forward(self, x):
        # default
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 500)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        # return F.log_softmax(x)

        # more conv layers
        x = F.relu(self.batch1(self.conv1(x)))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.batch3(self.conv3(x)))
        # x = F.relu(F.max_pool2d(self.conv4_drop(self.batch4(self.conv4(x))), 2))
        x= self.upsample1(x)
        x= self.upsample2(x)
        x= self.upsample3(x)
        x= F.relu(self.conv4(x))
        # x = F.leaky_relu(self.batch1(self.conv1(x)))
        # x = F.leaky_relu(F.max_pool2d(self.batch2(self.conv2(x)), 2))
        # x = F.leaky_relu(self.batch3(self.conv3(x)))
        # x = F.leaky_relu(F.max_pool2d(self.conv4_drop(self.batch4(self.conv4(x))), 2))
        # x = x.view(-1, 400)
        # x = F.relu(self.fc1(x))
        # x = F.leaky_relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        return x
