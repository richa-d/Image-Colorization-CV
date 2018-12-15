import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB has 43 classes

class Net(nn.Module):
    def _get_padding(self, size, kernel_size, stride):
        padding = ((size - 1) * (stride - 1) + (kernel_size - 1)) //2
        return padding

    def __init__(self):
        super(Net, self).__init__()

        pad_32 = self._get_padding(128,3,2)
        pad_31 = self._get_padding(128,3,1)
        pad_22 = self._get_padding(128,2,2)

        # print("Padding: 32")
        # print(pad_32)
        # print("Padding: 31")
        # print(pad_31)
        # print("Padding: 22")
        # print(pad_22)
        self.conv0 = nn.Conv2d(1,8, kernel_size=3, stride=2)#, padding=pad_32)
        self.batch0 = nn.BatchNorm2d(8)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(500, 50)
        # self.fc2 = nn.Linear(50, nclasses)

        # more conv layers
        # 1 x 128 x 128 -> 15 x 124 x 124
        self.conv1 = nn.Conv2d(8, 8, kernel_size=3)#, padding=pad_31)
        self.batch1 = nn.BatchNorm2d(8)

        # 15 x 124 x 124 -> 50 x 12 x 12
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)#, padding=pad_31)
        self.batch2 = nn.BatchNorm2d(16)

        # 50 x 12 x 12 -> 75 x 8 x 8
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=2)#, padding=pad_32)
        # self.conv3_drop = nn.Dropout2d()
        self.batch3 = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 32, kernel_size=3)#, padding=pad_31)
        self.batch4 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=2)#, padding=pad_32)
        self.batch5 = nn.BatchNorm2d(32)
        
        # 75 x 8 x 8 -> 100 x 2 x 2
        self.upsample1 = nn.Upsample(scale_factor=2)
        
        self.conv6 = nn.Conv2d(32, 32, kernel_size=2, stride=2)#, padding=pad_32)
        self.batch6 = nn.BatchNorm2d(32)

        self.upsample2 = nn.Upsample(scale_factor=2)

        self.conv7 = nn.Conv2d(32, 16, kernel_size=2, stride=2)#, padding=pad_22)
        self.batch7 = nn.BatchNorm2d(16)

        self.upsample3 = nn.Upsample(scale_factor=2)

        self.conv8 = nn.Conv2d(16, 2, kernel_size=2, stride=2)#, padding=pad_22)
        self.batch8 = nn.BatchNorm2d(2)



        # self.conv4 = nn.Conv2d(128, 2, stride=1, kernel_size=1, padding=pad_11)
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
        x = F.relu(self.batch0(self.conv0(x)))
        x = F.pad( x, (32,33,32,33) )
        # print("Conv0 ")
        # print(x.size())

        x = F.relu(self.batch1(self.conv1(x)))
        x = F.pad( x, (1,1,1,1) )
        # print("Conv1 ")
        # print(x.size())

        x = F.relu(self.batch2(self.conv2(x)))
        x = F.pad( x, (1,1,1,1) )
        # print("Conv2 ")
        # print(x.size())

        x = F.relu(self.batch3(self.conv3(x)))
        x = F.pad( x, (32,33,32,33) )
        # print("Conv3 ")
        # print(x.size())

        x = F.relu(self.batch4(self.conv4(x)))
        x = F.pad( x, (1,1,1,1) )
        # print("Conv4 ")
        # print(x.size())

        x = F.relu(self.batch5(self.conv5(x)))
        x = F.pad( x, (32,33,32,33) )
        # print("Conv5 ")
        # print(x.size())

        # x = F.relu(F.max_pool2d(self.conv4_drop(self.batch4(self.conv4(x))), 2))
        x = self.upsample1(x)
        x = F.relu(self.batch6(self.conv6(x)))
        # print("Conv6 ")
        # print(x.size())
        x = self.upsample2(x)
        x = F.relu(self.batch7(self.conv7(x)))
        # print("Conv7 ")
        # print(x.size())
        x = self.upsample3(x)
        x = F.relu(self.batch8(self.conv8(x)))
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
