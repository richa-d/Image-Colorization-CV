from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
import matplotlib.pyplot as plt
# %matplotlib inline
from skimage import color
import numpy as np
import pytorch_colors as colors
from PIL import Image


# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
cuda = torch.cuda.is_available()
if cuda:
        torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)

### Data Initialization and Loading
from data import initialize_data, data_transforms, vgg_train_transform, resnet_train_transform # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train',
                         transform=data_transforms),
                         # transform=vgg_train_transform),
                         # transform=resnet_train_transform),
    batch_size=args.batch_size, shuffle=True, num_workers=1)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val',
                         transform=data_transforms),
                         # transform=vgg_train_transform),
                         # transform=resnet_train_transform),
    batch_size=args.batch_size, shuffle=False, num_workers=1)


# dataloader = torch.utils.data.DataLoader(*torch_dataset*, batch_size=4096, shuffle=False, num_workers=4)

####################################################################################
############### CALCULATING THE MEAN AND STANDARD DEVIATION ########################
####################################################################################
# pop_mean = []
# pop_std0 = []
# pop_std1 = []
# for i, (data,target) in enumerate(train_loader, 0):
#     # shape (batch_size, 3, height, width)
#     numpy_image = (Variable(data)).numpy()
    
#     # shape (3,)
#     batch_mean = np.mean(numpy_image, axis=(0,2,3))
#     batch_std0 = np.std(numpy_image, axis=(0,2,3))
#     batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
    
#     pop_mean.append(batch_mean)
#     pop_std0.append(batch_std0)
#     pop_std1.append(batch_std1)

# # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
# pop_mean = np.array(pop_mean).mean(axis=0)
# pop_std0 = np.array(pop_std0).mean(axis=0)
# pop_std1 = np.array(pop_std1).mean(axis=0)

# print(pop_mean)
# print(pop_std0)
# print(pop_std1)



### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
# from model_new import Net
from model import Net
model = Net()
# import torchvision.models as models
# model = models.resnet18(pretrained=False)
# model.cuda()
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 43)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
criterion = nn.MSELoss()

def transform_lab():
    count=0
    for batch_idx, (data, target) in enumerate(train_loader):
        count+=target.size()[0]
        data, target = (Variable(data)), (Variable(target))
        print("Data ",end="")
        print(data.size())
        print("Target ",end="")
        print(target.size())
        data_lab = colors.rgb_to_lab(data)
        print("LAB ",end="")
        print(data_lab.size())
        print("L ",end="")
        data_l = data_lab[:,0,:,:]#.unsqueeze(0)
        print(data_l.size())
        print("AB ",end="")
        data_ab = data_lab[:,1:,:,:]
        print(data_ab.size())
        # print("A",end="")
        # data_a = data_lab[:,:,0,:]#.unsqueeze(0)
        # print(data_a.size())
        # print("B",end="")
        # data_b = data_lab[:,:,:,0]#.unsqueeze(0)
        # print(data_b.size())
        # new1= colors.lab_to_rgb(data_lab)
        new = data_lab[0].permute(1, 2, 0)
        plt.imshow(data_l[0])
        plt.show()
        target = data_lab
    print(count)
# transform_lab()

def train(epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = (Variable(data)), (Variable(target))
        # print("Data",end="")
        # print(data.size())
        # print("Target",end="")
        # print(target.size())
        optimizer.zero_grad()
        # data_np = data.numpy()
        # data_lab_np = color.rgb2lab(data_np)
        # data_lab = torch.from_numpy(data_lab_np
        data_lab = colors.rgb_to_lab(data)
        # print("LAB",end="")
        # print(data_lab.size())
        data_l = data_lab[:,0,:,:].unsqueeze(1)
        # print("L",end="")
        # print(data_l.size())
        data_ab = data_lab[:,1:,:,:]
        # print(data_ab)
        target = data_ab
        # print("AB",end="")
        # print(target.size())
        output = model(data_l)
        # print("Output",end="")
        # print(output.size())
        # output = model(data)
        # loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        loss.backward()
        print("Loss: ",end="")
        print(loss)
        optimizer.step()
        # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        # correct += pred.eq(target.data.view_as(pred)).sum()
        if  batch_idx % args.log_interval == 0:
            # print("Train epoch: ")
            # print(epoch)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    return correct

# newimage_array=torch.empty()
newimage_array=[]
def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        with torch.no_grad():
            data, target = (Variable(data, volatile=True)), (Variable(target))
            data_lab = colors.rgb_to_lab(data)
            data_l = data_lab[:,0,:,:].unsqueeze(1)
            output_l=data_lab[:,0,:,:]
            # print("L",end="")
            # print(data_l.size())
            data_ab = data_lab[:,1:,:,:]
            # data_lab = colors.rgb_to_lab(data)
            # data_l = data_lab[0]
            target = data_ab.float()
            output = model(data_l)
            # data_l = data_lab[0]
            # target = data_lab
            # output = model(data_l)
            # output = model(data)
            # validation_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            # validation_loss += criterion(output, target).data[0]
            # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            # correct += pred.eq(target.data.view_as(pred)).sum()
            #Compare original LAB vs data_l+output(for ab)
            output_a=output[:,0,:,:].detach()
            output_b=output[:,1,:,:].detach()
            # print("Actual_L",end="")
            # print(output_l.size())
            # print("Predicted_A",end="")
            # print(output_a.size())
            # print("Predicted_B",end="")
            # print(output_b.size())
            final_output=torch.stack([output_l,output_a,output_b],dim=1)
            # print("Predicted_Image",end="")
            # print(final_output.size())
            # new = final_output[0].permute(1, 2, 0)
            # plt.imshow(new)
            # plt.show()
            newimage_array.append(final_output[0])
    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return 100. * correct / len(val_loader.dataset)

tr_acc = []
val_acc = []

for epoch in range(1, args.epochs + 1):
    tr_acc.append(100.0 * train(epoch) / len(train_loader.dataset))
    val_acc.append(validation())
    model_file = 'model_plotting' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model' + model_file + '` to generate the Kaggle formatted csv file')

i=0
for new_image in newimage_array:
    i+=1
    # print(new_image.size())
    # print(new_image.type())
    rgb_pred_image = colors.lab_to_rgb(new_image.unsqueeze(0))
    # print(rgb_pred_image.size())
    # new = rgb_pred_image[0].permute(1, 2, 0)
    # plt.imshow(new)
    # plt.show()
    s=args.data+"/img/iimages"+str(i)+".jpeg"
    utils.save_image(rgb_pred_image[0],s)


# plt.plot(tr_acc)
# plt.title("Training accuracy")
# plt.show()
#
# plt.plot(val_acc)
# plt.title("Validation accuracy")
# plt.show()
