from __future__ import print_function
import zipfile
import os

import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 32 x 32 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # for rgb data
    # transforms.Normalize([0.5484779, 0.5317458, 0.5059532], [0.3107016, 0.30501202, 0.3177048])
    # for lab data
    transforms.Normalize([0.52016187, 0.508488, 0.55626583], [0.06621674, 0.04544095, 0.2970322])
])

resnet_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

resnet_test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

vgg_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

vgg_test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def initialize_data(folder):
    # train_zip = folder + '/train_images.zip'
    # test_zip = folder + '/test_images.zip'
    # if not os.path.exists(train_zip) or not os.path.exists(test_zip):
    #     raise(RuntimeError("Could not find " + train_zip + " and " + test_zip
    #           + ', please download them from https://www.kaggle.com/c/nyu-cv-fall-2018/data '))
    # extract train_data.zip to train_data
    train_folder = folder + '/train'
    # if not os.path.isdir(train_folder):
    #     print(train_folder + ' not found, extracting ' + train_zip)
    #     zip_ref = zipfile.ZipFile(train_zip, 'r')
    #     zip_ref.extractall(folder)
    #     zip_ref.close()
    # extract test_data.zip to test_data
    test_folder = folder + '/test'
    # if not os.path.isdir(test_folder):
    #     print(test_folder + ' not found, extracting ' + test_zip)
    #     zip_ref = zipfile.ZipFile(test_zip, 'r')
    #     zip_ref.extractall(folder)
    #     zip_ref.close()

    # make validation_data by using images 00000*, 00001* and 00002* in each class
    val_folder = folder + '/val'
    if not os.path.isdir(val_folder):
        print(val_folder + ' not found, making a validation set')
        os.mkdir(val_folder)
        for dirs in os.listdir(train_folder):
            # if dirs.startswith('000'):
            os.mkdir(val_folder + '/' + dirs)
            for f in os.listdir(train_folder + '/' + dirs):
                if f.startswith('image_0001') or f.startswith('image_0005') or f.startswith('image_0010'):
                    # move file to validation folder
                    os.rename(train_folder + '/' + dirs + '/' + f, val_folder + '/' + dirs + '/' + f)
