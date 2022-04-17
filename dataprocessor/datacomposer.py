from cProfile import label
from cgi import test
import imp
from random import shuffle
import random
from re import X
from dataprocessor.datareader import get_CIFAR10_dataloader
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
import hyperparameters as HP
import os
from torchvision import transforms
from PIL import Image
from utils import single_channel_to_3_channel

def get_CIFAR10_re_dataset():
    '''
    based on the specific task, re-organize the labels of the CIFAR10 data
    divide the instances into 2 categories, where 0 for 'machine', 1 for 'animal'
    return the dataloader and the new labels (i.e. macro labels) on both the training set and test set

    the size of the macro_labels is (batch_num, 1), where batch_num = instance_num / batch_size
    '''


    trainloader, testloader = get_CIFAR10_dataloader() # get the dataloader (batchsize=HP.batch_size)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    machine = (0,1,8,9)
    animal = (2,3,4,5,6,7)
    train_macro_labels = []
    test_macro_labels = []

    print('operating training data...')
    for images, labels in trainloader:
        batch_labels = []
        for label in labels:
            if label in machine:
                batch_labels.append(0)
            else:
                batch_labels.append(1)
        train_macro_labels.append(batch_labels)
    print('traning data finished.')
    train_macro_labels = np.array(train_macro_labels)

    print('operating test data...')
    for images, labels in testloader:
        batch_labels = []
        for label in labels:
            if label in machine:
                batch_labels.append(0)
            else:
                batch_labels.append(1)
        test_macro_labels.append(batch_labels)
    print('test data finished.')
    test_macro_labels = np.array(test_macro_labels)


    return trainloader, testloader, train_macro_labels, test_macro_labels


def get_101_OC_data():
    x = torch.load('./data/101_Object_Categories/data/data.pt')
    y = torch.load('./data/101_Object_Categories/data/macro_label.pt')

    print(x.size(), y.size())

    deal_dataset = TensorDataset(x,y)

    length=len(deal_dataset)
    print(length)
    train_size,test_size=int(0.8*length),int(0.2*length)
    if train_size + test_size != length:
        train_size += length - (train_size + test_size)
    #first param is data set to be saperated, the second is list stating how many sets we want it to be.
    train_set,test_set=torch.utils.data.random_split(deal_dataset,[train_size,test_size])
    #print(train_set,validate_set)
    print(len(train_set),len(test_set))

    train_loader = DataLoader(dataset=train_set, batch_size=HP.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_set, batch_size=HP.batch_size, shuffle=True, num_workers=2)

    return train_loader, test_loader

def get_101_data_split_by_macro_label(is_enumerate = False):
    root_dir = './data/101_data_split_by_macro_label'

    label_list = []

    label_to_idx = {}

    train_data_loader_dict = {}

    #train_data_dataloader_iter_dict = {}

    test_data_tensors_list = []
    test_data_labels_list = []

    idx = 0
    for item in os.listdir(root_dir):
        # to load the data.
        x = torch.load(root_dir + '/' + item)

        # to split the data to train and test set.
        x_train, x_test = torch.split(x, int(0.8*x.size()[0]), dim=0)

        # to create the label, label_to_idx, and label_list.
        label = item.replace('.pt','')
        label_to_idx[label] = idx
        label_list.append(label)
        
        # to record the test data tensors in list.
        test_data_tensors_list.append(x_test)

        # to record the test data labels in list.
        for i in range(x_test.size()[0]):
            test_data_labels_list.append(label_to_idx[label])
        
        # to record the train data tensors in dict by using label as keys.
        cur_train_label_list = []
        for i in range(x_train.size()[0]):
            cur_train_label_list.append(label_to_idx[label])
        train_data_dataset = TensorDataset(x_train, torch.tensor(cur_train_label_list))
        if is_enumerate:
            train_data_dataloader = DataLoader(dataset = train_data_dataset, batch_size = 1, shuffle = True, num_workers = 2)
        else:
            train_data_dataloader = DataLoader(dataset = train_data_dataset, batch_size = HP.batch_size, shuffle = True, num_workers = 2)

        train_data_loader_dict[label] = train_data_dataloader
        #train_data_dataloader_iter_dict[label] = iter(train_data_dataloader)

        idx += 1

    # to make the test data and label to tensor
    test_data_tensors = torch.cat(test_data_tensors_list, dim=0)
    test_data_labels = torch.tensor(test_data_labels_list)

    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_data_labels), batch_size = HP.batch_size, shuffle = True, num_workers = 2)

    #for item in label_to_idx.keys():
    #    print(item)


    return label_to_idx, train_data_loader_dict, test_data_loader


def get_CIFAR100_full_training_tensor():
    training_set = torch.load('./data/CIFAR100/training.pt')
    Xs = []
    Ys = []
    for img, label in training_set:
        x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)

    index = torch.LongTensor(random.sample(range(HP.train_set_size), HP.sample_num))
    training_data_tensors = torch.index_select(training_data_tensors, dim=0, index=index)
    training_label_tensors = torch.index_select(training_label_tensors, dim=0, index=index)

    print('get random tensor for drawing',training_data_tensors.size(),training_label_tensors.size())

    return training_data_tensors, training_label_tensors


def get_CIFAR100_data_loader():
    training_set = torch.load('./data/CIFAR100/training.pt')
    print('length of training set: ', len(training_set))
    Xs = []
    Ys = []
    for img, label in training_set:
        x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)

    test_set = torch.load('./data/CIFAR100/test.pt')
    print('length of test set: ', len(test_set))
    Xs = []
    Ys = []
    for img, label in test_set:
        x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    test_data_tensors = torch.stack(Xs)
    test_label_tensors = torch.tensor(Ys)
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)


    print(training_data_tensors.size(), training_label_tensors.size())
    print(test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader

def get_FashionMNIST_data_loader():
    training_set = torch.load('./data/FashionMNIST/training.pt')
    print('length of training set: ', len(training_set))
    Xs = []
    Ys = []
    for img, label in training_set:
        img = img.convert("RGB")
        x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)

    test_set = torch.load('./data/FashionMNIST/test.pt')
    print('length of test set: ', len(test_set))
    Xs = []
    Ys = []
    for img, label in test_set:
        img = img.convert("RGB")
        x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    test_data_tensors = torch.stack(Xs)
    test_label_tensors = torch.tensor(Ys)
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)


    print('training tensors: ', training_data_tensors.size(), training_label_tensors.size())
    print('test tensors: ', test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader

def get_VOC_data_loader():
    training_set = torch.load('./data/VOC/test.pt')
    print('length of training set: ', len(training_set))
    Xs = []
    Ys = []
    for img, label in training_set:
        img = img.resize((300, 300),Image.ANTIALIAS)
        x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)

    test_set = torch.load('./data/VOC/training.pt')
    print('length of test set: ', len(test_set))
    Xs = []
    Ys = []
    for img, label in test_set:
        img = img.resize((300, 300),Image.ANTIALIAS)
        x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    test_data_tensors = torch.stack(Xs)
    test_label_tensors = torch.tensor(Ys)
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)


    print('training tensors: ', training_data_tensors.size(), training_label_tensors.size())
    print('test tensors: ', test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader

def get_MNIST_arc_data_loader():
    training_set = torch.load('./data/MNIST_arc/training.pt')
    img, label = training_set

    training_data_tensors = single_channel_to_3_channel(img) # 60000
    training_label_tensors = label

    training_data_tensors = training_data_tensors.float()
    
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)


    test_set = torch.load('./data/MNIST_arc/test.pt')
    img, label = test_set

    test_data_tensors = single_channel_to_3_channel(img) # 10000
    test_label_tensors = label

    test_data_tensors = test_data_tensors.float()
    
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)


    print('training tensors: ', training_data_tensors.size(), training_label_tensors.size())
    print('test tensors: ', test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader


def get_MNIST_orientation_data_loader():
    training_set = torch.load('./data/MNIST_orientation/training.pt')
    img, label = training_set

    training_data_tensors = single_channel_to_3_channel(img) # 60000
    training_label_tensors = label

    training_data_tensors = training_data_tensors.float()
    
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)


    test_set = torch.load('./data/MNIST_orientation/test.pt')
    img, label = test_set

    test_data_tensors = single_channel_to_3_channel(img) # 10000
    test_label_tensors = label

    test_data_tensors = test_data_tensors.float()
    
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)


    print('training tensors: ', training_data_tensors.size(), training_label_tensors.size())
    print('test tensors: ', test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader


def getData(dataset_name):
    if dataset_name == 'CIFAR10':
        return get_CIFAR10_dataloader()
    
    elif dataset_name == '101':
        return get_101_OC_data()

    elif dataset_name == 'CIFAR100':
        return get_CIFAR100_data_loader()

    elif dataset_name == 'FashionMNIST':
        return get_FashionMNIST_data_loader()

    elif dataset_name == 'VOC':
        return get_VOC_data_loader()

    elif dataset_name == 'MNIST_arc':
        return get_MNIST_arc_data_loader()

    elif dataset_name == 'MNIST_orientation':
        return get_MNIST_orientation_data_loader()

    else:
        raise ValueError("No Such Dataset")

def get_full_data(dataset_name):
    if dataset_name == 'CIFAR100':
        return get_CIFAR100_full_training_tensor()

    else:
        raise ValueError("No Such Dataset")