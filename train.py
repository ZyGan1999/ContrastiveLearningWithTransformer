import imp
from operator import mod
from os import remove
from re import X
from tkinter import Y
from model.CNN import CNN_NET
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torch
import torchvision
import numpy as np
from dataprocessor.datacomposer import get_sample_data
import hyperparameters as HP
from model.resnet50 import ResNet50
from tensorboardX import SummaryWriter
from model.TC2 import TransformerContrastive
import random

from utils import get_iter_dict
from utils import calc_accuracy
from utils import writer
from utils import freeze_by_names
from utils import draw
from utils import tag_name

from model.contrastive import ContrastiveLoss

#torch.cuda.set_device(3)

#label_to_idx, train_data_loader_dict, test_loader = get_101_data_split_by_macro_label()
#idx_to_label = dict(zip(label_to_idx.values(), label_to_idx.keys()))


#print(label_to_idx)
#print(idx_to_label)


#tag_name = '[dataset]='+HP.data_set+' - '+'[batch_size]='+str(HP.batch_size)+' - '+'[dim_k]='+str(HP.dim_k)+' - '+'[dim_v]='+str(HP.dim_v)+' - '+'[n_heads]='+str(HP.n_heads)+' - '+'[lr]='+str(HP.learning_rate)

# writer = SummaryWriter(comment = '[resnet_with_attention_in_one_batch]'+tag_name)
# writer = SummaryWriter(comment = '[resnet+attention]'+tag_name)
# writer = SummaryWriter(comment = '[resnet+attention+SAME LABEL IN ONE BATCH] '+tag_name)






def train_by_gathering_same_label_data_in_one_batch(_net, label_to_idx_dict, train_data_loader_dict, test_loader):

    # to train with the same label data gathered in one batch
    '''
    net (nn.Module) the model for training
    label_to_idx_dict (Dict) {label(str):idx(int)}
    train_data_loader_dict (Dict) {label(str):dataloader} batchsize = HP.batch_size
    test_loader (Dataloader) the dataloader of test data
    '''
    net = _net

    optimizer = optim.SGD(net.parameters(), lr = HP.learning_rate, momentum = 0.9)
    loss_func = torch.nn.CrossEntropyLoss()

    
    

    EPOCH = HP.epoch_num

    batch_num = HP.train_set_size / HP.batch_size

    

    for epoch in range(EPOCH):
        TL = test_loader
        #if epoch == 50:
        #    freeze_by_names(net, 'slf_embed')
        
        
        
        # to randomly select a batch from the 17 macro categories
        # then feed the batch to model
        # until all the train data is fed
        epoch_loss = 0

        label_list = list(label_to_idx_dict.keys())

        train_data_dataloader_iter_dict = get_iter_dict(train_data_loader_dict)

        while len(label_list) > 0:
        
            cur_label = random.choice(label_list)

            cur_loader_iter = train_data_dataloader_iter_dict[cur_label]

            try:
                b_x, b_y = next(cur_loader_iter)
                # to train
                b_x = b_x.cuda()
                b_y = b_y.cuda()
                outputs = net(b_x) # 喂给 net 训练数据 x, 输出预测值
                #print(b_x.size(), b_y, outputs)

                loss = loss_func(outputs, b_y) # 计算两者的误差
                optimizer.zero_grad() # 清空上一步的残余更新参数值
                loss.backward() # 误差反向传播, 计算参数更新值
                optimizer.step() # 将参数更新值施加到 net 的 parameters 上
                # 打印状态信息
                epoch_loss += loss.item()

            except StopIteration:
                label_list.remove(cur_label)
                #print(cur_label + ' removed')

        epoch_loss /= batch_num
        writer.add_scalar('loss', epoch_loss, global_step = epoch)

        accuracy = calc_accuracy(net, TL)
        writer.add_scalar('accuracy', accuracy, global_step = epoch)

        print('[epoch %d] loss = %.3f   accuracy = %.9f' % (epoch,epoch_loss,accuracy))
        epoch_loss = 0
    print('Finish Training.')



def train_by_allocate_different_label_in_one_batch(net, label_to_idx_dict, train_data_loader_dict, test_loader):
    # to train with one batch filled with different label data
    '''
    net (nn.Module) the model for training
    label_to_idx_dict (Dict) {label(str):idx(int)}
    train_data_loader_dict (Dict) {label(str):dataloader} batchsize = 1
    test_loader (Dataloader) the dataloader of test data
    '''
    optimizer = optim.SGD(net.parameters(), lr = HP.learning_rate, momentum = 0.9)
    loss_func = torch.nn.CrossEntropyLoss()

    EPOCH = HP.epoch_num

    batch_num = HP.train_set_size / HP.batch_size

    for epoch in range(EPOCH):
        epoch_loss = 0

        label_list = list(label_to_idx_dict.keys())

        train_data_dataloader_iter_dict = get_iter_dict(train_data_loader_dict)

        
        
        while len(label_list) > 0:
            # to generate one batch
            Xs = []
            Ys = []
            batch_count = 0 
            idx = 0
            
            while batch_count != HP.batch_size and len(label_list) != 0:
                cur_label = label_list[idx % len(label_list)]
                # select one instance
                try:
                    x, y = next(train_data_dataloader_iter_dict[cur_label])
                    #print(x.size(), y.size())
                    # add it to the list which will be used to generate a batch
                    Xs.append(x)
                    Ys.append(y)
                    batch_count += 1
                    idx += 1
                    
                except StopIteration:
                    label_list.remove(cur_label)

            # cat the tensors together
            # this is the data for one batch
            random.shuffle(Xs)
            random.shuffle(Ys)
            b_x = torch.cat(Xs,0)
            b_y = torch.cat(Ys,0)
            
            # to train
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            outputs = net(b_x) # 喂给 net 训练数据 x, 输出预测值

            loss = loss_func(outputs, b_y) # 计算两者的误差
            optimizer.zero_grad() # 清空上一步的残余更新参数值
            loss.backward() # 误差反向传播, 计算参数更新值
            optimizer.step() # 将参数更新值施加到 net 的 parameters 上
            # 打印状态信息
            epoch_loss += loss.item()

            # end of one batch training
        
        # end of one epoch training
        epoch_loss /= batch_num
        writer.add_scalar('loss', epoch_loss, global_step = epoch)

        accuracy = calc_accuracy(net, test_loader)
        writer.add_scalar('accuracy', accuracy, global_step = epoch)

        print('[epoch %d] loss = %.3f   accuracy = %.3f' % (epoch,epoch_loss,accuracy))
        epoch_loss = 0
    # end of whole training
    print('Finish Training.')




def train_con(net, trainloader, testloader):
    
    # to train with contrastive loss added
    '''
    net (nn.Module) the model for training
    trainloader (Dataloader) the dataloader for training data
    testloader (Dataloader) the dataloader for test data
    '''
    
    EPOCH = HP.epoch_num
    net = net.cuda()
    optimizer = optim.SGD(net.parameters(), lr = HP.learning_rate, momentum = 0.9)
    #loss_func = torch.nn.CrossEntropyLoss()
    contra_loss_func = ContrastiveLoss()
    contra_loss_func = contra_loss_func.cuda()
    cls_loss_func = torch.nn.CrossEntropyLoss()
    batch_num = HP.train_set_size / HP.batch_size
    for epoch in range(EPOCH):
        epoch_loss = 0.0
        con_loss = 0.0
        cls_loss = 0.0
        running_loss = 0.0
        for step, (b_x,b_y)in enumerate(trainloader):
            # print(b_x)
            #b_y = torch.tensor(train_macro_labels[step])
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            representation, cls_rtn = net(b_x) # 喂给 net 训练数据 x, 输出预测值
            #print(outputs,b_y)
            loss1 = contra_loss_func(representation, b_y)
            loss2 = cls_loss_func(cls_rtn, b_y)
            loss = HP.alpha * loss1 + (1-HP.alpha) * loss2 # 计算两者的误差
            optimizer.zero_grad() # 清空上一步的残余更新参数值
            loss.backward() # 误差反向传播, 计算参数更新值
            optimizer.step() # 将参数更新值施加到 net 的 parameters 上
            # 打印状态信息
            running_loss += loss.item()
            epoch_loss += loss.item()
            con_loss += loss1
            cls_loss += loss2
            if step % 500 == 499:    # 每500批次打印一次
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1, running_loss / 200))
                running_loss = 0.0
        epoch_loss /= batch_num
        writer.add_scalar('loss', epoch_loss, global_step = epoch)
        accuracy = calc_accuracy(net, testloader)
        writer.add_scalar('accuracy', accuracy, global_step = epoch)
        print('[epoch %d] total_loss = %.3f   contra_loss = %.3f   cls_loss = %.3f   accuracy = %.3f' % (epoch,epoch_loss,con_loss,cls_loss,accuracy))
        epoch_loss = 0
    print('Finished Training')


def train_raw(net, trainloader, testloader):
    
    # to train without any special operation on the data
    '''
    net (nn.Module) the model for training
    trainloader (Dataloader) the dataloader for training data
    testloader (Dataloader) the dataloader for test data
    '''
    
    EPOCH = HP.epoch_num
    net = net.cuda()
    optimizer = optim.SGD(net.parameters(), lr = HP.learning_rate, momentum = 0.9)
    best_model = net
    best_acc = 0.0
    #loss_func = torch.nn.CrossEntropyLoss()
    cls_loss_func = torch.nn.CrossEntropyLoss()
    batch_num = HP.train_set_size / HP.batch_size
    data_tensor,label_tensor = get_full_data(HP.data_set)
    print(data_tensor.size(),label_tensor.size())

    for epoch in range(EPOCH):
        if epoch == 0:
            with torch.no_grad():
                representation,_ = net(data_tensor.cuda())
                print(representation.size())
                draw(X=representation.cpu(),Y=label_tensor,msg='BeforeTraining')

        elif epoch % 20 == 0:
            with torch.no_grad():
                representation,_ = net(data_tensor.cuda())
                draw(X=representation.cpu(),Y=label_tensor,msg='Training,'+'epoch='+str(epoch))

        epoch_loss = 0.0
        running_loss = 0.0
        for step, (b_x,b_y)in enumerate(trainloader):
            # print(b_x)
            #b_y = torch.tensor(train_macro_labels[step])
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            representation, cls_rtn = net(b_x) # 喂给 net 训练数据 x, 输出预测值
            #print(outputs,b_y)
            loss = cls_loss_func(cls_rtn, b_y)
            optimizer.zero_grad() # 清空上一步的残余更新参数值
            loss.backward() # 误差反向传播, 计算参数更新值
            optimizer.step() # 将参数更新值施加到 net 的 parameters 上
            # 打印状态信息
            running_loss += loss.item()
            epoch_loss += loss.item()
            if step % 500 == 499:    # 每500批次打印一次
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1, running_loss / 200))
                running_loss = 0.0
        epoch_loss /= batch_num
        writer.add_scalar('loss', epoch_loss, global_step = epoch)
        accuracy = calc_accuracy(net, testloader)
        if accuracy > best_acc:
            best_acc = accuracy
            best_model = net
        writer.add_scalar('accuracy', accuracy, global_step = epoch)
        print('[epoch %d] loss = %.3f   accuracy = %.3f' % (epoch,epoch_loss,accuracy))
        epoch_loss = 0
    
    #representation, cls_rtn = best_model()
    with torch.no_grad():
        representation,_ = best_model(data_tensor.cuda())
        draw(X=representation.cpu(),Y=label_tensor,msg='BestTraining')

    print('Finished Training')


def train(net, trainloader, testloader, is_con):
    # to train
    '''
    net (nn.Module) the model for training
    trainloader (Dataloader) the dataloader for training data
    testloader (Dataloader) the dataloader for test data
    is_con: whether to add contrastive loss
    '''
    
    EPOCH = HP.epoch_num
    net = net.cuda()
    optimizer = optim.SGD(net.parameters(), lr = HP.learning_rate, momentum = 0.9)
    best_model = net
    best_acc = 0.0

    cls_loss_func = torch.nn.CrossEntropyLoss()
    contra_loss_func = ContrastiveLoss()
    contra_loss_func = contra_loss_func.cuda()

    batch_num = HP.train_set_size / HP.batch_size
    data_tensor,label_tensor = get_sample_data(HP.data_set)

    for epoch in range(EPOCH):

        with torch.no_grad():
            representation,_ = net(data_tensor.cuda())
            draw(X=representation.cpu(),Y=label_tensor,msg='Training,'+'epoch='+str(epoch))
        
        epoch_loss = 0.0
        running_con_loss = 0.0
        running_cls_loss = 0.0
        for step, (b_x,b_y)in enumerate(trainloader):
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            print(b_x.size(),b_y.size())
            representation, cls_rtn = net(b_x) # 喂给 net 训练数据 x, 输出预测值

            cls_loss = cls_loss_func(cls_rtn, b_y)
            contra_loss = contra_loss_func(representation, b_y)
            if is_con:
                loss = HP.alpha * contra_loss + (1-HP.alpha) * cls_loss
            else:
                loss = cls_loss
            optimizer.zero_grad() # 清空上一步的残余更新参数值
            loss.backward() # 误差反向传播, 计算参数更新值
            optimizer.step() # 将参数更新值施加到 net 的 parameters 上
            # 打印状态信息
            epoch_loss += loss.item()
            running_con_loss += contra_loss
            running_cls_loss += cls_loss

        epoch_loss /= batch_num
        running_con_loss /= batch_num
        running_cls_loss /= batch_num
        writer.add_scalar('loss', epoch_loss, global_step = epoch)
        accuracy = calc_accuracy(net, testloader)
        if accuracy > best_acc:
            best_acc = accuracy
            best_model = net
        writer.add_scalar('accuracy', accuracy, global_step = epoch)
        if is_con:
            print('[epoch %d] total_loss = %.3f   contra_loss = %.3f   cls_loss = %.3f   accuracy = %.3f' % (epoch,epoch_loss,running_con_loss,running_cls_loss,accuracy))
        else:
            print('[epoch %d] loss = %.3f   accuracy = %.3f' % (epoch,epoch_loss,accuracy))
        epoch_loss = 0
    
    with torch.no_grad():
        representation,_ = best_model(data_tensor.cuda())
        draw(X=representation.cpu(),Y=label_tensor,msg='Best-Training')

    torch.save({'model': best_model.state_dict()}, tag_name+'.pth')

    print('Finished Training')