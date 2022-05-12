from asyncio import FastChildWatcher
import datetime
import time



batch_size = 64 # batch size in the training
learning_rate = 0.001 # learning rate in the training
epoch_num = 100 # max epoch number in the training

dim_k = 2048
dim_v = 2048
n_heads = 8


data_set = 'CIFAR100'
backbone = 'ResNet50'

alpha = 0.5
lmd = 5.0

attention = True
contrastive  = True

G = True

TARGET = False

from utils import get_train_set_size
from utils import get_cls_num
train_set_size = get_train_set_size(data_set)
cls_num = get_cls_num(data_set)

sample_num = 300

curr_time = datetime.datetime.now()


outname = f'[TARGET:{TARGET}]-[G:{G}]-[Attention:{attention}]-[Contrastive:{contrastive}]-[backbone:{backbone}]-[dataset:{data_set}]-[batch_size:{batch_size}]-[dim_k:{dim_k}]-[dim_v:{dim_v}]-[n_heads:{n_heads}]-[lr:{learning_rate}]-[alpha:{alpha}]-[lmd:{lmd}]-[time:{curr_time}]'

def get_outname():
    outname = f'[TARGET:{TARGET}]-[G:{G}]-[Attention:{attention}]-[Contrastive:{contrastive}]-[backbone:{backbone}]-[dataset:{data_set}]-[batch_size:{batch_size}]-[dim_k:{dim_k}]-[dim_v:{dim_v}]-[n_heads:{n_heads}]-[lr:{learning_rate}]-[alpha:{alpha}]-[lmd:{lmd}]-[time:{curr_time}]'
    return outname

#outname = f'[G:{G}]-[backbone:{backbone}]-[dataset:{data_set}]-[attention:{attention}]-[contrastive:{contrastive}]-time:{curr_time}'



