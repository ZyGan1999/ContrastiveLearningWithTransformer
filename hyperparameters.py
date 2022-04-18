from tkinter.tix import Tree


batch_size = 64 # batch size in the training
learning_rate = 0.001 # learning rate in the training
epoch_num = 200 # max epoch number in the training

dim_k = 2048
dim_v = 2048
n_heads = 8


data_set = 'mini-imagenet'

alpha = 0.1

attention = True
contrastive  = False


from utils import get_train_set_size
from utils import get_cls_num
train_set_size = get_train_set_size(data_set)
cls_num = get_cls_num(data_set)

sample_num = 300