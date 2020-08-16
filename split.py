#use torch.utils.data.Dataset to build my dataset from train.csv and test.csv
import numpy as np
import pandas as pd
from sklearn.model_selection  import train_test_split
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable

def train_val_split(train = 'train.csv',train_flie='train_set.csv',val_file='val_set.csv'):
    #training set "train.csv" was downloaded from kaggle.com
    train_data = pd.read_csv(train)
    #training datas contains Feature and Label.
    #divide training datas into training set and validation set
    train_set, val_set = train_test_split(train_data, test_size=0.5)
    #wirte csv files
    train_set.to_csv(train_flie,index = False )
    val_set.to_csv(val_file,index = False )
    print('train_data.shape:',train_data.shape)
    print('train_set.shape:',train_set.shape)
    print('val_set.shape:',val_set.shape)

train_val_split('train.csv','train_set.csv','val_set.csv')