from typing import Any, Union

import pandas as pd
import numpy as np
import torch

from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import copy
from sklearn.utils import shuffle
import pandas as pd
from torch.utils.data.dataset import random_split

CUDA_DEVICES = 1

class LeNet(nn.Module):
	def __init__(self):
		super(LeNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, (3,3), padding=(1,1)) #6*28*28
		self.conv2 = nn.Conv2d(6, 16, (3,3), padding=(1,1)) #16*14*14
		self.conv3 = nn.Conv2d(16, 26, (3,3), padding=(1,1)) #26*7*7
		self.fc1   = nn.Linear(26*3*3, 120)
		self.fc2   = nn.Linear(120, 84)
		self.fc3   = nn.Linear(84, 64)
		self.fc4 = nn.Linear(64, 10)
	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), (2,2)) #6*14*14
		x = F.max_pool2d(F.relu(self.conv2(x)), (2,2)) #16*7*7
		x = F.max_pool2d(F.relu(self.conv3(x)), (2,2)) #26*2*2
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)

		return x
	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

net = LeNet()
print (net)

use_gpu = torch.cuda.is_available()
if use_gpu:

	net = net.cuda()
	print ('USE GPU')
else:
	print ('USE CPU')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)



#print ("1. Loading data")
train = pd.read_csv("train_set.csv").values
valid = pd.read_csv("val_set.csv").values
train = shuffle(train)
#print(len(train))
test = pd.read_csv("test.csv").values

#print ("2. Converting data")
train_data = train[:, 1:].reshape(train.shape[0], 1, 28, 28)
train_data = train_data.astype(float)
train_data /= 255.0
train_data = torch.from_numpy(train_data);
train_label = train[:, 0];
train_label = train_label.astype(int);
train_label = torch.from_numpy(train_label);
train_label = train_label.view(train.shape[0], -1);
#print(train_label.size(), train_label.size())

valid_data = valid[:, 1:].reshape(valid.shape[0], 1, 28, 28)
valid_data = valid_data.astype(float)
valid_data /= 255.0
valid_data = torch.from_numpy(valid_data);
valid_label = valid[:, 0];
valid_label = valid_label.astype(int);
valid_label = torch.from_numpy(valid_label);
valid_label = valid_label.view(valid.shape[0], -1);
#print(valid_label.size(), valid_label.size())

#print ("3. Training phase")
nb_train = train.shape[0]
nb_epoch = 100000
nb_index = 0
nb_batch = 8

for epoch in range(nb_epoch):
	if nb_index + nb_batch >= nb_train:
		nb_index = 0
	else:
		nb_index = nb_index + nb_batch

	mini_data  = Variable(train_data[nb_index:(nb_index+nb_batch)].clone())
	mini_label = Variable(train_label[nb_index:(nb_index+nb_batch)].clone(), requires_grad = False)
	mini_data  = mini_data.type(torch.FloatTensor)
	mini_label = mini_label.type(torch.LongTensor)
	if use_gpu:
		mini_data  = mini_data.cuda()
		mini_label = mini_label.cuda()
	optimizer.zero_grad()
	mini_out   = net(mini_data)
	mini_label = mini_label.view(nb_batch)
	mini_loss  = criterion(mini_out, mini_label)
	mini_loss.backward()
	optimizer.step()

	if (epoch + 1) % 5000 == 0:
		print("Epoch = %d, Loss = %f" %(epoch+1, mini_loss.data))

#print ("4. Validating phase")
total = 0
total_correct = 0
#Y_data = test.reshape(test.shape[0], 1, 28, 28)
#Y_data = Y_data.astype(float)
#Y_data /= 255.0
#Y_data = torch.from_numpy(Y_data);
#print(Y_data.size())
#nb_test = test.shape[0]
nb_val = valid.shape[0]
net.eval()

valid_prediction = np.ndarray(shape=(nb_val, 2), dtype=int)
for each_sample in range(nb_val):
	sample_data = Variable(valid_data[each_sample:each_sample+1].clone())
	sample_data = sample_data.type(torch.FloatTensor)

	#valid_data = valid_data.type(torch.FloatTensor)
	#valid_label = valid_label.type(torch.LongTensor)
	if use_gpu:
		sample_data = sample_data.cuda()
	sample_out = net(sample_data)
	_, pred = torch.max(sample_out, 1)
	valid_prediction[each_sample][0] = 1 + each_sample
	valid_prediction[each_sample][1] = pred.data
	#final_prediction[each_sample][0] = 1 + each_sample
	#final_prediction[each_sample][1] = pred.data

#for i in range(nb_val):
	#total_correct += (final_prediction[i][1] == valid_label[i][0])
total_correct += (valid_prediction == valid_label).sum().item()
valid_acc = total_correct / nb_val
#print(total_correct)
#print(nb_val)
print('Accuracy on the validation images: %d %%' % (100 * valid_acc))

#print("4. Testing phase")

Y_data = test.reshape(test.shape[0], 1, 28, 28)
Y_data = Y_data.astype(float)
Y_data /= 255.0
Y_data = torch.from_numpy(Y_data)
#print(Y_data.size())
nb_test = test.shape[0]

net.eval()

final_prediction = np.ndarray(shape=(nb_test, 2), dtype=int)
for each_sample in range(nb_test):
	sample_data = Variable(Y_data[each_sample:each_sample+1].clone())
	sample_data = sample_data.type(torch.FloatTensor)
	if use_gpu:
		sample_data = sample_data.cuda()
	sample_out = net(sample_data)
	_, pred = torch.max(sample_out, 1)
	final_prediction[each_sample][0] = 1 + each_sample
	final_prediction[each_sample][1] = pred.data
#	if (each_sample + 1) % 2000 == 0:
#		print("Total tested = %d" %(each_sample + 1))

#print('5. Generating submission file')

submission = pd.DataFrame(final_prediction, dtype=int, columns=['ImageId', 'Label'])
submission.to_csv('pytorch_LeNet.csv', index=False, header=True)

# end