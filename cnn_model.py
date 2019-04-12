import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

class cnn(nn.Module):
	def __init__(self):
		super(cnn,self).__init__()
		self.conv1 = nn.Conv2d(4,16,kernel_size = 8, stride = 4) #output size = 20*20
		self.conv2 = nn.Conv2d(16,32,kernel_size = 4, stride = 2) #output size = 9*9
		self.policy1 = nn.Linear(32*9*9,256)
		self.policy2 = nn.Linear(256,6)
		self.value1 = nn.Linear(32*9*9,256)
		self.value2 = nn.Linear(256,1)

	def forward(self,X):
		X = torch.from_numpy(X).float()
		X = X/225.0
		X = X.unsqueeze(0)
		X = F.relu(self.conv1(X))
		X = F.relu(self.conv2(X))
		X = X.view(-1,32*9*9)
		policy = F.relu(self.policy1(X))
		policy = F.relu(self.policy2(policy))
		value = F.relu(self.value1(X))
		value = F.relu(self.value2(value))

		return policy,value

	

