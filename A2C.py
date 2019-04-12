import torch
import numpy as np 
import gym 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from statistics import mean,stdev
from torch.distributions import Categorical
from cnn_model import cnn
import threading 
import wrapper 

GLOBAL = cnn()
optimizer = optim.Adam(GLOBAL.parameters(), lr=3e-4)
environment = gym.make('PongNoFrameskip-v4')
environment = wrapper.wrap_dqn(environment)
counter = 0
class worker():

	def __init__(self):
		self.env = wrapper.wrap_dqn(gym.make('PongNoFrameskip-v4'))
		self.gamma = 0.99
		self.model_return = []
		

	def choose_action(self,frame):
		global counter
		counter += 1
		policy, value = GLOBAL.forward(frame)
		policy_prob = F.softmax(policy,dim = -1)
		# print(policy_prob)
		categorical = Categorical(policy_prob)
		action = categorical.sample()
		self.model_return.append((-categorical.log_prob(action),value))

		return action

	def train(self,episode):
		reward = []
		policy_loss = []
		value_loss = []
		R = 0

		for r in reversed(episode):
			R = r + self.gamma*R
			reward.append(R)

		reward = reward[::-1]

		for idx,Gt in enumerate(reward):
			log_prob,value = self.model_return[idx]
			log_prob = log_prob.squeeze(0)
			value = value.squeeze(0)
			advantage = Gt - value.item()

			policy_loss.append(log_prob*advantage)
			value_loss.append((torch.tensor([r])-value)**2)

		del self.model_return[:] 

		loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
		
		loss.backward()


	def main(self):
		frame = self.env.reset().__array__()
		episode = []
		while(1):
			action = self.choose_action(frame).item()
			f, reward, is_done, _ = self.env.step(action)
			frame = f.__array__()
			episode.append(reward)
			if is_done:
				break
		self.train(episode)

def choose_action(frame):
	policy, value = GLOBAL.forward(frame)
	policy_prob = F.softmax(policy,dim = -1)
	categorical = Categorical(policy_prob)
	action = categorical.sample()
	return action

def test():
	R  = 0
	cnt = 0
	while(1):
		cnt += 1
		frame = environment.reset().__array__()
		while(1):
			action = choose_action(frame).item()
			f, reward, is_done, _ = environment.step(action)
			frame = f.__array__()

			R += reward
			if is_done:
				break
		if cnt%5 == 0:
			print(R)
			break

no_of_workers = 8
workers = []
threads = []

for i in range(no_of_workers):
	workers.append(worker())
def make_thread():
	for i in range(no_of_workers):
		threads.append(threading.Thread(target= workers[i].main()))

cnt = 0
while(1):
	cnt += 1
	del threads[:]
	optimizer.zero_grad()
	make_thread()
	for i in threads:
		i.start()
	for i in threads:
		i.join()
	optimizer.step()
	if cnt%5 == 0:
		test()







