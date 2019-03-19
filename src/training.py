###########
# Imports #
###########

import sys,os,datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
import models

from torch.autograd import Variable
from torch.optim import Adam
from models import get_model
from BreakHisDataset import BreakHis
from Performance import Performance

##############
#  Functions #
##############

def get_time():
	time = str(datetime.datetime.now())
	time = time.split(" ")
	dot = time[1].index(".")
	time[1] = time[1].replace(":", "-")
	return time[0]+"_"+time[1][:dot]

def get_classes(dataset):
	return 2 if dataset == "binary" in dataset else 8

###########
#  Paths  #
###########	

# Path to directory
PATH = "../"

# CSVs
dataset = "binary"
mag = 40

train_csv = PATH + "csv/train_{}_{}.csv".format(mag, dataset)
test_csv = PATH + "csv/test_{}_{}.csv".format(mag, dataset)

# Results and directory
directory = get_time()
results_path = PATH + "results/" + directory + "/"

# IMG path
img_path = PATH + "dataset/breakhis/"

# Results file
weights_file = "weights.pt"

# performance path
performance_path = results_path + "performance/"
# Make path
os.mkdir(results_path)
os.mkdir(performance_path)

##############
#  Hyperpams #
##############

num_classes = get_classes(dataset)
batch_size = 32
preprocessing = "color"
lr = 0.001
n_epochs = 50
weight_file = None
architecture = "densenet"
weighted = torch.FloatTensor([0.68, 0.32]).cuda() if (num_classes == 2) else torch.FloatTensor([.2, .09, .21, .13, .02, .13, .09, .13]).cuda()

# Log the hyperparameters
with open(results_path + "readme.txt","w") as file:
	file.write("Model: {}\n".format(architecture))
	file.write("Train and test files: {} and {}\n".format(train_csv, test_csv))
	file.write("Image size: {}\n".format(img_size))
	file.write("Preproccesing: {}\n".format(preprocessing))
	file.write("Classes: {}\n".format(num_classes))
	file.write("Train and test batches size: {}\n".format(batch_size))
	file.write("Epochs: {}\n".format(n_epochs))

############################
#  Model, Dataset & Loader #
############################

(model, img_size) = get_model(num_classes, weight_file, architecture).cuda()
train_dataset = BreakHis(train_csv ,img_path, preprocessing, img_size, True)
test_dataset = BreakHis(test_csv, img_path, preprocessing, img_size, False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10, shuffle=True)

#########################################
#  Optimizer, criterion and performance #
#########################################


optimizer = Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(weight=weighted)
performance = Performance(performance_path)

#######################
#  Training & Testing #
#######################

best_model = 0

for epoch in range(n_epochs):
	print("Training epoch {}".format(epoch))
	###########
	#  TRAIN  #
	###########
	model.train()
	train_loss = 0
	for batch_id, (data, target) in enumerate(train_loader):
		data, target = data.cuda(), target.cuda()
		
		optimizer.zero_grad()
		
		output = model(data)
		
		loss = criterion(output, target)
				
		loss.backward()
		
		optimizer.step()
	
	###########
	#   TEST  #
	###########        
	model.eval()
	performance.clear()
	test_loss = 0
	
	print("Testing epoch {}".format(epoch))
	for batch_id, (data, target, patient_id) in enumerate(test_loader):

		data, target = data.cuda(), target.cuda()		
		output = model(data)	
		_, predict = torch.max(output, dim=1)
		
		performance.add_results(target.cpu().tolist(), predict.cpu().tolist(), patient_id.cpu().tolist(), output.cpu().tolist())
		test_loss += criterion(output, target).item()

	test_loss =  test_loss / len(test_loader)
	performance.loss = test_loss
	###########
	# Results #
	###########  

	# save the performance
	performance.end_epoch(epoch)

	# save the model
	if performance.accuracy > best_model:
		best_model = performance.accuracy
		torch.save(model.state_dict(), results_path+weights_file)