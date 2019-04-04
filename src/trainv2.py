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
import argparse
import progressbar

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
#  Args  #
###########	
parser = argparse.ArgumentParser()
parser.add_argument("--architecture")
parser.add_argument("--epochs")
parser.add_argument("--mag")
parser.add_argument("--dataset")
parser.add_argument("--batch_size")
parser.add_argument("--lr")
parser.add_argument("--classification")
parser.add_argument("--preprocessing")
parser.add_argument("--kfold")
args = parser.parse_args()

###########
#  Paths  #
###########	

# Path to directory
PATH = "../"

# CSVs
dataset_class = args.classification
mag = int(args.mag)

train_csv = PATH + "csvs/train_{}_{}_kfold_{}.csv".format(mag, dataset_class, args.kfold)
test_csv = PATH + "csvs/test_{}_{}_kfold_{}.csv".format(mag, dataset_class, args.kfold)

# Results and directory
directory = get_time()
results_path = PATH + "results/" + directory + "/"

# IMG path
dataset = args.dataset
img_path = PATH + "datasets/" + dataset

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

num_classes = get_classes(dataset_class)
batch_size = int(args.batch_size)
lr = float(args.lr)
n_epochs = int(args.epochs)
weight_file = None
architecture = args.architecture
weighted = torch.FloatTensor([0.68, 0.32]).cuda() if (num_classes == 2) else torch.FloatTensor([.2, .09, .21, .13, .02, .13, .09, .13]).cuda()


############################
#  Model, Dataset & Loader #
############################

(model, img_size) = get_model(num_classes, weight_file, architecture)
model = model.cuda()

preprocessing = args.preprocessing
train_dataset = BreakHis(img_path, train_csv, img_size, training=True, preprocessing=preprocessing)
test_dataset = BreakHis(img_path, test_csv, img_size, training=False, preprocessing=preprocessing)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#########################################
#  Optimizer, criterion and performance #
#########################################

optimizer = Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(weight=weighted)
performance = Performance(performance_path, num_classes)

# Log the hyperparameters
with open(results_path + "readme.txt","w") as file:
	file.write("Architecture: {}\n".format(architecture))
	file.write("Train and test files: {} and {}\n".format(train_csv, test_csv))
	file.write("Dataset: {} \n".format(dataset))
	file.write("Preprocessing: {} \n".format(preprocessing))
	file.write("Image size: {}\n".format(img_size))
	file.write("Classes: {}\n".format(num_classes))
	file.write("Train and test batches size: {}\n".format(batch_size))
	file.write("Epochs: {}\n".format(n_epochs))

#######################
#  Training & Testing #
#######################

best_model = 0
epoch_bar = progressbar.ProgressBar(maxval=n_epochs, \
    widgets=[progressbar.Bar('=', '[', ']'), '', progressbar.Percentage()])

epoch_bar.start()

for epoch in range(n_epochs):
	epoch_bar.update(epoch)
	#print("Training epoch {}".format(epoch))
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
	
	#print("Testing epoch {}".format(epoch))
	for batch_id, (data, target, patient_id) in enumerate(test_loader):

		data, target = data.cuda(), target.cuda()		
		output = model(data)	
		_, predict = torch.max(output, dim=1)
		
		performance.add_results(target.cpu().tolist(), predict.cpu().tolist(), patient_id.tolist(), output.cpu().tolist())
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

epoch_bar.finish()