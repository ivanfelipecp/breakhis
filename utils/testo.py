###############################
#           IMPORTS           #
###############################

print(__file__)

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
import sys, os

from torch.autograd import Variable
from models import get_model
from BreakHisDataset import BreakHis


###############################
#       GLOBAL VARIABLES      #
###############################

USE_CUDA = True
NUM_CLASSES = 2
capsule_net = get_model("weights.pt", 1)
if USE_CUDA:
    capsule_net = capsule_net.cuda()

preproccesing = "none"
test_csv = "../csv/test_40_binary.csv"
img_path = "../datasets/breakhis/"
test_dataset = BreakHis(test_csv, img_path, preproccesing) #BreakHis()
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)

accuracy = 0

for batch_id, (data, target) in enumerate(test_loader):
	target = torch.eye(NUM_CLASSES).index_select(dim=0, index=target)
	data, target = Variable(data), Variable(target)
	
	if USE_CUDA:
		data, target = data.cuda(), target.cuda()

	output, reconstructions, masked = capsule_net(data)
	loss = capsule_net.loss(data, output, target, reconstructions)
	
	_, target = torch.max(target, dim=1)
	_, masked = torch.max(masked, dim=1)
	
	#print(target)
	#print(masked)
	#print(torch.eq(target, masked))
	#print(torch.sum(torch.eq(target, masked)))
	#print(torch.sum(torch.eq(target, masked)).item())
	#sys.exit(0)
	accuracy += torch.sum(torch.eq(target,masked)).item()

print(accuracy/len(test_loader.dataset))