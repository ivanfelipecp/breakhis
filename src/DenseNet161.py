import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes, weights=None):
	pre = False if weights else True
	model = models.densenet161(pretrained = pre)

	#model.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
	model.classifier = nn.Sequential(
		nn.Linear(in_features=2208, out_features=num_classes, bias=True),
		nn.Softmax(dim=1)
	)

	if not pre:
		model.load_state_dict(torch.load(weights))
		print("weights loaded")

	return model