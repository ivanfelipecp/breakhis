import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes, weights=None):
	pre = False if weights else True
	model = models.inception_v3(pretrained = pre)

	#model.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
	model.fc = nn.Sequential(
		nn.Linear(in_features=2048, out_features=num_classes, bias=True),
		nn.Softmax(dim=1)
	)
	model.aux_logits = False
	if not pre:
		model.load_state_dict(torch.load(weights))
		print("weights loaded")

	return model