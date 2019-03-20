import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

def get_model(num_classes, weights=None):
	pre = False if weights else True
	model = models.squeezenet1_1(pretrained = pre)

	final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
	model.classifier = nn.Sequential(
		nn.Dropout(p=0.5),
		final_conv,
		nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1)),
		nn.Softmax(dim=1),
		nn.AdaptiveAvgPool2d(output_size=(1, 1))
  	)

	if not pre:
		model.load_state_dict(torch.load(weights))
		print("weights loaded")

	return model