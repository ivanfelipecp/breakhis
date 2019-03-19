import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Net(nn.Module):
	def __init__(self, n_classes, in_channels=3):
		super(Net, self).__init__()

		self.conv1_2 = nn.Sequential(
			*self.conv_relu(in_channels, 10, 3),
			*self.conv_relu(10, 20, 2)
		)

		self.conv3_4 = nn.Sequential(
			*self.conv_relu(20, 10, 2),
			*self.conv_relu(10, 20, 2)
		)

		self.convs_output = 20*55*55

		self.FC = nn.Linear(self.convs_output, n_classes)
		
	def conv_relu(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
		return [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding), nn.ReLU()]

	def forward(self, x):
		x = F.max_pool2d(self.conv1_2(x), 2)
		x = F.max_pool2d(self.conv3_4(x), 2)
		x = x.view(-1, self.convs_output)
		x = F.softmax(self.FC(x), dim=1)
		return x

def get_model(num_classes, weights=None):
	pre = False if weights else True
	model = Net(num_classes)

	if not pre:
		model.load_state_dict(torch.load(weights))
		print("weights loaded")

	return model

#m = Net(3, 10).cuda()
#summary(m, (3,229,229))