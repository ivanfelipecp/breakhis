import InceptionV3
import DenseNet161
import TraditionalCNN
import SqueezeNet
import FractalNet
import TraditionalV2

def get_model(num_classes, weights=None, m=1):
	model = None
	default_size = 229
	img_size = default_size

	if m == "densenet161":
		model = DenseNet161.get_model
	elif m == "inceptionv3":
		model = InceptionV3.get_model
		img_size = 299
	elif m == "traditionalcnn":
		model = TraditionalCNN.get_model
	elif m == "squeezenet":
		model = SqueezeNet.get_model
		img_size = 256
	elif m == "fractalnet":
		model = FractalNet.get_model
	elif m == "traditionalv2"		:
		model = TraditionalV2.get_model

	return (model(num_classes, weights), img_size)