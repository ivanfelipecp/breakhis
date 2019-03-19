import InceptionV3
import DenseNet161

def get_model(num_classes, weights=None, m=1):
	model = None
	img_size = None
	if m == "densenet161":
		model = DenseNet.get_model
		img_size = 229
	elif m == "inceptionv3":
		model = InceptionV3.get_model
		img_size = 299

	return (model(num_classes, weights), img_size)