import pandas as pd
import numpy as np
import cv2
import re

from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset


class BreakHis(Dataset):
	def __init__(self, img_path, csv_path, image_size, training=True, preprocessing="rgb"):
		"""
		Args:
			csv_path (string): path to csv file
			img_path (string): path to the folder where images are
			preprocessing: can be 'clahe' , 'he', 'none' or 'color'
		"""
		# Folder where images are
		self.img_path = img_path

		# Self channels
		self.channels = cv2.IMREAD_COLOR # if (preprocessing == "color") else cv2.IMREAD_GRAYSCALE
		
		# Read the csv file, ignores header
		self.data_info = pd.read_csv(csv_path, header=0)
		
		# First column contains the image paths
		self.image_arr = np.asarray(self.data_info.iloc[:, 0])
		
		# Second column is the labels
		self.label_arr = np.asarray(self.data_info.iloc[:, 1])
		
		# Transform operator
		self.transform = transforms.Compose([
			transforms.Resize((image_size, image_size), Image.BICUBIC),
			transforms.ToTensor()
		])
		
		# Calculate len
		self.data_len = len(self.data_info.index)
		
		# Training/testing mode
		self.training = training

		self.preprocessing = self.get_preprocessing(preprocessing)

	def get_patient_id(self, file_name):
		split = re.split("_|-|\.",file_name)
		patient_id = int(split[-2])
		return patient_id		

	def __getitem__(self, index):
		# Get image name from the pandas df
		single_image_name = self.img_path + "/" + self.image_arr[index]
		
		# Open image in grayscale
		img_as_img = cv2.imread(single_image_name, self.channels)
		
		# apply preprocessing
		img_as_img = self.preprocessing(img_as_img)   

		# numpy 2 PIL
		img_as_img = Image.fromarray(img_as_img)
		
		# Transforms the image to tensor
		img_as_tensor = self.transform(img_as_img)

		# Get label
		single_image_label = self.label_arr[index]
		
		# Result		
		result = (img_as_tensor, single_image_label)
		
		if not self.training:
			patient_id = self.get_patient_id(self.image_arr[index])
			result = (img_as_tensor, single_image_label, patient_id)

		return result

	def __len__(self):
		return self.data_len
		
	def do_nothing(self, img):
		return img

	def HE(self, img):
		img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
		# equalize the histogram of the Y channel
		img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
		# convert the YUV image back to RGB format
		img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
		return img_output		

	def CLAHE(self, img):
		lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
		lab_planes = cv2.split(lab)
		clahe = cv2.createCLAHE(clipLimit=2.0)
		lab_planes[0] = clahe.apply(lab_planes[0])
		lab = cv2.merge(lab_planes)
		bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
		return bgr

	def get_preprocessing(self, preprocessing):
		preproc_method = self.do_nothing
		if preprocessing == "HE":
			preproc_method = self.HE
		elif preprocessing == "CLAHE"			:
			preproc_method = self.CLAHE
		return preproc_method