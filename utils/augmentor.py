import Augmentor
import os
import sys
import string
import random

# Create the samples
def data_augmentation(path, samples=1000):
	pipeline = Augmentor.Pipeline(path)
	pipeline.rotate90(probability=1)
	pipeline.rotate180(probability=1)
	pipeline.rotate270(probability=1)
	pipeline.flip_left_right(probability=1)
	pipeline.flip_top_bottom(probability=1)
	pipeline.crop_random(probability=1, percentage_area=0.8)
	pipeline.crop_centre(probability=1, percentage_area=0.8)
	pipeline.sample(samples)

def get_random_code(N=6):
	return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))
	#return ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))

def join_name(name):
	new_name = ""
	m = len(name)
	for i in range(m - 1):
		new_name += name[i] + "-"
	i += 1		
	new_name += name[i]		
	return new_name

def parse_name(file):
	cont = 0
	start = 0
	end = 0
	
	for i in range(len(file)):
		if file[i] == "_":
			cont += 1
			pass
		if cont == 2:
			start = i - 2
			pass
		elif cont == 5:
			end = i
			break
	
	new_name = file[start:end]
	new_name = new_name.split("-")
	new_name[2] = get_random_code()
	new_name = join_name(new_name)
	return new_name

# Rename the samples
# A_original_SOB_B_A-14-22549AB-40-001.png_1e778df5-56e6-4a98-9167-c483b6f904e6
def rename_data(path, new_path):
	images = os.listdir(path)
	for i in images:
		new_name = parse_name(i)
		os.rename(path+i, new_path+new_name)

mags = ["40", "100", "200", "400"]
tumours = ["A", "F", "PT", "TA", "DC", "LC", "MC", "PC"]
new_path = "./da_dataset/"

for i in mags:
	for j in tumours:
		path = "./{}/{}/".format(i,j)
		output_path = path + "output/"
		data_augmentation(path)
		rename_data(output_path, new_path)