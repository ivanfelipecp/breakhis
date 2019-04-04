import os
import re
import numpy as np
import csv
from random import shufle

# Globals
TUMOUR_INDEX = 2
MAG_INDEX = 5
EXT = ".png"

# Aux list
mags_list = ["40","100","200","400"]
benign_list = ["A","F","PT","TA"]
malign_list = ["DC","LC","MC","PC"]
tumours_list = benign_list + malign_list

# Create dicts
mags = dict()
for i in mags_list:
    tumours = dict()
    mags[i] = tumours
    for j in tumours_list:
        tumours[j] = list()

# create tumours classes
tumour_class = dict()
binary_class = dict()
for i in range(len(tumours_list)):
    t = tumours_list[i]
    tumour_class[t] = i
    binary_class[t] = 0 if t in benign_list else 1

def split_name(file_name):
    file_name = file_name.replace(".","-.")
    return re.split("_|-",file_name)

def get_tumour(file_split):
    return file_split[TUMOUR_INDEX]

def get_mag(file_split):
    return file_split[MAG_INDEX]

def roll(array, test_perc):
    return list(np.roll(array, test_perc))



# Get all images
images_path = "../datasets/breakhis/"
images = os.listdir(images_path)

# Add images to current magnifition
for i in images:
    splited = split_name(i)
    m = get_mag(splited)
    t = get_tumour(splited)
    mags[m][t].append(i)

# Iterate over mags
k_fold = 5
test_percent = 0.2
for m in mags_list:
    A = mags[m]["A"]
    F = mags[m]["F"]
    PT = mags[m]["PT"]
    TA = mags[m]["TA"]
    DC = mags[m]["DC"]
    LC = mags[m]["LC"]
    MC = mags[m]["MC"]
    PC = mags[m]["PC"]
    temp_tum = [A,F,PT,TA,DC,LC,MC,PC]
    for k in range(k_fold):
        train = list()
        test = list()
        for i in range(len(temp_tum)):
            tum = temp_tum[i]
            test_perc = int(len(tum) * test_percent)
            test += tum[:test_perc]
            train += tum[test_perc:]            
            temp_tum[i] = roll(tum, test_perc)
        # end del fold k
        # create multi and binary csv
        train_multi = open("train_{}_multi_kfold_{}.csv".format(m, k), "w")
        train_bin = open("train_{}_binary_kfold_{}.csv".format(m, k), "w")
        test_multi = open("test_{}_multi_kfold_{}.csv".format(m, k), "w")
        test_bin = open("test_{}_binary_kfold_{}.csv".format(m, k), "w")

        # Create csv files
        fields = ["filename","class"]
        train_multi_writter = csv.DictWriter(train_multi, fieldnames=fields)
        train_bin_writter = csv.DictWriter(train_bin, fieldnames=fields)
        test_multi_writter = csv.DictWriter(test_multi, fieldnames=fields)
        test_bin_writter = csv.DictWriter(test_bin, fieldnames=fields)

        # Write headers
        train_multi_writter.writeheader()
        train_bin_writter.writeheader()
        test_multi_writter.writeheader()
        test_bin_writter.writeheader()

        # Write on train files
        for image in train:
            splited = split_name(image)
            t = get_tumour(splited)
            multi_c = tumour_class[t]
            binary_c = binary_class[t]

            train_multi_writter.writerow({"filename": image, "class":multi_c})
            train_bin_writter.writerow({"filename": image, "class":binary_c})

        # Write on test files
        for image in test:
            splited = split_name(image)
            t = get_tumour(splited)
            multi_c = tumour_class[t]
            binary_c = binary_class[t]

            test_multi_writter.writerow({"filename": image, "class":multi_c})
            test_bin_writter.writerow({"filename": image, "class":binary_c})

        # Close files
        train_multi.close()
        train_bin.close()
        test_multi.close()
        test_bin.close()