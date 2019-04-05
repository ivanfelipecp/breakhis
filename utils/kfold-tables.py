import os
import json
import sys
import numpy as np
from tables_vars import *

architectures = ["squeezenet","traditional"]
magnifications = ["40"]
preprocs = ["rgb","um","he","clahe","dnlm1"]

kfold_dir = "../results/kfold"
results_dir = "/performance/"
epochs = 100

key_ILA = "image_level_accuracy"

def get_json(filepath):
	with open(filepath, "r") as f:
		datastore = json.load(f)
		return datastore

def search_best(directory):
    best_acc = 0
    best_json = {}
    best_i = 0
    for i in range(epochs):
        file_i = "results_{}.json".format(i)
        results_i = get_json(directory+file_i)
        if results_i[key_ILA] > best_acc:
            best_acc = results_i[key_ILA]
            best_json = results_i
            best_i = i
    print("The best result in {} is {} \n".format(directory, best_i))
    return best_json

def get_best_results(m,a,p):
    current_dir = "{}/{}/{}/{}/".format(kfold_dir, m, a, p)
    dirs = os.listdir(current_dir)
    best_results = {}
    for i in range(len(dirs)):
        print("Looking for best results in {}".format(current_dir))
        best_result = search_best(current_dir+dirs[i]+results_dir)
        best_results[i] = best_result
    return best_results

for m in magnifications:
    for a in architectures:
        for p in preprocs:
            best_results = get_best_results(m,a,p)