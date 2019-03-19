import re
import csv
from random import shuffle
import sys
import os

# globales

BINARY = 1
MULTI = 2
MAGNIFICATION = 5

MULTI = ["A", "F", "PT", "TA", "DC", "LC", "MC", "PC"]
BINARY = ["B", "M"]

M_CLASS = {
    "A": 0,
    "F": 1,
    "PT": 2,
    "TA": 3,
    "DC": 4,
    "LC": 5,
    "MC": 6,
    "PC": 7
}

MAGS = {0:"40", 1:"100", 2:"200", 3:"400"}

B_CLASS = {
    "B":0,
    "M":1
}

def parse_it(imgs):
    return [re.split("_|-",i) for i in imgs]

def get_magnitications(imgs):
    x40 = []
    x100 = []
    x200 = []
    x400 = []

    parsed = parse_it(imgs)
    for i in range(len(parsed)):
        if parsed[i][MAGNIFICATION] == MAGS[0]:
            x40.append(imgs[i])
        elif parsed[i][MAGNIFICATION] == MAGS[1]:
            x100.append(imgs[i])
        elif parsed[i][MAGNIFICATION] == MAGS[2]:
            x200.append(imgs[i])
        elif parsed[i][MAGNIFICATION] == MAGS[3]:
            x400.append(imgs[i])

    return [x40, x100, x200, x400]

def get_containers(binary):
    containers = [[], []]
    if not binary:
        containers = [[],[],[],[],[],[],[],[]]
    return containers       

def split_magnification(magnification, binary=True):
    containers = get_containers(binary)
    index = BINARY if binary else MULTI
    classes = B_CLASS if binary else M_CLASS
    parsed_imgs = parse_it(magnification)

    for i in range(len(parsed_imgs)):
        tumour = parsed_imgs[i][index]
        container_i = classes[tumour]
        containers[container_i].append(magnification[i])

    return containers

def writte_csv(writter, rows):
    for i in rows:
        writter.writerow(i)

# imgs can be binary o multiclass -> [[],[]]
def create_csv(imgs, magnification, binary=True):
    classes = B_CLASS if binary else M_CLASS
    mag_name = MAGS[magnification]

    train_p = 0.7

    file = "slide"
    classn = "tumour"

    fieldnames = [file, classn]
    train = []
    test = []

    for i in range(len(imgs)):
        # i es la clase ya
        current_class = imgs[i]
        shuffle(current_class)
        mid = int(len(current_class) * train_p)

        train_data = current_class[:mid]
        test_data = current_class[mid:]

        for j in train_data:
            train.append({file: j, classn: i})

        for j in test_data:
            test.append({file: j, classn: i})
    
    iden = "_binary.csv" if binary else "_multi.csv"
    train_csv = open("aug_train_"+MAGS[magnification]+iden, "w")
    test_csv = open("aug_test_"+MAGS[magnification]+iden, "w")

    writter_train = csv.DictWriter(train_csv, fieldnames=fieldnames)
    writter_test = csv.DictWriter(test_csv, fieldnames=fieldnames)

    writter_train.writeheader()
    writter_test.writeheader()

    writte_csv(writter_train, train)
    writte_csv(writter_test, test)

    train_csv.close()
    test_csv.close()

    
def iterate_mags(imgs):
    mags = get_magnitications(imgs)
    for m in range(len(mags)):
        #input()
        binary = split_magnification(mags[m], True)
        multiclass = split_magnification(mags[m], False)
        #print(binary)
        #input()
        create_csv(binary, m)
        create_csv(multiclass, m, False)


def main():
    #iterate_mags(NAMES)
    names = []
    for i in os.listdir("./da_dataset/"):
        if i[:3] == "SOB":
            names.append(i)
    iterate_mags(names)
main()  