from shutil import copy2
import os
import sys

ROOT = "./BreaKHis_v1/histology_slides/breast/"
PATH = "./breakhis/"

for i in os.listdir(ROOT):
    # benign - malign
    if i[0] != ".":
        ROOT_1 = ROOT + i + "/"
        for j in os.listdir(ROOT_1):
            # SOB
            if j[0] != ".":
                ROOT_2 = ROOT_1 + j + "/"
                for k in os.listdir(ROOT_2):
                    # 4 types
                    if k[0] != ".":
                        ROOT_3 = ROOT_2 + k + "/"
                        for l in os.listdir(ROOT_3):
                            # SOBs
                            if l[0] != ".":
                                ROOT_4 = ROOT_3 + l + "/"
                                for m in os.listdir(ROOT_4):
                                    # 40x, 100x, 200x, 400x
                                    if m[0] != ".":
                                        ROOT_5 = ROOT_4 + m + "/"
                                        for img in os.listdir(ROOT_5):
                                            if img[0] != ".":
                                                copy2(ROOT_5+img, PATH)