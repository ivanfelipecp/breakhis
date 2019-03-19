import os
import sys
import re
from shutil import copy2

def folders():
	q = ["40", "100", "200", "400"]
	w = ["A","F","PT","TA","DC","LC","MC","PC"]

	for i in q:
		os.mkdir(i)
		for j in w:
			os.mkdir("./"+i+"/"+j)

def get_dict():
	return {
	"A": [],
	"F": [],
	"PT": [],
	"TA": [],
	"DC": [],
	"LC": [],
	"MC": [],
	"PC": []}


TUMOURS = {
	"A": [],
	"F": [],
	"PT": [],
	"TA": [],
	"DC": [],
	"LC": [],
	"MC": [],
	"PC": []
}

MAGS = {
	"40": [],
	"100": [],
	"200": [],
	"400": []
}

cont = {
	"40": 0,
	"100": 0,
	"200": 0,
	"400": 0
}

folders()

# SEPARATING MAGS
dirr = "./breakhis/"
all_imgs = os.listdir(dirr)


for i in all_imgs:
	if i[:3] == "SOB":
		split = i.split("-")
		mag = split[3]
		MAGS[mag].append(i)
		#cont[mag] += 1

#print(MAGS["40"])
#sys.exit(1)

# Separate in every mags
mags_dirr = list(MAGS.keys())
tk = list(TUMOURS.keys())

print(mags_dirr)
print(tk)

for m in mags_dirr:
	tums = get_dict()
	slides = MAGS[m]
	# FOR IN SLIDES IN EVERY MAG, CREATE TUMS
	for s in slides:
		split = re.split("_|-",s)
		key = split[2]
		tums[key].append(s)
	# MOVE EVERY SLIDES IN TUMS SUBLISTS
	cont = 0
	for t in tk:
		images = tums[t]
		for i in images:
			copy2(dirr+i, m+"/"+t+"/"+i)
			cont += 1
	print("AGREGADAS EN MAGNITUD {}: {}".format(m,cont))
	cont += 1