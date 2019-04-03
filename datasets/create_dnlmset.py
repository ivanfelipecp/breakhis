import os

exec_file = "./dnlmfilter"
dataset = "breakhis/"
DEST = "DNLM_1"

if not os.path.exists(DEST):
	os.mkdir(DEST)
DEST += "/"

# PARAMS
W=15
W_N=7
SIGMA=2
LAMBDA=2
KERNEL_LEN=19
KERNEL_STD=5

images = os.listdir(dataset)
for image in images:
	command = "{} {} {} {} {} {} {} {} {}".format(exec_file, dataset+image, W, W_N, SIGMA, LAMBDA, KERNEL_LEN, KERNEL_STD, DEST)
	os.system(command)
