import csv

M_CLASS = {
	"0": "A",
	"1": "F",
	"2": "PT",
	"3": "TA",
	"4": "DC",
	"5": "LC",
	"6": "MC",
	"7": "PC"
}

def get_c():
	return  {
		"0": 0,
		"1": 0,
		"2": 0,
		"3": 0,
		"4": 0,
		"5": 0,
		"6": 0,
		"7": 0
	}


total = get_c()
percs = get_c()
files = ["../csvs/train_100_multi.csv", "../csvs/test_100_multi.csv"]


print("Tipos de cancer: ", M_CLASS, "\n")
for i in ["40", "100", "200", "400"]:
	total = get_c()
	M = 0
	files = ["../csvs/train_{}_multi.csv".format(i), "../csvs/test_{}_multi.csv".format(i)]
	for f in files:
		count = get_c()
		with open(f, "r") as file:
			reader = csv.DictReader(file, fieldnames=["file","class"])
			next(reader)
			for row in reader:
				count[row["class"]] += 1
				total[row["class"]] += 1
				M += 1
			print(f)
			print(count)
			print("\n")

	print("train+test")
	print(total)
	for i in total.keys():
		print(M_CLASS[i],total[i])
	print("Total muestras: ", M, "\n")
	print("Porcentajes")
