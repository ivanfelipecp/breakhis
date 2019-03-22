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

count_classes = {
	"0": 0,
	"1": 0,
	"2": 0,
	"3": 0,
	"4": 0,
	"5": 0,
	"6": 0,
	"7": 0
}

total = count_classes.copy()
percs = count_classes.copy()
files = ["train_40_multi.csv", "test_40_multi.csv"]
M = 0

print("Tipos de cancer: ", M_CLASS, "\n")
for f in files:
	count = count_classes.copy()
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
print("Total muestras: ", M, "\n")
print("Porcentajes")
custom = count_classes.copy()
for i in total.keys():
	p = total[i]/M
	percs[i] += p
	custom[i] = 1/8
custom    
print(percs)
print(custom)

