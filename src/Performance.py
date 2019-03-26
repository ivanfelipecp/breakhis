import csv
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
import scikitplot as skplt

from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from ConfusionMatrix import plot_confusion_matrix

# ["B", "M"] if classes == 2 else ["A", "F", "PT", "TA", "DC", "LC", "MC", "PC"]    
class Performance:
	def __init__(self, path, n_classes):
		self.path = path
		self.n_classes = n_classes
		self.predicts_label = "predicts"
		self.targets_label = "targets"
		self.file_results = "results_"
		self.file_ext = ".json"
		self.img_cm = "confusion_matrix_"
		self.img_roc = "roc_curve_"
		self.img_ext = ".png"
		self.targets = None
		self.predicts = None
		self.patients = None
		self.accuracy = None
		self.loss = None

	def clear(self):
		self.targets = []
		self.predicts = []
		self.patients = []
		self.outputs = []
		self.loss = None

	def get_classes(self):
		classes = {}
		for i in range(self.n_classes):
			classes[i] = 0
		return classes
		
		
	def add_results(self, targets, predicts, patients, outputs):
		self.targets += targets
		self.predicts += predicts
		self.patients += patients
		self.outputs += outputs

	def get_patients(self):
		patients = {}
		for i in range(len(self.patients)):
			patient = self.patients[i]
			if not patient in patients:
				patients[patient] = {self.targets_label:self.get_classes(), self.predicts_label:self.get_classes()}

			# patients
			predict = 1 if self.predicts[i] == self.targets[i] else 0

			patients[patient][self.targets_label][self.targets[i]] += 1
			patients[patient][self.predicts_label][self.predicts[i]] += predict			

		return patients

	def end_epoch(self, epoch):
		patients = self.get_patients()
		results = self.do_metrics(patients)
		
		self.save(epoch, results)

	def save(self, epoch, results):
		epoch = str(epoch)
		self.save_results(epoch, results)
		self.save_confusion_matrix(epoch)
		self.save_roc_curve(epoch)

	def save_results(self, epoch, results):
		with open(self.path + self.file_results + epoch + self.file_ext, "w") as f:
			json.dump(results, f)

	def save_confusion_matrix(self, epoch):
		fig_cm = plot_confusion_matrix(self.targets, self.predicts, self.n_classes, normalize=True,
                      title='Confusion matrix')
		fig_cm.savefig(self.path + self.img_cm + epoch + self.img_ext)

	def save_roc_curve(self, epoch):
		skplt.metrics.plot_roc(self.targets, self.outputs)
		plt.savefig(self.path + self.img_roc + epoch + self.img_ext)

	def do_metrics(self, patients):
		# General use
		(targets, predicts) = self.remove_patients(patients)

		# Patient level accuracy
		patient_level_accuracy = self.patients_level_accuracy(patients)
		
		# Image level accuracy
		image_level_accuracy = accuracy_score(self.targets, self.predicts)
		self.accuracy = image_level_accuracy

		# ROC analysis
		# Confusion matrix
		confusion_m = confusion_matrix(self.targets, self.predicts)

		FP = confusion_m.sum(axis=0) - np.diag(confusion_m)  
		FN = confusion_m.sum(axis=1) - np.diag(confusion_m)
		TP = np.diag(confusion_m)
		TN = confusion_m[:].sum() - (FP + FN + TP)

		# Sensitivity, hit rate, recall, or true positive rate
		TPR = TP/(TP+FN)
		# Specificity or true negative rate
		TNR = TN/(TN+FP) 
		# Precision or positive predictive value
		PPV = TP/(TP+FP)
		# Negative predictive value
		NPV = TN/(TN+FN)
		# Fall out or false positive rate
		FPR = FP/(FP+TN)
		# False negative rate
		FNR = FN/(TP+FN)
		# False discovery rate
		FDR = FP/(TP+FP)

		# Overall accuracy
		ACC = (TP+TN)/(TP+FP+FN+TN)

		results = {
			"loss": float(self.loss),
			"patient_level_accuracy": float(patient_level_accuracy),
			"image_level_accuracy": float(image_level_accuracy),
			"TP":self.map_float(TP),
			"FP":self.map_float(FP),
			"TN":self.map_float(TN),
			"FN":self.map_float(FN),
			"TPR":self.map_float(TPR),
			"TNR":self.map_float(TNR),
			"PPV":self.map_float(PPV),
			"NPV":self.map_float(NPV),
			"FPR":self.map_float(FPR),
			"FNR":self.map_float(FNR),
			"FDR":self.map_float(FDR),
			"ACC":self.map_float(ACC)
		}

		return results

	def map_float(self, array):
		array = np.nan_to_num(array)
		array = list(array)
		return list(map(float, array))

	def patients_level_accuracy(self, patients):
		keys = patients.keys()
		patient_level_accuracy = 0
		for i in keys:
			patient_level_accuracy += self.get_patient_level_accuracy(patients[i])
		
		patient_level_accuracy /= len(keys)
		return patient_level_accuracy

	def get_patient_level_accuracy(self, patient):
		accuracy = 0
		keys = patient[self.targets_label].keys()
		
		Nc = 0
		Nt = 0
		for i in keys:			
			target = patient[self.targets_label][i]
			predict = patient[self.predicts_label][i]

			# Si tiene imágenes de una clase de tumor...
			if target > 0:
				Nt += target
				Nc += predict

		S = (Nc/Nt)
		return S

	def remove_patients(self, patients):
		predicts = self.get_classes()
		targets = self.get_classes()
		keys = patients.keys()

		for i in keys:
			patient = patients[i]
			for j in patient[self.targets_label].keys():
				targets[j] += patient[self.targets_label][j]
				predicts[j] += patient[self.predicts_label][j]
		return (targets, predicts)		
"""
performance = Performance(".", "mul")
patients = [1] * 8 #+ [2] * 6
targets =  [0,1,2,3,4,5,6,7] #+ [0,1,2,3,4,5]
predicts = [0,1,2,3,4,5,6,7] #+ [0,1,2,3,4,1]
outputs = []

for i in range(8):
	out = [0] * 8
	out[i] = 0.9
	outputs.append(out)

performance.clear()
performance.add_results(targets, predicts, patients, outputs)
performance.loss = 0
performance.end_epoch(0)
"""