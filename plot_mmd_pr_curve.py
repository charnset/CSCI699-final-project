import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, PrecisionRecallDisplay

label_sets = {}
prediction_sets = {}

def extract_dir(directory):
	filenames = [os.path.join(directory, f) for f in os.listdir(directory)]
	for fn in filenames:
		set_ = fn.split('/')[-1].split('_')[0].upper()
		# print(fn)
		with open(fn, "rb") as f:
			data = np.load(f)
			# print(data)
			# print(data.shape)
			prediction_sets[set_] = data
			if set_ in ["WIKI", "RANDOM"]:
				label_sets[set_] = np.array([1] * data.shape[0])
			else:
				label_sets[set_] = np.array([0] * data.shape[0])

	for set_ in prediction_sets:
		prediction_set = np.mean(prediction_sets[set_])
		sd = np.std(prediction_sets[set_])
		print(set_)
		print("mean", prediction_set)
		print("sd", sd)

	for benign_set in prediction_sets:
		# print(label_set[set_])
		# print(prediction_set[set_])
		if benign_set in ["WIKI", "RANDOM"]: continue

		benign_label_set = label_sets[benign_set]
		# print(benign_label_set, benign_label_set.shape)
		benign_prediction_set = prediction_sets[benign_set]
		# print(benign_prediction_set, benign_prediction_set.shape)

		# sample_size = min(benign_label_set, )

		for attack_set in ["WIKI", "RANDOM"]:
			attack_label_set = label_sets[attack_set]
			attack_prediction_set = prediction_sets[attack_set]

			sampling_size = min(benign_label_set.shape[0], attack_label_set.shape[0])
			# print(sampling_size)

			# print(benign_label_set.shape, attack_label_set.shape)
			# print(benign_prediction_set.shape, attack_prediction_set.shape)
			y_true = np.concatenate((benign_label_set[:sampling_size], attack_label_set[:sampling_size]), axis=0)
			# print(y_true)
			y_score = np.concatenate((benign_prediction_set[:sampling_size], attack_prediction_set[:sampling_size]), axis=0)
			# print(y_score)

			precision, recall, _ = precision_recall_curve(y_true, y_score)

			# y_random = np.array([1] * y_true.shape[0])
			# precision_r, recall_r, _ = precision_recall_curve(y_true, y_random)

			print("BENIGN : {} (N={}), ATTACK : {} (N={}), #SAMPLE: {}, AUPR: {}".format( \
				benign_set, benign_label_set.shape[0], \
				attack_set, attack_label_set.shape[0], \
				y_true.shape[0], \
				auc(recall, precision)))



extract_dir("./QA_MMD_20")