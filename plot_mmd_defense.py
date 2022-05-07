import os
import matplotlib.pyplot as plt
import numpy as np

mmd = {}

def get_accumulate_mmd(data):
	interval = 100
	x = [(i+1) * 100 for i in range(len(data))]
	# x = [i+1 for i in range(len(data))]

	total = 0
	y = []
	for i, m in enumerate(data):
		total += m
		avg = total / (i + 1)
		y.append(avg)

	assert len(x) == len(y)

	return x, y

def extract_dir(directory):
	filenames = [os.path.join(directory, f) for f in os.listdir(directory)]
	for fn in filenames:
		label = fn.split('/')[-1].split('_')[0].upper()
		# print(fn)
		with open(fn, "rb") as f:
			data = np.load(f)
			# print(data)
			x, y = get_accumulate_mmd(data)
			mmd[label] = {'x': x, 'y': y}
	print(mmd.keys())

	fig, ax = plt.subplots(figsize=(9, 6))

	for label in mmd:
		ax.plot(mmd[label]['x'], mmd[label]['y'], label=label)

	ax.set_xlabel("Cummulative number of queries", fontsize=12)
	ax.set_ylabel("MMD", fontsize=12)
	ax.set_title("SC: The Change in MMD Over Time", fontsize=14)
	leg = ax.legend(prop={"size":9})
	plt.show()

extract_dir("./SA_MMD")