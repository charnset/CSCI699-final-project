import os
import matplotlib.pyplot as plt
import numpy as np

focus_set = {
	"BENIGN",
    "RANDOM",
    "WIKI"
}

msp = {}

def get_accumulate_msp(data):
	interval = 100
	x = [(i+1) * 100 for i in range(len(data))]
	# x = [i+1 for i in range(len(data))]

	total = 0
	y = []
	data = np.mean(data, axis=1)
	#print(data)
	#print(data.shape)
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
		if label in focus_set:
			print(fn)
			with open(fn, "rb") as f:
				data = np.load(f)
				n = len(data) // 100
				data = data[:100*n].reshape(100, n)
				# print(data)
				# print(data.shape)
				x, y = get_accumulate_msp(data)
				msp[label] = {'x': x, 'y': y}
	print(msp.keys())

	fig, ax = plt.subplots(figsize=(9, 6))

	for label in msp:
		ax.plot(msp[label]['x'], msp[label]['y'], label=label)

	ax.set_xlabel("Cummulative number of queries", fontsize=12)
	ax.set_ylabel("MaxProb", fontsize=12)
	ax.set_title("SC: The Change in MaxProb Over Time", fontsize=16)
	leg = ax.legend(prop={"size":9})
	plt.show()

extract_dir("./SA_MSP")