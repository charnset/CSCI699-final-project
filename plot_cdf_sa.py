import os
import time
import numpy as np
import matplotlib.pyplot as plt 

#in_domain = set(["HotpotQA", "NewsQA", "NaturalQuestions", "SearchQA", "TriviaQA"])
#out_domain = set(["BioASQ", "DuoRC", "RelationExtraction", "DROP", "RACE", "TextbookQA"])
#attacks = set(["RANDOM", "WIKI"])

benign_set = {
    "SST2",
    "IMDB",
    "AMAZON",
    "YELP",
    "APPS",
    "FINANCE",
    "TWEET"
}

attack_set = {
    "RANDOM",
    "WIKI"
}

cdf = {}

def get_cdf(data):

	# start with sorted list of data
	x = [i for i in sorted(data)]

	current_sum = 0
	total_sum = sum(data)
	cdf = []

	for xs in x:
		# get the sum of the values less than each data point and store that value
		# this is normalised by the sum of all values
		current_sum += xs
		cum_val = current_sum/total_sum
		cdf.append(cum_val)

	return x, cdf

def extract_dir(directory):
	filenames = [os.path.join(directory, f) for f in os.listdir(directory)]
	for fn in filenames:
		label = fn.split('/')[-1].split('_')[0].upper()
		#print(fn)
		with open(fn, "rb") as f:
			data = np.load(f)
			x, y = get_cdf(data)
			cdf[label] = {'x': x, 'y': y}
	print(cdf.keys())

	#Plot CDF
	fig, ax = plt.subplots(figsize=(8, 6))

	# ax.plot(cdf["TWEET"]['x'], cdf["TWEET"]['y'], linewidth=4, linestyle="-", color="#39ff14", label="TWEET")

	for label in benign_set:
		ax.plot(cdf[label]['x'], cdf[label]['y'], linestyle='-', label=label)

	for label in attack_set:
		ax.plot(cdf[label]['x'], cdf[label]['y'], linestyle="--", label=label)

	# ax.plot(cdf["BENIGN"]['x'], cdf["BENIGN"]['y'], linewidth=1, linestyle="-", color="#39ff14", label="BENIGN")

	ax.set_xlabel("Maximum Softmax Probability", fontsize=12)
	ax.set_ylabel("Fraction of Data", fontsize=12)
	ax.set_title("CDF of MSP for Queries on RoBERTa-base pre-trained on TWEET", fontsize=14)
	leg = ax.legend(prop={"size":9})
	plt.show()


extract_dir("./SA_MSP")
