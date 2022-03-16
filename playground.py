import os
import argparse
import json
import gzip
from utils import random_sampling, MMD
import numpy as np
from allennlp.common.file_utils import cached_path
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

memo = {}

def read_data(path):
	with gzip.open(cached_path(path), 'rb') as f:
		query_data = []
		for i, line in enumerate(f):
			# if i == 201:
			# 	break
			obj = json.loads(line)

			# Skip headers.
			if i == 0 and 'header' in obj:
				continue

			context = obj['context']
			context = context.replace('[PAR]', '\n\n')
			context = context.replace('[DOC]', '\n\n')

			# Iterate questions:
			questions = []
			for qa in obj['qas']:
				questions.append(qa['question'])

			query_data.append({"context": context, "questions": questions})

		return query_data

def create_dataset(query_data):
	dataset = []
	for i, query in enumerate(query_data):
		context = query["context"]
		embedded_context = model.encode(context)
		questions = query["questions"]
		embedded_questions = model.encode(questions)
		# print(embedded_context.shape)
		# print(embedded_questions.shape)
		for embedded_qt in embedded_questions:
			embedded_query = np.concatenate((embedded_context, embedded_qt))
			# print(embedded_query.shape)
			dataset.append(embedded_query)

	return np.array(dataset)

def cached_dataset(path):
	if path in memo:
		print("Cached: {}".format(path))
		return memo[path]
	else:
		print("Loading: {} ...".format(path))
		query_data = read_data(path)
		dataset = create_dataset(query_data)
		# print(dataset.shape)
		dataset = random_sampling(dataset, 500)
		# print(dataset_x.shape)
		memo[path] = dataset
		return dataset

def extract_dir(dataset_x, directory):
	filenames = [os.path.join(directory, f) for f in os.listdir(directory) if ".gz" in f]
	for fn in filenames:
		dataset_y = cached_dataset(fn)
		# Compute MMD of dataset_x and dataset_y
		print("MMD: {:.3f}".format(MMD(dataset_x, dataset_y, "rbf")))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-b', type=str, help="path to training dataset")
	parser.add_argument('-i', type=str, help="path to directory of in-domain datasets")
	parser.add_argument('-o', type=str, help="path to directory of out-domain datasets")
	args = parser.parse_args()

	print("TRAINING DATASET (x): {}".format(args.b))
	dataset_x = cached_dataset(args.b)
	print("IN-DOMAIN DATASET (y)")
	extract_dir(dataset_x, args.i) #in-domain
	print("OUT-DOMAIN DATASET (y)")
	extract_dir(dataset_x, args.o) #out-domain


# Compute MMD of dataset_x and dataset_y
# print("x: {}, y: {}, MMD: {:.3f}".format(args.x, args.y, MMD(dataset_x, dataset_y, "rbf")))
