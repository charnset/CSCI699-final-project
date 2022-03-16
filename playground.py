import argparse
import json
import gzip
from utils import compute_mmd
import numpy as np
from allennlp.common.file_utils import cached_path
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

parser = argparse.ArgumentParser()
parser.add_argument('-x', type=str, help="path to first dataset (x)")
parser.add_argument('-y', type=str, help="path to second dataset (y)")
args = parser.parse_args()

def read_data(path):
	with gzip.open(cached_path(path), 'rb') as f:
		query_data = []
		for i, line in enumerate(f):
			# if i == 101:
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

query_data_x = read_data(args.x)
dataset_x = create_dataset(query_data_x)
query_data_y = read_data(args.y)
dataset_y = create_dataset(query_data_y)
# print(query_data_x)
print(dataset_x.shape)
# print(query_data_x)
print(dataset_y.shape)

# Compute MMD of dataset_x and dataset_y
print("x: {}, y: {}, MMD: {:.3f}".format(args.x, args.y, compute_mmd(dataset_x, dataset_y)))
