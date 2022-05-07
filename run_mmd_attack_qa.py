import os
import argparse
import sys
import random
import time
import json
import gzip
import numpy as np
from allennlp.common.file_utils import cached_path
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
import torch
from tqdm import tqdm
import logging

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("device: {}".format(device))

# Load tokenizer and model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model.eval()
model.to(device)

def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def read_data(path):
    with gzip.open(cached_path(path), 'rb') as f:
        #query_data = []
        query_data = {}
        query_data["context"] = []
        query_data["question"] = []
        for i, line in enumerate(f):
            #if i == 500:
                #break
            obj = json.loads(line)

            # Skip headers.
            if i == 0 and 'header' in obj:
                continue

            context = obj['context']
            context = context.replace('[PAR]', '\n\n')
            context = context.replace('[DOC]', '\n\n')

            # Iterate questions:
            for qa in obj['qas']:
                #query_data.append({"context": context, "question": qa['question']})
                query_data["context"].append(context)
                query_data["question"].append(qa["question"])

        return query_data

def read_attack(path):
    with open(path, 'r') as f:
        json_data = json.loads(f.read())
    
    query_data = {}
    query_data["context"] = []
    query_data["question"] = []
    for i, instance in enumerate(json_data["data"]):
        for j, paragraph in enumerate(instance["paragraphs"]):
            context = paragraph["context"]
            for k, qa in enumerate(paragraph["qas"]):
                query_data["context"].append(context)
                query_data["question"].append(qa["question"])

    return query_data


def create_dataset(query_data, sampling_size):
    
    def encode(query):

        encoded_query = tokenizer(query["context"], query["question"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        encoded_query["input_ids"] = encoded_query["input_ids"][0] #collapse to 1d
        encoded_query["token_type_ids"] = encoded_query["token_type_ids"][0] #collapse to 1d
        encoded_query["attention_mask"] = encoded_query["attention_mask"][0] #collapse to 1d

        return encoded_query

    dataset = Dataset.from_dict(query_data).shuffle(seed=0).select(range(sampling_size))
    dataset = dataset.map(encode)

    dataset.set_format(type="torch", columns=['input_ids', 'token_type_ids', 'attention_mask'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

    embedded_dataset = []  
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            mean_outputs = mean_pooling(outputs, batch["attention_mask"])
            #print(mean_outputs.shape)
            embedded_dataset.append(mean_outputs)
    #print(len(embedded_dataset), embedded_dataset[0].shape)
    embedded_dataset = torch.cat(embedded_dataset, axis=0)

    return embedded_dataset


def extract_dir(training_file, directory):
    testing_file_s = [os.path.join(directory, f) for f in os.listdir(directory)]
    for testing_file in testing_file_s:
        data_x = read_data(training_file)
        data_y = read_attack(testing_file)
        logger.info("Reading {} ...".format(training_file))
        logger.info("Number of questions: {}".format(len(data_x["question"])))
        logger.info("Reading {} ...".format(testing_file))
        logger.info("Number of questions: {}".format(len(data_y["question"])))

        sampling_size = min(len(data_x["question"]), len(data_y["question"]))
                       
        dataset_x = create_dataset(data_x, sampling_size)
        dataset_y = create_dataset(data_y, sampling_size)
        logger.info("Embedding {} ...".format(training_file))
        logger.info("{}".format(dataset_x.shape))
        logger.info("Embedding {} ...".format(testing_file))
        logger.info("{}".format(dataset_y.shape))

        logger.info("Running Time (Before MMD): {:.4f} sec".format(time.time() - start_time))

        logger.info("MMD: {}".format(MMD(dataset_x, dataset_y, "rbf")))

        logger.info("Running Time (After MMD): {:.4f} sec".format(time.time() - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=str, help="path to training dataset")
    parser.add_argument('-a', type=str, help="path to directory of attack datasets: random and wiki")
    args = parser.parse_args()

    start_time = time.time()
    logger.info('-'*20 + "ATTACK DATASETS" + '-'*20)
    extract_dir(args.b, args.a) #attack


