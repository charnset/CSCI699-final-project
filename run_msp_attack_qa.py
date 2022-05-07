import os
import argparse
import sys
import random
import time
import json
import gzip
import numpy as np
from allennlp.common.file_utils import cached_path
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
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

softmax = torch.nn.Softmax(dim=1)

# Load tokenizer and model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad", local_files_only=True)
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad", local_files_only=True)
model.eval()
model.to(device)

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

def MSP(query_data):

    def encode(query):
        encoded_query = tokenizer(query["question"], query["context"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        encoded_query["input_ids"] = encoded_query["input_ids"][0] #collapse to 1d
        encoded_query["token_type_ids"] = encoded_query["token_type_ids"][0] #collapse to 1d
        encoded_query["attention_mask"] = encoded_query["attention_mask"][0] #collapse to 1d
        return encoded_query

    dataset = Dataset.from_dict(query_data).shuffle(seed=0)
    dataset = dataset.map(encode)

    dataset.set_format(type="torch", columns=['input_ids', 'token_type_ids', 'attention_mask'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

    def get_answer(input_ids, answer_start_scores, answer_end_scores):
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
        )
        print(answer)

    max_softmax_data = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            answer_start_softmax_s = softmax(outputs.start_logits)
            answer_end_softmax_s = softmax(outputs.end_logits)

            answer_start_softmax = torch.max(answer_start_softmax_s, dim=1)
            answer_end_softmax = torch.max(answer_end_softmax_s, dim=1)

            answer_start_max_softmax = torch.reshape(answer_start_softmax.values, (1, -1))
            answer_end_max_softmax = torch.reshape(answer_end_softmax.values, (1, -1))

            ### stack answer_start_value (index-0) and answer_end_value (index-1)
            ### answer_stack.shape = (2, batch_size)
            answer_stack_max_softmax = torch.cat((answer_start_max_softmax, answer_end_max_softmax), dim=0)

            ### average between max softmax of "start" and max softmax of "end" for each answer
            ### answer_start_max_softmax + answer_end_max_softmax) / 2
            answer_mean_max_softmax = torch.mean(answer_stack_max_softmax, dim=0)

            max_softmax_data.append(answer_mean_max_softmax)

    max_softmax_data = torch.cat(max_softmax_data, axis=0).cpu().numpy()

    return max_softmax_data


def extract_dir(directory):
    filenames = [os.path.join(directory, f) for f in os.listdir(directory)]
    for fn in filenames:
        data = read_attack(fn)
        logger.info("Reading {} ...".format(fn))
        logger.info("Number of questions: {}".format(len(data["question"])))


        logger.info("Computing MSP ...")
        max_softmax_data = MSP(data)
        logger.info("{}".format(max_softmax_data.shape))


        save_file = fn.split('/')[-1].split('.')[0].split('_')[-1] + "_msp.npy"
        logger.info("Saving MSP (numpy) to a file: {}".format(save_file))


        with open(save_file, 'wb') as f:
            np.save(f, max_softmax_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', type=str, help="path to directory of attack datasets")
    args = parser.parse_args()

    start_time = time.time()
    logger.info('-'*20 + "ATTACK DATASETS" + '-'*20)
    extract_dir(args.a) #attack

