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

def read_data(path):
    with gzip.open(cached_path(path), 'rb') as f:
        #query_data = []
        query_data = {}
        query_data["qid"] = []
        query_data["context"] = []
        query_data["question"] = []
        for i, line in enumerate(f):
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
                query_data["qid"].append(qa["qid"])
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
    '''
    print(dataset[0])
    print(dataset[1])
    print(dataset[2])
    '''

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
            #if i == 6:
                #break

            #print(batch)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            #print(outputs)
            answer_start_softmax_s = softmax(outputs.start_logits)
            answer_end_softmax_s = softmax(outputs.end_logits)

            answer_start_softmax = torch.max(answer_start_softmax_s, dim=1)
            answer_end_softmax = torch.max(answer_end_softmax_s, dim=1)
            #print(answer_start_softmax)
            #print(answer_end_softmax)

            answer_start_max_softmax = torch.reshape(answer_start_softmax.values, (1, -1))
            answer_end_max_softmax = torch.reshape(answer_end_softmax.values, (1, -1))
            #print(answer_start_max_softmax)
            #print(answer_end_max_softmax)

            ### stack answer_start_value (index-0) and answer_end_value (index-1)
            ### answer_stack.shape = (2, batch_size)
            answer_stack_max_softmax = torch.cat((answer_start_max_softmax, answer_end_max_softmax), dim=0)
            #print(answer_stack_max_softmax)

            ### average between max softmax of "start" and max softmax of "end" for each answer
            ### answer_start_max_softmax + answer_end_max_softmax) / 2
            answer_mean_max_softmax = torch.mean(answer_stack_max_softmax, dim=0)
            #print(answer_mean_max_softmax)

            max_softmax_data.append(answer_mean_max_softmax)

    #logger.info(len(max_softmax_data))
    #logger.info(max_softmax_data[0])
    #logger.info(max_softmax_data[0].shape)
    max_softmax_data = torch.cat(max_softmax_data, axis=0).cpu().numpy()
    #logger.info(max_softmax_data.shape)
    #print(max_softmax_data)

    return max_softmax_data



def extract_dir(directory):
    filenames = [os.path.join(directory, f) for f in os.listdir(directory) if ".gz" in f]
    for fn in filenames:
        data = read_data(fn)
        logger.info("Reading {} ...".format(fn))
        logger.info("Number of questions: {}".format(len(data["question"])))


        logger.info("Computing MSP ...")
        max_softmax_data = MSP(data)
        logger.info("{}".format(max_softmax_data.shape))


        save_file = fn.split('/')[-1].split('.')[0] + "_msp.npy"
        logger.info("Saving MSP (numpy) to a file: {}".format(save_file))


        with open(save_file, 'wb') as f:
            np.save(f, max_softmax_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, help="path to directory of in-domain datasets")
    parser.add_argument('-o', type=str, help="path to directory of out-domain datasets")
    args = parser.parse_args()

    start_time = time.time()
    logger.info('-'*20 + "IN-DOMAIN DATASETS" + '-'*20)
    extract_dir(args.i) #in-domain
    logger.info('-'*20 + "OUT-DOMAIN DATASETS" + '-'*20)
    extract_dir(args.o) #out-domain





'''
start_time = time.time()


data = read_data("in_domain_dev/SQuAD.jsonl.gz")
logger.info("read data")
logger.info("{}".format(len(data["question"])))


logger.info("time")
logger.info("{:.4f}".format(time.time() - start_time))


logger.info("compute MSP")
max_softmax_data = MSP(data)


logger.info("time")
logger.info("{:.4f}".format(time.time() - start_time))


logger.info("save MSP (numpy) to a file")
with open('SQuAD_msp.npy', 'wb') as f:
    np.save(f, max_softmax_data)
'''




'''
get_answer(batch["input_ids"][0], answer_start_scores_s[0], answer_end_scores_s[0])
get_answer(batch["input_ids"][1], answer_start_scores_s[1], answer_end_scores_s[1])
break
'''
