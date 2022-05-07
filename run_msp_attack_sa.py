import os
import argparse
import random
import time
from operator import methodcaller
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, load_dataset
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
tokenizer = AutoTokenizer.from_pretrained("elozano/tweet_sentiment_eval", local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained("elozano/tweet_sentiment_eval", local_files_only=True)
model.eval()
model.to(device)

# SA datasets
dataset_list = {
    "glue": {"subset": "sst2", "split": "train"},
    "imdb": {"subset": None, "split": "train"},
    "amazon_polarity": {"subset": None, "split": "train"},
    "yelp_polarity": {"subset": None, "split": "train"},
    "app_reviews": {"subset": None, "split": "train"},
    "financial_phrasebank": {"subset": "sentences_50agree", "split": "train"},
    "tweet_eval": {"subset": "sentiment", "split": "train"}
}

# Text equavalent (for renaming feature columns)
dataset_text = {
    "glue": "sentence",
    "imdb": "text",
    "amazon_polarity": "content",
    "yelp_polarity": "text",
    "app_reviews": "review",
    "financial_phrasebank": "sentence",
    "tweet_eval": "text"
}

max_sampling_size = 10000

def read_attack(tsv_path):
    query_data = {}
    with open(tsv_path) as file: # read tsv file
        data = file.read().splitlines()
        #print(len(data))
        data_split = map(methodcaller("rsplit", "\t", 1), data)
        texts, labels = map(list, zip(*data_split))
        query_data["text"] = texts
        query_data["label"] = labels
    dataset_ = Dataset.from_dict(query_data)
    return dataset_

def MSP(dataset, sampling_size):

    def encode(query):
        encoded_query = tokenizer(query["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        encoded_query["input_ids"] = encoded_query["input_ids"][0] #collapse to 1d
        encoded_query["attention_mask"] = encoded_query["attention_mask"][0] #collapse to 1d
        return encoded_query

    dataset = dataset.shuffle(seed=0).select(range(sampling_size))
    dataset = dataset.map(encode)

    dataset.set_format(type="torch", columns=['input_ids', 'attention_mask'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

    max_softmax_data = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            #if i == 2:
                #break

            #print(batch)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            #print(outputs)

            answer_softmax = softmax(outputs.logits)
            #print(answer_softmax)
            max_softmax = torch.max(answer_softmax, dim=1)
            #print(max_softmax)
            max_softmax_data.append(max_softmax.values)

    #logger.info(len(max_softmax_data))
    #logger.info(max_softmax_data[0])
    #logger.info(max_softmax_data[0].shape)
    max_softmax_data = torch.cat(max_softmax_data, axis=0).cpu().numpy()
    #logger.info(max_softmax_data.shape)
    #print(max_softmax_data)

    return max_softmax_data

def extract_dir(directory):
    attack_file_s = [os.path.join(directory, f) for f in os.listdir(directory) if ".tsv" in f and "train" in f]
    for attack_file in attack_file_s:
        print(attack_file)
        dataset = read_attack(attack_file)
        logger.info("Reading {} ...".format(attack_file.split('/')[-1]))
        logger.info("Number of questions: {}".format(dataset.num_rows))
        print(dataset)

        sampling_size = min(max_sampling_size, dataset.num_rows)

        logger.info("Computing MSP ...")
        max_softmax_data = MSP(dataset, sampling_size)
        logger.info("{}".format(max_softmax_data.shape))

        save_file = attack_file.split('/')[-1].split('.')[0].split('_')[-1] + "_msp.npy"
        logger.info("Saving MSP (numpy) to a file: {}".format(save_file))

        with open(save_file, 'wb') as f:
            np.save(f, max_softmax_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', type=str, help="path to directory of attack datasets: random and wiki")
    args = parser.parse_args()

    start_time = time.time()
    logger.info('-'*20 + "ATTACK DATASETS" + '-'*20)
    extract_dir(args.a) #attack
