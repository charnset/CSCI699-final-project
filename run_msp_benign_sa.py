import os
import argparse
import math
import random
import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import concatenate_datasets, load_dataset
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

def read_data(dt):
    subset = dataset_list[dt]["subset"]
    split = dataset_list[dt]["split"]
    dataset_ = load_dataset(path=dt, name=subset, split=split) \
        if subset else load_dataset(path=dt, split=split)
    if "text" not in dataset_.features:
        dataset_ = dataset_.rename_column(dataset_text[dt], "text")
    # only keep the "text" column
    dataset_ = dataset_.remove_columns([col for col in dataset_.column_names if col != "text"])
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

def extract_dir():
    dataset_sample_list = []
    sub_sampling_size = math.ceil(max_sampling_size / len(dataset_list))

    for i, dataset_name in enumerate(dataset_list):
        dataset = read_data(dataset_name)
        logger.info("Reading {} ...".format(dataset_name))
        logger.info("Number of questions: {}".format(dataset.num_rows))
        #print(dataset)

        dataset_sample = dataset.shuffle(seed=0).select(range(sub_sampling_size))
        #print(dataset_sample)
        dataset_sample_list.append(dataset_sample)

    #print(dataset_sample_list)
    benign_dataset = concatenate_datasets(dataset_sample_list)
    print(benign_dataset)
    
    logger.info("Computing MSP ...")
    max_softmax_data = MSP(dataset, max_sampling_size)
    logger.info("{}".format(max_softmax_data.shape))

    save_file = "benign_msp.npy"
    logger.info("Saving MSP (numpy) to a file: {}".format(save_file))


    with open(save_file, 'wb') as f:
        np.save(f, max_softmax_data)
    


if __name__ == "__main__":
    start_time = time.time()
    extract_dir()
