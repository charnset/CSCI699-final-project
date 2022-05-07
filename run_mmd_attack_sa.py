import os
import argparse
import random
import time
from operator import methodcaller
import numpy as np
from transformers import AutoTokenizer, AutoModel
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

# Load tokenizer and model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
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
    "tweet_eval": "text",
    "attack": "sentence"
}

max_sampling_size = 10000

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

def read_data(dt):
    subset = dataset_list[dt]["subset"]
    split = dataset_list[dt]["split"]
    dataset_ = load_dataset(path=dt, name=subset, split=split) \
        if subset else load_dataset(path=dt, split=split)
    if "text" not in dataset_.features:
        dataset_ = dataset_.rename_column(dataset_text[dt], "text")
    return dataset_

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

def create_dataset(dataset, sampling_size): #embedding dataset
    
    def encode(query):
        encoded_query = tokenizer(query["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        encoded_query["input_ids"] = encoded_query["input_ids"][0] #collapse to 1d
        encoded_query["token_type_ids"] = encoded_query["token_type_ids"][0] #collapse to 1d
        encoded_query["attention_mask"] = encoded_query["attention_mask"][0] #collapse to 1d

        return encoded_query

    dataset = dataset.shuffle(seed=0).select(range(sampling_size))
    #dataset = dataset.shuffle(seed=0).select(range(3))
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

def extract_dir(directory, training_dataset_name="tweet_eval"):
    testing_file_s = [os.path.join(directory, f) for f in os.listdir(directory) if ".tsv" in f and "train" in f]
    for testing_file in testing_file_s:
        print(testing_file)
        data_x = read_data(training_dataset_name)
        data_y = read_attack(testing_file)
        logger.info("Reading {} ...".format(training_dataset_name))
        logger.info("Number of questions: {}".format(data_x.num_rows))
        logger.info("Reading {} ...".format(testing_file.split('/')[-1]))
        logger.info("Number of questions: {}".format(data_y.num_rows))
        #print(data_y)

        sampling_size = min(max_sampling_size, min(data_x.num_rows, data_y.num_rows))

        dataset_x = create_dataset(data_x, sampling_size)
        dataset_y = create_dataset(data_y, sampling_size)
        logger.info("Embedding {} ...".format(training_dataset_name))
        logger.info("{}".format(dataset_x.shape))
        logger.info("Embedding {} ...".format(testing_file.split('/')[-1]))
        logger.info("{}".format(dataset_y.shape))

        logger.info("Running Time (Before MMD): {:.4f} sec".format(time.time() - start_time))

        logger.info("MMD: {}".format(MMD(dataset_x, dataset_y, "rbf")))

        logger.info("Running Time (After MMD): {:.4f} sec".format(time.time() - start_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', type=str, help="path to directory of attack datasets: random and wiki")
    args = parser.parse_args()

    start_time = time.time()
    logger.info('-'*20 + "ATTACK DATASETS" + '-'*20)
    extract_dir(args.a) #attack

