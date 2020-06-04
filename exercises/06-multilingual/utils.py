#!/usr/bin/env python
# coding: utf-8
import time
import numpy as np
import random
import datetime
import pandas as pd
import pickle


def load_data():
    with open("data/absa.pkl", "rb") as file:
        data = pickle.load(file)
    return data


random.seed(11)
pd.set_option('precision', 3)


def get_pos_ratio(data_lang):
    return sum([l == "pos" for s, l in data_lang])/len(data_lang)


def print_data_stats(data, max_len=30):
    stats = {}
    for lang in data.keys():
        stats[lang] = {}
        stats[lang]["#train"] = len(data[lang]["train"])
        stats[lang]["#test"] = len(data[lang]["test"])
        stats[lang]["train-pos%"] = get_pos_ratio(data[lang]["train"])
        stats[lang]["test-pos%"] = get_pos_ratio(data[lang]["test"])
        sample_sent, sample_label = random.choice(
            [(s, l) for s, l in data[lang]["train"] if len(s) < max_len])
        stats[lang]["sample"] = sample_sent
        stats[lang]["label"] = sample_label

    df = pd.DataFrame.from_dict(stats, orient='index')
    return display(df)


# Function to calculate the accuracy of our predictions vs labels


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def subset_data(data: dict, subset_size=None) -> dict:
    subset = {l: {} for l in data.keys()}
    if subset_size == None:
        subset_size = min([len(data[lang]["train"]) for lang in data.keys()])
    for lang in data:
        subset[lang]["train"] = random.sample(data[lang]["train"], subset_size)
        subset[lang]["test"] = data[lang]["test"]
    return subset
