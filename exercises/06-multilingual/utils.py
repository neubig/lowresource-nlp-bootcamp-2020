#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pickle

def load_data():
    with open("data/absa.pkl","rb") as file:
        data = pickle.load(file)
    return data


# In[12]:


import random
random.seed(11)
import pandas as pd
pd.set_option('precision', 2)

def get_pos_ratio(data_lang):
    return sum([l=="pos" for s,l in data_lang])/len(data_lang)

def print_data(data, max_len=30):
    stats = {}
    for lang in data.keys():
        stats[lang] = {}
        stats[lang]["#train"] = len(data[lang]["train"])
        stats[lang]["#test"] = len(data[lang]["test"])
        stats[lang]["train-pos%"] = get_pos_ratio(data[lang]["train"])
        stats[lang]["test-pos%"] = get_pos_ratio(data[lang]["test"])
        sample_sent, sample_label = random.choice([(s,l) for s,l in data[lang]["train"] if len(s)<max_len])
        stats[lang]["sample"] = sample_sent
        stats[lang]["label"] = sample_label
    
    df = pd.DataFrame.from_dict(stats, orient='index')
    return display(df)

