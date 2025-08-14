#---- Load Libraries
import json
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

#---- Load Data
file_path='./fars-news-1399.json'
with open(file_path,'r',encoding='utf-8') as data_file:
  data=json.load(data_file)
data=data[:50000]
print(f'The count of selected data is: {len(data)}')
