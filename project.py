

from datasets import Dataset
from datasets import DatasetDict
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import Trainer
import csv
import numpy as np
import evaluate

# labels to ids
id2label = {0: "negative", 1: "neutral", 2:"positive"}
label2id = {v: k for k, v in id2label.items()}

# fuction to creat dataset from file
def file_to_dataset(filename):
    tweets_list = []
    labels_list = []
    with open(filename, "r", encoding="utf8") as file:
        tsv_reader = csv.reader(file, delimiter="\t")
        for line in tsv_reader:
            (id, label, tweet) = line
            tweets_list.append(tweet)
            labels_list.append(label2id[label])
    return Dataset.from_dict({"text": tweets_list, "label":labels_list})

# creat the dataset
dev_data = file_to_dataset("semeval-2017-tweets_Subtask-A/downloaded/twitter-2016dev-A.tsv")
devtest_data = file_to_dataset("semeval-2017-tweets_Subtask-A/downloaded/twitter-2016devtest-A.tsv")
dataset = DatasetDict({"train": dev_data, "validation":devtest_data})

# show sample
sample = dataset["train"].shuffle().select(range(3))
for row in sample:
    print(f"\n'>>> tweet: {row['text']}'")
    print(f"'>>> Label: {row['label']}'")


# load pretraind modle and tokenizer from checkpoint
model_checkpoint = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3, 
    id2label=id2label,
    label2id=label2id)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# tokenize the data
def tokenize_function(tweet):
    return tokenizer(tweet["text"] , truncation=True, return_tensors="pt", padding=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# define the metrics computation
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# initialize trainer 
trainer = Trainer(
    model,
    TrainingArguments("test-trainer", evaluation_strategy="epoch"),
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()