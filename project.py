# %%
## Imports

from datasets import Dataset
from datasets import DatasetDict
from datasets import concatenate_datasets
from datasets import load_metric
from datasets import Features
from datasets import Value
from datasets import ClassLabel
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import Trainer
from transformers import pipeline
import csv
import numpy as np
import os
import pandas
from imblearn.over_sampling import SMOTE
import torch
import numpy
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# %%
### Data to pandas

# labels to ids
id2label = {0: "negative", 1: "neutral", 2:"positive"}
label2id = {v: k for k, v in id2label.items()}

# fuction to creat dataset from file
def pandas_from_file(filename):
    # print("\n\n************ "+filename)
    data = pandas.read_table(filename, delimiter="\t", names=["label", "tweet"], usecols=[1,2])
    # print(data.sample(n=3))
    return data

# training data
directory = 'semeval-2017-tweets_Subtask-A/downloaded/'
training_dataframes = []
for filename in os.listdir(directory):
    training_dataframes.append(pandas_from_file(directory+filename))

# mearge
training_merged = pandas.concat(training_dataframes).drop_duplicates()

# test data
testing_dataframe = pandas_from_file("SemEval2017-task4-test.subtask-A.english.txt")

# sanity checks
assert(training_merged.notna().values.all())
assert(training_merged["label"].isin(label2id.keys()).all())

print("TRAIN DATA:")
print(training_merged["label"].value_counts())
print(training_merged.sample(n=3))

print("TEST DATA:")
print(testing_dataframe["label"].value_counts())
print(testing_dataframe.sample(n=3))


# %%
# ## crate dict with all rations


# # start dict with the training data
# ratiod_dataframes = [training_merged]

# # rebalnce based on geven rations and add to dict
# no_neutral = training_merged[training_merged["label"] != "neutral"]
# neutrals = training_merged[training_merged["label"] == "neutral"]

neg_count = training_merged[training_merged["label"] == "negative"].shape[0]
neut_count = training_merged[training_merged["label"] == "neutral"].shape[0]
pos_count = training_merged[training_merged["label"] == "positive"].shape[0]
total = neg_count + pos_count

# ratios = [pos_count, int(0.1*neg_count), total-int(0.1*neg_count), int(0.01*neg_count), total-int(0.01*neg_count) , 2*neg_count]
# for negs in ratios:
#     print({"negative": negs, "positive": total-negs})
#     print(no_neutral["tweet"])
#     print(no_neutral["label"])
#     smote = SMOTE(sampling_strategy={"negative": negs, "positive": total-negs})
#     resampled = smote.fit_resample(no_neutral["tweet"], no_neutral["label"])
#     # ratiod_dataframes.append(pandas.concat([resampled, neutrals]))

# # # sanity
# # for df in ratiod_dataframes.items():
# #     print(df['label'].value_counts()) 


# %%
### pandas to dataset

# dataset features
features = Features({'tweet': Value('string'), 'label': ClassLabel(names=list(label2id.keys()))})

# create the dataset
train_dataset = Dataset.from_pandas(training_merged.replace({"labels": label2id}), preserve_index=False, features=features)
test_dataset = Dataset.from_pandas(testing_dataframe.replace({"labels": label2id}), preserve_index=False, features=features)

dataset = DatasetDict({"train": train_dataset, "validation":test_dataset})

# ratiod_datasets = {}
# for (ratio , training_pand) in ratiod_dataframes.items(): 
#     # convert pandas to dataset
#     train_dataset = Dataset.from_pandas(training_pand.replace({"labels": label2id}), preserve_index=False, features=features)
#     test_dataset = Dataset.from_pandas(testing_dataframe.replace({"labels": label2id}), preserve_index=False, features=features)

#     # create the dataset
#     dataset = DatasetDict({"train": train_dataset, "validation":test_dataset})

#     # add to dict
#     ratiod_datasets[ratio] = dataset


# %%
## Metrics
# define metrics function
def compute_metrics(eval_pred):
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    metrics = ["precision", "recall", "f1"]
    results = {m : load_metric(m).compute(predictions=predictions, references=labels, average="micro")[m] for m in metrics}
    results["accuracy"] = load_metric("accuracy").compute(predictions=predictions, references=labels)["accuracy"]
    print("classification report:")
    print(classification_report(labels, predictions))
    print("confusion matrix:")
    print(confusion_matrix(labels, predictions))
    return results

# %%
### Model and Training

negs = [0, pos_count, int(0.1*neg_count), total-int(0.1*neg_count), int(0.01*neg_count), total-int(0.01*neg_count) , 2*neg_count]

for n in negs:  

    ## Model and Tokenizer

    # load pretraind modle and tokenizer from checkpoint
    model_checkpoint = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3, 
        id2label=id2label,
        label2id=label2id)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    ## Tokinization

    # tokenize the data
    tokenized_datasets = dataset.map(
        (lambda tweet: tokenizer(tweet["tweet"] , truncation=True, padding='max_length')), #, max_length=100)), 
         batched=True
        )

    # print(tokenized_datasets)

    if (n != 0):
        # print(tokenized_datasets["train"])

        tokenized_train_panda = tokenized_datasets["train"].to_pandas()

        # print("orig:")
        # print(tokenized_train_panda["label"].value_counts())
        # print(tokenized_train_panda.columns)
        # torched = torch.cat(tokenized_train_panda.drop(["label", "tweet"], axis=1))
        # print(torched)

        #sampling_strategy={0: n, 1:neut_count, 2: total-n}
        tokenized_train_panda = pandas.concat([
            tokenized_train_panda[tokenized_train_panda["label"] == 0].sample(n      , replace=(n>neg_count)),
            tokenized_train_panda[tokenized_train_panda["label"] == 1].sample(neut_count),
            tokenized_train_panda[tokenized_train_panda["label"] == 2].sample(total-n, replace=((total-n)>pos_count))
        ])
        # print("resampled:")
        # print(tokenized_train_panda["label"].value_counts())
        tokenized_datasets["train"] = Dataset.from_pandas(tokenized_train_panda, preserve_index=False)

        # X = np.array([
        #         tokenized_train_panda[col_name]
        #         for col_name in tokenized_train_panda.drop(["label", "tweet"], axis=1).columns
        #     ]).T
        # Y = tokenized_train_panda["label"]

        # smote = SMOTE() # sampling_strategy={0: n, 1:neut_count, 2: total-n})
        # # print(tokenized_train_panda.drop(["label", "tweet"], axis=1).applymap(len).value_counts())
        # # print(tokenized_train_panda["label"].value_counts())
        # # np = numpy.array(tokenized_train_panda.drop("tweet", axis=1))
        # # smoted_train = smote.fit_resample(np[:,1:3].astype('object'), np[:,0].astype('int'))
        # smoted_train = smote.fit_resample(X,Y)
        # tokenized_datasets = Dataset.from_pandas(smoted_train)
        
        # print(tokenized_datasets["train"])
        # print(tokenized_datasets["train"]["label"].to_pandas().value_counts())

    print("RATIO:")
    print(tokenized_datasets["train"].to_pandas()["label"].value_counts())
    

    # data collector
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ## Trainer

    # FIXME change loss function?
    # training args
    training_args = TrainingArguments(
    output_dir="finetuning",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    learning_rate=6e-6,
    evaluation_strategy="epoch"
    )


    # print(tokenized_datasets)
    # print(len(tokenized_datasets["train"].filter(lambda row: row['label']==0)))
    # print(len(tokenized_datasets["train"].filter(lambda row: row['label']!=1)))

    # initialize trainer 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    ## Train and Eval
    trainer.train()
    #trainer.evaluate()

# %%



