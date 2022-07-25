import pandas as pd
import torch
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = 3

output_base_path = f"../models/mbert/lowerCase/size{window_size}/"

dfnew = pd.read_csv(f"../20220321_Hansard_DB_MP_only_window_{window_size}.csv")[['text', 'Labels_Final']]

dfnew['text'] = dfnew['text'].astype(str)
dfnew.columns = ['text', 'labels']
dfnew.drop(dfnew[(dfnew.labels == 'N') | (dfnew.labels == 'U')].index, inplace=True)
dfnew['text'] = dfnew['text'].astype(str).str.lower()
dfnew['labels'] = dfnew['labels'].replace({"B": 0, "M": 1, "P": 2})
dfnew['labels'] = dfnew['labels'].astype(int)

from sklearn.model_selection import train_test_split

train, test = train_test_split(dfnew, test_size=0.3, random_state=100, stratify=dfnew['labels'])

from datasets import Dataset, DatasetDict

dataset = DatasetDict({"train":Dataset.from_pandas(train, preserve_index=False), "test":Dataset.from_pandas(test, preserve_index=False)})

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# dfnew['tokenized_length'] = dfnew['text'].apply(lambda x: len(tokenizer.encode(x, padding="longest", truncation=True))).astype(int)
# max_token_length = dfnew['tokenized_length'].max()
# print(f'\n\nThe max length of the tokenized sentence is {max_token_length}\n\n') # 31
max_token_length = 31

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", max_length=max_token_length, truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=3)

from transformers import TrainingArguments

import numpy as np
from datasets import load_metric

# https://huggingface.co/metrics
metric = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="micro")

from transformers import TrainingArguments, Trainer

# construct name of output directory with time
import datetime

current_datetime = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# add callback
from transformers import EarlyStoppingCallback

training_args = TrainingArguments(  num_train_epochs=4,
                                    per_device_train_batch_size=1024, # Approximately 23 GB 
                                    overwrite_output_dir=True,
                                    learning_rate=8e-6,
                                    output_dir=output_base_path+current_datetime+"/m_bert_hf_out/",
                                    evaluation_strategy="epoch",
                                    optim="adamw_torch",
                                    load_best_model_at_end=True,
                                    save_strategy="epoch",
                                    metric_for_best_model="f1",
                                )


model.to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0005)],
)

torch.cuda.empty_cache()
del dataset, dfnew, max_token_length, train, test, tokenizer, tokenized_datasets, model, training_args
gc.collect()

trainer.train()

trainer.save_model(output_base_path+current_datetime+'/model')
