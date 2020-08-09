# REQUIRED: DOWNLOAD SST FROME GLUE $ python transformers/utils/download_glue_data.py --tasks SST
from pathlib import Path
import time
import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

import torch
from transformers import AutoTokenizer, EvalPrediction, GlueDataset, GlueDataTrainingArguments, AutoModelWithHeads, AdapterType
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_tasks_num_labels,
    set_seed,
)

import gc
gc.collect()
torch.cuda.empty_cache()
# OPTIONAL: SPEICIFY GPU TO USE $ export CUDA_VISIBLE_DEVICES=0,1,2

model_name = "roberta-base"
experiment_name = "/reload"
logging.basicConfig(level=logging.INFO)

data_args = GlueDataTrainingArguments(
    task_name="sst-2", data_dir="../data/glue_data/SST-2")
training_args = TrainingArguments(
    logging_first_step=True,
    logging_steps=1000,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    save_steps=1000,
    evaluate_during_training=True,
    output_dir="../models"+experiment_name,
    overwrite_output_dir=False,
    do_train=True,
    do_eval=True,
    do_predict=True,
    learning_rate=0.0001,
    num_train_epochs=3,
)
set_seed(training_args.seed)
num_labels = glue_tasks_num_labels[data_args.task_name]


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithHeads.from_pretrained(model_name)
model.add_adapter("sst-2", AdapterType.text_task)
model.train_adapter(["sst-2"])

model.add_classification_head("sst-2", num_labels=num_labels)
model.set_active_adapters([["sst-2"]])

train_dataset = GlueDataset(data_args, tokenizer=tokenizer)
eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")


def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return glue_compute_metrics(data_args.task_name, preds, p.label_ids)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)


starttime = time.time()
trainer.train()
trainer.evaluate()
print('Total time in sec', time.time()-starttime)

adpater_path = Path("../adapters"+experiment_name)
adpater_path.mkdir(parents=True, exist_ok=True)
model.save_all_adapters(adpater_path)
