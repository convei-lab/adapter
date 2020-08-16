# REQUIRED: DOWNLOAD SST FROME GLUE $ python transformers/utils/download_glue_data.py --tasks MNLI

# Let's run what is happening here..
# 00:10:55 up 41 days,  6:43,  8 users,  load average: 1.17, 1.15, 1.11
# USER     TTY      FROM             LOGIN@   IDLE   JCPU   PCPU WHAT
# yen      pts/8    :pts/23:S.0      22:42    1:25m  1:21m  1:21m python train_ratio_code.py --learning_rate=1e-5 --warmup_count 200 --data_dir data/dstc7_from_original/ --a
# yen      pts/11   :pts/24:S.0      22:44    1:25m  0.22s  0.01s screen -d -r gpu1
# bwoo     pts/15   165.132.145.133  Thu17    3days  0.08s  0.08s -bash
# yen      pts/14   165.132.145.76   Wed20    2days  5:46   5:46  watch -n1 nvidia-smi
# yen      pts/22   125.187.44.47    22:40    1:29m 32.76s 32.67s /home/yen/anaconda3/envs/tf/bin/python /home/yen/anaconda3/envs/tf/bin/tensorboard --logdir=runs --bind_all
# yen      pts/23   :pts/11:S.0      22:42    1:25m  0.30s  0.09s screen -d -r gpu2
# yen      pts/24   125.187.44.47    22:44    1:25m  0.07s  0.00s screen -d -r gpu0
# shinwoo  :1       :1               Fri10   ?xdm?  22:21m  0.01s /usr/lib/gdm3/gdm-x-session --run-script env GNOME_SHELL_SESSION_MODE=ubuntu gnome-session --session=ubuntu
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

model_name = "roberta-large"
experiment_name = "/mnli_pilot"
logging.basicConfig(level=logging.INFO)

data_args = GlueDataTrainingArguments(
    task_name="mnli", data_dir="../data/glue_data/MNLI")
training_args = TrainingArguments(
    logging_first_step=True,
    logging_steps=1000,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    save_steps=1000,
    evaluate_during_training=True,
    output_dir="../models"+experiment_name,
    overwrite_output_dir=False,
    do_train=True,
    do_eval=True,
    do_predict=True,
    learning_rate=0.00003,
    num_train_epochs=3,
)
set_seed(training_args.seed)
num_labels = glue_tasks_num_labels[data_args.task_name]


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithHeads.from_pretrained(model_name, num_labels=4)
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
print('Experiment name:', experiment_name)
print('Total time in sec', time.time()-starttime)

adpater_path = Path("../adapters"+experiment_name)
adpater_path.mkdir(parents=True, exist_ok=True)
model.save_all_adapters(adpater_path)
