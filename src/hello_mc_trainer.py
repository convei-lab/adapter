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
from transformers import BertConfig, BertTokenizer, BertForMultipleChoice, EvalPrediction
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers import AdapterType, AdapterConfig

#sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('/home/convei_intern1/kadapter/transformers/examples/multiple_choice')
from utils_multiple_choice import MultipleChoiceDataset, Split, processors
#from ...transformers.examples.multiple_choice.utils_multiple_choice import MultipleChoiceDataset, Split, processors

# OPTIONAL: SPEICIFY GPU TO USE $ export CUDA_VISIBLE_DEVICES=0,1,2

logging.basicConfig(level=logging.INFO)

model_name = "bert-base-uncased"
#model_name = "roberta-large"
model_name_or_path = model_name
tokenizer_name = None
experiment_name = "/mnli_pilot"
task_name="commonsenseqa", 
data_dir="/home/convei_intern1/kadapter/data/commonsenseqa"
#cache_dir=data_dir
config_name = None
num_labels = 5
max_seq_length = 80
overwrite_cache = False

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

config = BertConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    finetuning_task=task_name,
    #cache_dir=cache_dir,
)

tokenizer = BertTokenizer.from_pretrained(
    model_name_or_path
    #cache_dir=cache_dir,
)

model = BertForMultipleChoice.from_pretrained(
    model_name_or_path,
    #from_tf=bool(".ckpt" in model_name_or_path),
    config=config
)

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelWithHeads.from_pretrained(model_name, num_labels=4)
# model.add_adapter("sst-2", AdapterType.text_task)
#model = AutoModel.from_pretrained(model_name)
config = AdapterConfig.load("pfeiffer")
model.load_adapter("comsense/csqa@ukp", "text_task", config=config)
model.add_apdapter("csqa", AdapterType.text_task)
model.train_adapter(["csqa"])
model.set_active_adapters([["csqa"]])

# Get datasets
train_dataset = (
    MultipleChoiceDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        task=task_name,
        max_seq_length=max_seq_length,
        overwrite_cache=overwrite_cache,
        mode=Split.train,
    )
    if training_args.do_train
    else None
)
eval_dataset = (
    MultipleChoiceDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        task=task_name,
        max_seq_length=max_seq_length,
        overwrite_cache=overwrite_cache,
        mode=Split.dev,
    )
    if training_args.do_eval
    else None
)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": simple_accuracy(preds, p.label_ids)}


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
