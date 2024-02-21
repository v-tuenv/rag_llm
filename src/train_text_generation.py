
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os

"""
colab issues
https://github.com/huggingface/datasets/issues/5923
# Restart session-runtime and run it again

"""
from argparse import ArgumentParser

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    PhiForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    GenerationConfig
)
from tqdm import tqdm
import torch
import time
import json
import pandas as pd
import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from dataclasses import dataclass
from typing import Optional, Union
from peft import LoraConfig, get_peft_model


@dataclass
class DataCollatorForLLM:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        flattened_features = features
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch['labels'] = batch['input_ids'].clone()
        return batch
    
def load_dataset(args):
    PREFIX= '''You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Context: {context}
    Question: {question}
    Answer: {ans}
    '''
    prompt_template = lambda context, query, response: PREFIX.format(
        context=context,
        question=query,
        ans=response
    )
    dataset = pd.read_csv(args.csv_path)
    dataset = Dataset.from_pandas(pd.DataFrame(dataset))
    dataset = dataset.map(
        lambda x: {
            'text_train': prompt_template(x['content'], x['question'], x['answer'])
        }
    )
    return dataset


def tokenize(sample):
    result = tokenizer(
        sample['text_train'],
        truncation=True,
        max_length=512,
    )
    result["labels"] = result["input_ids"].copy()

    return result



parser = ArgumentParser(
    prog="""
        This is script finetune model llm
        python3 train_text_generation.py --csv-path data.csv --output-folder ./log_run_20_2_2024 --save-steps 50 --logging-steps=25
        note: data.csv has 3 columns: 'content', 'question' and 'answer'
    """
)
parser.add_argument(
    "--csv-path",
    type=str,
    help="file pandas dataset",
    required=True
)
parser.add_argument(
    "--output-folder",
    type=str,
    help="Folder save output",
    required=True
)

parser.add_argument(
    "--model-name",
    type=str,
    help="name of model transformer to fintune",
    default="microsoft/phi-2"
)
parser.add_argument("--lora-r", type=int, default=64, help="rank of lora finetune")

parser.add_argument("--lora-alpha", type=float, default=128, help="scale of lora finetune")


parser.add_argument(
    "--num-epochs",
    type=int, default=2,
    help='number of epoch training'
)
parser.add_argument("--learning-rate", type=float, default=1e-4, help='learning-rate training')
parser.add_argument("--logging-steps", type=int, default=25, help='number of step to logging')
parser.add_argument("--save-steps", type=int, default=50, help="saving for eacg save-steps")
parser.add_argument("--per-device-train-batch-size", type=int, default=2)
parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
args = parser.parse_args()

# Load tokenizer 
model_name = args.model_name
compute_dtype = getattr(torch, "bfloat16")
device_map = {"": 0}
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,padding_side="right",add_eos_token=True,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

print("load tokenizer done")

# load model 

original_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map=device_map,
                                                torch_dtype=compute_dtype,
                                                trust_remote_code=True)

print("load model done")

lora_dropout = 0.0
lora_r = args.lora_r
lora_alpha = args.lora_alpha
print(f"Lora-r={lora_r}\nLora-alpha={lora_alpha}")
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    target_modules=[
        'q_proj',
#         'k_proj',
        'v_proj',
        'dense'
    ],
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(original_model, peft_config)
model.print_trainable_parameters()
#---------------------------------------------------LOAD DATASETS----------------------------------------
dataset = load_dataset(args)
tokenize_dataset = dataset.map(tokenize, remove_columns=['text_train', 'content', 'question', 'answer'])



#----------------------------------------------------START TRAINING--------------------------------------
dirout=args.output_folder
output_dir = os.path.join(dirout, 'log_training')
per_device_train_batch_size = args.per_device_train_batch_size
gradient_accumulation_steps = args.gradient_accumulation_steps
optim = "adamw_hf"
save_steps = args.save_steps
logging_steps = args.logging_steps
learning_rate = args.learning_rate
max_grad_norm = 2
max_steps = -1
num_epochs = 2
warmup_ratio = 0.2 # 20%  total step warm-up
lr_scheduler_type='constant'
tokenizer.pad_token = tokenizer.eos_token

INFO_SAVE=dict(
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim = optim,
    save_steps = save_steps,
    logging_steps = logging_steps,
    learning_rate = learning_rate,
    max_grad_norm = max_grad_norm,
    max_steps = max_steps,
    num_epochs = num_epochs,
    warmup_ratio = warmup_ratio, # 20%  total step warm-up
    lr_scheduler_type='constant',
    output_dir=output_dir,
    )

trainer = Trainer(
    model=model,
    train_dataset=tokenize_dataset,
    args=TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=False,
        bf16=False,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        num_train_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        report_to='none',
        push_to_hub=False,
        do_eval=False,
        save_only_model=True,
        save_total_limit=1

    ),
    data_collator=DataCollatorForLLM(tokenizer),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer_output=trainer.train()
INFO_SAVE['trainer_metrics'] = trainer_output.metrics
print(trainer_output)

print("TRAIN DONE")
#_-----------------------------------------------DONE--------------------------------------#

model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
model_to_save.save_pretrained( os.path.join(dirout, 'model_lora'))
model_to_save.config.use_cache = True  # silence the warnings. Please re-enable for inference!
tokenizer.save_pretrained(os.path.join(dirout, 'model_lora'))
#------------------------------------------------SAVE MODEL DONE----------------------------#