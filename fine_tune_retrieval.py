import argparse
import pandas as pd
import torch.nn as nn
import json
import torch.optim as optim
import time 
import torch
import wandb
import os

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from torch.utils.data import DataLoader

# from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType

from datasets import MyDataset
from utils import preprocess


def fine_tune(args):

    wandb.login()
    run = wandb.init(
        project=args.project_name,
        tags=args.tags.split("-")
    )
    
    train_data = pd.read_csv(os.path.join(args.path_data, "train_question_answer.csv"))
    test_data = pd.read_csv(os.path.join(args.path_data, "my_test.csv"))

    train_data = preprocess(train_data)
    test_data = preprocess(test_data)

    print("Training sample: ", len(train_data.index))
    print("Test sample: ", len(test_data.index))

    #load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)  

    train_dataset = MyDataset(train_data, tokenizer, max_length=args.max_length, add_bm25=args.add_bm25)
    test_dataset = MyDataset(test_data, tokenizer, max_length=args.max_length, add_bm25=args.add_bm25)
    
    #load model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)

    #setup training arguments
    training_args = TrainingArguments(
        args.output,
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epoch,
        weight_decay= args.weight_decay,
        logging_dir=args.log
    )

    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    #fine-tune
    if args.resume:
        print("Resume from checkpoint")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    wandb.finish()
    
    print(trainer.state.log_history)

if __name__ == "__main__":
    parsers = argparse.ArgumentParser()
    parsers.add_argument("--model_name", type=str, help="Enter model name")
    parsers.add_argument("--resume", type=bool, default=False)
    parsers.add_argument("--path_data", default="data\zac2021-ltr-data", type=str, help="Enter model name")
    parsers.add_argument("--output", default="output", type=str, help="Enter ouput dir to save model")
    parsers.add_argument("--log", default="log", type=str, help="Enter log dir to save log")
    parsers.add_argument("--epoch", default=3, type=int, help="Enter num epoch")
    parsers.add_argument("--lr", default=6.25e-5, type=float, help="Enter learning rate")
    parsers.add_argument("--max_length", default=256, type=int, help="Enter max_length to padding")
    parsers.add_argument("--batch_size", default=32, type=int, help="Enter batch size")
    parsers.add_argument("--num_labels", default=2, type=int, help="Enter num labels")
    parsers.add_argument("--weight_decay", default=0.002, type=float, help="Enter weight decay")
    parsers.add_argument("--add_bm25", default=False, type=bool, help="Do you want add BM25 score to train")
    parsers.add_argument("--project_name", default="VLSP", type=str, help="project name on wandb")
    parsers.add_argument("--tags", default="Finetune_Retrieval", type=str, help="Do you want to segment text for training?")

    args = parsers.parse_args()
    fine_tune(args)
