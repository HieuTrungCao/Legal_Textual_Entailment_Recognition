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
from utils import compute_metrics, get_data_frame


def fine_tune(args):

    wandb.login()
    run = wandb.init(
        project=args.project_name,
        tags=["Finetune"]
    )
    
    train_data = get_data_frame("./data/datasets/VLSP/train.json", "./data/VLSP2023-LTER-Data/legal_passages.json")
    test_data = get_data_frame("./data/datasets/VLSP/test.json", "./data/VLSP2023-LTER-Data/legal_passages.json")

    if args.extra_data:
        train_data = pd.concat([train_data, get_data_frame("./data/datasets/KSE/question.json", "./data/ALQAC 2021/law.json")])
    
    if args.data_aug:
        train_data = pd.concat([train_data, get_data_frame("./data/datasets/aug/kse.json", "./data/ALQAC 2021/law.json")])
        train_data = pd.concat([train_data, get_data_frame("./data/datasets/aug/vlsp.json", "./data/VLSP2023-LTER-Data/legal_passages.json")])


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
        compute_metrics=compute_metrics,
    )
    
    #fine-tune
    trainer.train()

    wandb.finish()
    
    print(trainer.state.log_history)

    preds, labels, metrics = trainer.predict(test_dataset)

    acc = compute_metrics((preds, labels))
    print("test accuracy: ", acc)

if __name__ == "__main__":
    parsers = argparse.ArgumentParser()
    parsers.add_argument("--model_name", type=str, help="Enter model name")
    parsers.add_argument("--output", default="output", type=str, help="Enter ouput dir to save model")
    parsers.add_argument("--log", default="log", type=str, help="Enter log dir to save log")
    parsers.add_argument("--epoch", default=3, type=int, help="Enter num epoch")
    parsers.add_argument("--lr", default=6.25e-5, type=float, help="Enter learning rate")
    parsers.add_argument("--max_length", default=256, type=int, help="Enter max_length to padding")
    parsers.add_argument("--batch_size", default=32, type=int, help="Enter batch size")
    parsers.add_argument("--num_labels", default=2, type=int, help="Enter num labels")
    parsers.add_argument("--weight_decay", default=0.002, type=float, help="Enter weight decay")
    parsers.add_argument("--extra_data", default=False, type=bool, help="Do you want to add more data?")
    parsers.add_argument("--add_bm25", default=False, type=bool, help="Do you want add BM25 score to train")
    parsers.add_argument("--project_name", default="VLSP", type=str, help="project name on wandb")
    parsers.add_argument("--data_aug", default=False, type=bool, help="Do you want to segment text for training?")

    args = parsers.parse_args()
    fine_tune(args)
