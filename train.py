import argparse
import os
import pandas as pd
import wandb

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from datasets import Training_Dataset
from utils import compute_metrics

def train(args):
    
    wandb.login()
    run = wandb.init(
        project=args.project_name,
        notes=args.notes,
        tags=["Training"]
    )

    train_df = pd.read_csv(os.path.join(args.path_data, "data_aug_train.csv"))
    test_df = pd.read_csv(os.path.join(args.path_data, "data_aug_test.csv"))

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_datasets = Training_Dataset(train_df, tokenizer, max_length=args.max_length)
    test_datasets = Training_Dataset(test_df, tokenizer, max_length=args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(args.model)

    training_args = TrainingArguments(
        args.output,
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epoch,
        logging_strategy="epoch",
        report_to="wandb"
    )

    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=train_datasets,
        valid_dataset=test_datasets,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    #fine-tune
    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--model", type=str, help="Enter model on huggingface to load")
    parse.add_argument("--path_data", default="./data/datasets/aug", type=str)
    parse.add_argument("--lr", default=1e-3, type=float)
    parse.add_argument("--batch_size", default=128, type=int)
    parse.add_argument("--epoch", default=10, type=int)
    parse.add_argument("--output", default="output")
    parse.add_argument("--max_length", default=256, type=int)
    parse.add_argument("--project_name", default="VLSP", type=str, help="project name on wandb")

    args = parse.parse_args()

    train(args)