import argparse
import pandas as pd
import torch
import json

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from datasets import MyDataset
from utils import preprocess, f2_score

if __name__ == "__main__":
    parsers = argparse.ArgumentParser()
    parsers.add_argument("--model_name", type=str, help="Enter model name")
    parsers.add_argument("--path_data", default="data/zac2021-ltr-data/my_test.csv", type=str, help="Enter path data")
    parsers.add_argument("--path_raw_data", default="data/zac2021-ltr-data/my_test.json", type=str, help="Enter path data")
    parsers.add_argument("--max_length", default=256, type=int, help="Enter max length to tokenizer")
    parsers.add_argument("--add_bm25", default=False, type=bool, help="Do you want add BM25 score to train")
    parsers.add_argument("--is_segment", default=False, type=bool, help="Do you want to segment text for training?")

    args = parsers.parse_args()


    ground_truth = {}

    raw_data = json.load(open(args.path_raw_data, encoding='utf-8'))["items"]
    for q in raw_data:
        relevents = []
        for r in q["relevant_articles"]:
            x = r["law_id"] + "_" + r["article_id"]
            relevents.append(x)
        ground_truth[q["question_id"]] = relevents


    test_data = pd.read_csv(args.path_data)
    test_data = preprocess(test_data)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    test_datasets = MyDataset(test_data, tokenizer, max_length=args.max_length, add_bm25=args.add_bm25)


    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

    trainer = Trainer(
        model
    )

    preds, labels, metrics = trainer.predict(test_datasets)

    if labels is not None:
        f2, p, r = f2_score((preds, labels),test_data["example_id"], test_data["id_legal"], ground_truth)
        print("F2: ", f2, end="\t")
        print("P: ", p, end="\t")
        print("R: ", r)