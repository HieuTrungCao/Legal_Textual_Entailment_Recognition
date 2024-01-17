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
from utils import compute_metrics, get_data_frame

if __name__ == "__main__":
    parsers = argparse.ArgumentParser()
    parsers.add_argument("--model_name", type=str, help="Enter model name")
    parsers.add_argument("--path_legal", default="data/VLSP2023-LTER-Data/legal_passages.json", type=str, help="Enter path data")
    parsers.add_argument("--path_data", default="data/VLSP2023-LTER-Data/test.json", type=str, help="Enter path data")
    parsers.add_argument("--max_length", default=256, type=int, help="Enter max length to tokenizer")
    parsers.add_argument("--add_bm25", default=False, type=bool, help="Do you want add BM25 score to train")
    parsers.add_argument("--is_segment", default=False, type=bool, help="Do you want to segment text for training?")

    args = parsers.parse_args()

    test_data = get_data_frame(args.path_data, args.path_legal)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    test_datasets = MyDataset(test_data, tokenizer, max_length=args.max_length, add_bm25=args.add_bm25)


    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

    trainer = Trainer(
        model
    )

    preds, labels, metrics = trainer.predict(test_datasets)

    if labels is not None:
        acc = compute_metrics((preds, labels))
        print("test accuracy: ", acc)

    example_ids = test_datasets.get_example_id()

    preds = torch.argmax(torch.tensor(preds), dim = 1)

    result = []
    
    for id, pred in zip(example_ids, preds):
        r = {
            "example_id": id,
            "label": test_datasets.id2label[pred.item()],
        }
        result.append(r)
    
    json.dump(result, open("result/result_prediction.json", "wt", encoding='utf8'), ensure_ascii=False, indent=2)