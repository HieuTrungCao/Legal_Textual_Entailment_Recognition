import argparse
import pandas as pd
import torch
import json
import numpy as np


from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoModel
)

from datasets import MyDataset
from utils import preprocess, f2_score

def get_result(pred, example_id, id_legal, ground_truth):
    preds = torch.argmax(torch.tensor(preds), dim = 1)
    preds = preds.tolist()

    predict = {}
    for p, e_id, l_id in zip(pred, example_id, id_legal):
        if p == 1:
            if l_id in ground_truth[e_id]:
                if e_id not in predict:
                    predict[e_id] = []
                predict.append(l_id)
    return predict

if __name__ == "__main__":
    parsers = argparse.ArgumentParser()
    parsers.add_argument("--model_name", type=str, help="Enter model name")
    parsers.add_argument("--model_cls", type="nguyenthanhasia/VNBertLaw", help="Enter model name")
    parsers.add_argument("--path_data", default="./data/zac2021-ltr-data/my_test.csv", type=str, help="Enter path data")
    parsers.add_argument("--path_raw_data", default="./data/zac2021-ltr-data/my_test.json", type=str, help="Enter path data")
    parsers.add_argument("--path_raw_data_train", default="./data/zac2021-ltr-data/train_question_answer.json", type=str, help="Enter path data")
    parsers.add_argument("--max_length", default=256, type=int, help="Enter max length to tokenizer")
    parsers.add_argument("--add_bm25", default=False, type=bool, help="Do you want add BM25 score to train")
    parsers.add_argument("--is_segment", default=False, type=bool, help="Do you want to segment text for training?")
    parsers.add_argument("--path_train_csv", default="./data/zac2021-ltr-data/train_question_answer.csv", type=str)

    args = parsers.parse_args()


    ground_truth = {}
    query_test = {}
    raw_data = json.load(open(args.path_raw_data, encoding='utf-8'))["items"]
    for q in raw_data:
        relevents = []
        query_test[q["question_id"]] = q["question"]
        for r in q["relevant_articles"]:
            x = r["law_id"] + "_" + r["article_id"]
            relevents.append(x)
        ground_truth[q["question_id"]] = relevents


    test_data = pd.read_csv(args.path_data)
    test_data = preprocess(test_data)


    """
    GET miss query
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    test_datasets = MyDataset(test_data, tokenizer, max_length=args.max_length, add_bm25=args.add_bm25)

    list_model = args.model_name.split("_")
    results = []
    for m in list_model:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

        trainer = Trainer(
            model
        )

        preds, labels, metrics = trainer.predict(test_datasets, test_data["example_id"], test_data["id_legal"], ground_truth)

        result = get_result(preds, test_data["example_id"], test_data["id_legal"], ground_truth)

        results += result.keys()
    
    query = list(set(results))
    
    m_query = []
    for e in ground_truth.keys():
        if e not in query:
            m_query.append(e)

    """
    Get data neighborhood
    """
    device = "gpu" if torch.cuda.is_available() else "cpu"

    model = AutoModel.from_pretrained(args.model_cls).to(device)
    data_train = json.load(open(args.path_raw_data_train, encoding="utf-8"))["items"]
    query_train = {}

    for q in data_train:
        token = tokenizer(q["question"], padding="max_length", truncation=True, max_length=args.max_length)
        data = {
            "input_ids": torch.tensor(token["input_ids"]).to(device),
            "attention_mask": torch.tensor(token["attention_mask"]).to(device),
            "token_type_ids": torch.tensor(token["token_type_ids"]).to(device),
        }
        query_train[q["quesion_id"]] = model(**data)["last_hidden_state"][0,0,:].numpy()


    miss_query = {}
    for m_q in m_query:
        token = tokenizer(query_test[m_q], padding="max_length", truncation=True, max_length=args.max_length)
        data = {
            "input_ids": torch.tensor(token["input_ids"]).to(device),
            "attention_mask": torch.tensor(token["attention_mask"]).to(device),
            "token_type_ids": torch.tensor(token["token_type_ids"]).to(device),
        }
        miss_query[m_q] = model(**data)["last_hidden_state"][0,0,:].numpy()

    
    get_id = []

    for m_q in miss_query.values():
        score = []
        for q_t in query_train.values():
            cosine = np.dot(m_q, q_t)/(np.linalg.norm(m_q)*np.linalg.norm(q_t))
            score.append(cosine)
        score = np.array(score)
        score = np.flip(np.argsort(score))[:10]

        for s in score:
            get_id.append(query_train.keys()[s])
    
    get_id = list(set(get_id))

    train_data = pd.read_csv(args.path_train_csv)

    new_df = pd.DataFrame({
        "example_id": [],
        "id_legal": [],
        "statement": [],
        "legal_passage": [],
        "label": [],
        "score": []
    })

    for i in range(len(train_data.index)):
        if train_data.iloc[i]["example_id"] in get_id:
            new_df.loc[len(new_df)] = [train_data.iloc[i]["example_id"],
                                        train_data.iloc[i]["id_legal"],
                                        train_data.iloc[i]["statement"],
                                        train_data.iloc[i]["legal_passage"],
                                        train_data.iloc[i]["label"],
                                        train_data.iloc[i]["score"],
                                        ]
            
    new_df.to_csv("./data/zac2021-ltr-data/new_train.csv", index=False)


    


    
    

