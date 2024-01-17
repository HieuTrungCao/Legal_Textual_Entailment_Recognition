from typing import Any
from torch.utils.data import Dataset
from pandas import DataFrame

import torch

class MyDataset(Dataset):
    def __init__(self, df, tokenizer, max_length: int, add_bm25: bool):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = {
          "Yes" : 0,
          "No" : 1,
        }
        self.id2label = {
            0: "Yes",
            1: "No"
        }
        self.add_bm25 = add_bm25

        self.df = df        

    def __len__(self):
        return len(self.df.index)

    def get_example_id(self):
        example_id = []

        for i in range(self.__len__()):
            example_id.append(self.df.iloc[i]["example_id"])

        return example_id
    
    def standard_bm25_score(self, score) -> str:
        score = (score - 42) / 6
        score = score * 100
        score = int(score)

        return str(score)
    
    def __getitem__(self, index: Any) -> Any:
        sentence1 = self.df.iloc[index]["statement"]
        sentence2 = self.df.iloc[index]["legal_passage"]
        if self.add_bm25:
            sentence1 = sentence1 + " [SEP] " + self.standard_bm25_score(self.df.iloc[index]["score"])
            
        token = self.tokenizer(sentence1, sentence2, padding="max_length", truncation=True, max_length=self.max_length)
        data = {
            "input_ids": torch.tensor(token["input_ids"]),
            "attention_mask": torch.tensor(token["attention_mask"]),
            "token_type_ids": torch.tensor(token["token_type_ids"]),
        }

        if "label" in self.df.keys():
            data['labels'] = self.label2id[self.df.iloc[index]["label"]]
        
        return data  









