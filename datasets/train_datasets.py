import torch

from torch.utils.data import Dataset

class Training_Dataset(Dataset):
    def __init__(self, df, tokenizer, max_length) -> None:
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        text = self.df.iloc[index]["text"]
        token = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length)
        data = {
            "input_ids": torch.tensor(token["input_ids"]),
            "attention_mask": torch.tensor(token["attention_mask"]),
            "token_type_ids": torch.tensor(token["token_type_ids"]),
        }

        data['labels'] = int(self.df.iloc[index]["label"])

    def __len__(self):
        return self.df.shape[0]
