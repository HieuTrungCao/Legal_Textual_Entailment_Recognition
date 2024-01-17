import torch
import json
import numpy as np
import pandas as pd

from .bm25 import get_scores

def compute_metrics(eval_pred):
  preds, labels = eval_pred

  preds = torch.argmax(torch.tensor(preds), dim = 1)
  labels = torch.tensor(labels)
  acc = (labels == preds).float().mean().item()
  return {
      "acc": acc
  }

def split_legal_passage(legal_passages):
  res = []
  temp = legal_passages.split('\n')
  for s in temp:
    if len(s) == 0:
      continue
    
    s = s.lower()
    if s[0].isdigit():
      s = s[s.find(" ") + 1:]

    if s[0].isalpha() and s[1] == ")":
      s = s[s.find(" ") + 1:]

    if s[-1] == ":":
      s = s[: -1]

    if s[-1] == ".":
      s = s[: -1]

    if s[-1] == ";":
      s = s[: -1]

    res.append(s)

  return res

def get_article(a, law):
  for d in law:
    if d["id"] == a["law_id"]:
      for ar in d["articles"]:
        if a["article_id"] == ar["id"]:
          return ar["text"]
  
  return ""

def get_legal_passage(statement, legal_passages, law):

  sentences = []

  for l in legal_passages:
    text = get_article(l, law)
    sens = split_legal_passage(text)
    sentences.append(sens)

  scores = get_scores(statement, sentences)
  id_max = np.argmax(scores)
  
  return sentences[id_max], scores[id_max]


def get_data_frame(path_data, path_law):
  law = json.load(open(path_law, encoding="utf-8"))
  data = json.load(open(path_data, encoding="utf-8"))

  example_id = []
  statement = []
  legal_passage = []
  score = []
  label = []

  for d in data:
    example_id.append(d["example_id"])
    statement.append(d["statement"].lower())
    if "label" in d.keys():
      label.append(d["label"])

    l, s = get_legal_passage(d["statement"].lower(), d["legal_passages"], law) 
    legal_passage.append(l)
    score.append(s)

  new_data = {
    "example_id": example_id,
    "statement": statement,
    "legal_passage": legal_passage,
    "score": score
  }

  if "label" in data[0].keys():
    new_data["label"] = label 
  return pd.DataFrame(new_data)