import torch


def compute_metrics(eval_pred):
  preds, labels = eval_pred

  preds = torch.argmax(torch.tensor(preds), dim = 1)
  labels = torch.tensor(labels)
  acc = (labels == preds).float().mean().item()
  return {
      "acc": acc
  }