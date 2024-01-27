import pandas as pd


def preprocess(data):
    
    id2label = {
        0: "No",
        1: "Yes"
    }

    example_id = []
    statement = []
    legal_passage = []
    label = []
    score = []
    id_legal = []

    max_length = 250
    stride = 100

    for i in range(data.shape[0]):
        text = data.iloc[i]["legal_passage"].split(" ")
        if len(text) > max_length:
            for i in range(len(text)):
                start = i * stride
                end = start + max_length
                if end > len(text):
                    end = len(text)
                t = " ".join(text[start: end])
                example_id.append(data.iloc[i]["example_id"])
                statement.append(data.iloc[i]["statement"])
                legal_passage.append(t)
                label.append(id2label[data.iloc[i]["label"]])
                score.append(data.iloc[i]["score"])
                id_legal.append(data.iloc[i]["id_legal"]) 

                if end == len(text):
                    break;       
        else:
            example_id.append(data.iloc[i]["example_id"])
            statement.append(data.iloc[i]["statement"])
            legal_passage.append(data.iloc[i]["legal_passage"])
            label.append(id2label[data.iloc[i]["label"]])
            score.append(data.iloc[i]["score"])
            id_legal.append(data.iloc[i]["id_legal"])

    return pd.DataFrame({
        "example_id": example_id,
        "id_legal": id_legal,
        "statement": statement,
        "legal_passage": legal_passage,
        "label": label,
        "score": score
    })

