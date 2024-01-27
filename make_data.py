import json
import os
import argparse
import numpy as np
import pandas as pd

from utils import (
    clean_text,
    word_segment,
    remove_stopword,
    normalize_text,
    BM25,
    list_stopwords    
)


def get_bm25(legal):
    new_legal = {}

    id_law = []

    for l in legal:
        k_l = l["law_id"]
        for a in l["articles"]:
            k = k_l + "_" + a["article_id"]
            text = a["text"]
            text = text.lower()
            text = text.replace("\n\n", " ")
            text = text.replace("\n", " ")
            text = clean_text(text)
            text = word_segment(text)
            text = remove_stopword(normalize_text(text))
            id_law.append(k)
            new_legal[k] = text

    texts = [
		[word for word in document.lower().split() if word not in list_stopwords]
		for document in new_legal.values()
	]
    bm25 = BM25()

    print("Training BM25...........")
    bm25.fit(texts)
    print("Done!!!!!!!!!!!!!!!!!!!")

    return bm25, new_legal, id_law    

def make_data(args, filename, bm25, legal, id_law):
    

    data_train = json.load(open(os.path.join(args.path_data, filename), encoding="utf-8"))
    
    example_id = []

    statement = []
    legal_passage = []
    label = []
    score = []
    id_legal = []

    for d in data_train["items"]:
        e_id = d["question_id"]
        s = d["question"]
        query = clean_text(s)
        query = word_segment(query)
        query = remove_stopword(normalize_text(query))
        query = query.split()

        scores = bm25.search(query)
        scores = np.array(scores)
        
        idx = np.argsort(scores)
        idx = list(np.flip(idx)[:150])

        label_relevants = []
        for r in d["relevant_articles"]:
            l = r["law_id"] + "_" + r["article_id"]
            label_relevants.append(l)

            i = id_law.index(l)
            if i not in idx:
                idx.append(i)
        

        for i in idx:
            example_id.append(e_id)
            statement.append(d["question"])
            legal_passage.append(legal[id_law[i]])
            
            if id_law[i] in label_relevants:
                label.append(1)
            else:
                label.append(0)
            score.append(scores[i])
            id_legal.append(id_law[i])

    df = pd.DataFrame({
        "example_id": example_id,
        "id_legal": id_legal,
        "statement": statement,
        "legal_passage": legal_passage,
        "label": label,
        "score": score
    })

    filename = filename[: -5] + ".csv"
    df.to_csv(os.path.join(args.path_data, filename), index=False)










if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--path_data", default= 'data\zac2021-ltr-data', type=str)

    args = parse.parse_args()

    legal = json.load(open(os.path.join(args.path_data, "legal_corpus.json"), encoding="utf-8"))

    bm25, legal, id_law = get_bm25(legal)

    make_data(args, "my_test.json", bm25, legal, id_law)

    make_data(args, "train_question_answer.json", bm25, legal, id_law)