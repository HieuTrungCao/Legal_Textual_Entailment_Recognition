import json
import random
import pandas

data_corpus = {}
id = 1
file_path = 'data/VLSP2023-LTER-Data/legal_passages.json'
data = json.load(open(file_path, 'r', encoding='utf-8'))

def split_random(data, subset_size):
    random.shuffle(data)
    return data[:subset_size], data[subset_size:]

def init_label(data):
    count = 0
    new_data = []
    label = []
    for i in range(len(data)):
        s = data[i].lower()
        l = 1
        if s.find("được")!= -1:
            if(count < len(data) * 0.45):
                if(s.find("không được") != -1):
                    s = s.replace("không được", "được", 1)
                else: s = s.replace("được", "không được", 1)
                count += 1
                l = 0

        elif s.find("phải")!= -1:
            if(count < len(data) * 0.48):
                if(s.find("không phải") != -1):
                    s = s.replace("không phải", "phải", 1)
                else: s = s.replace("phải", "không phải", 1)
                count += 1
                l = 0

        new_data.append(s)
        label.append(l)
    data_df = pandas.DataFrame({
        "text":new_data,
        "label":label
    })
    data_df = data_df.sample(frac = 1)
    return data_df





#Split data follow number
text = []
for d in data:
    id = d["id"]
    for a in d["articles"]:
        sentences = a["text"].split("\n")
        title_a = sentences[0]
        title = ""
        i = 1
        while i < len(sentences):
            s = sentences[i]
            if len(s) == 0:
                i += 1
                continue
            if s[0].isdigit():
                s = s[s.find(" ") + 1:]
                title =""
            if s[-1] == ":":
                title = s
                i = i + 1
                continue
            if s[0].isalpha() and s[1] == ")":
                s = s[s.find(" ") + 1:]
            t = id + ": " + title_a + ": "
            if title != "": t = t + title + ": " + s
            else: t = t + s
            text.append(t)
            i = i + 1
data_aug_test,data_aug_train = split_random(text, int(len(text) * 0.15))
data_aug_train = init_label(data_aug_train)
data_aug_test = init_label(data_aug_test)

data_aug_train.to_csv("./data/datasets/aug/data_aug_train.csv", index = False)
data_aug_test.to_csv("./data/datasets/aug/data_aug_test.csv", index = False)







