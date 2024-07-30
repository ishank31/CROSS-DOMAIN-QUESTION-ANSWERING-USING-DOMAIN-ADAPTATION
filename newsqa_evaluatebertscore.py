from transformers import pipeline

pipe = pipeline("question-answering", model="google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")

import pandas as pd

val = pd.read_csv('test_newsqa.csv')

import pandas as pd
import numpy as np
import concurrent.futures

temp_val = val.copy()

# Define a thread pool executor
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Apply the pipe function using the thread pool

    rows = [row for i, row in temp_val.iterrows()]
    temp_val['predicted_answers'] = list(executor.map(lambda x: pipe(question=x.question, context=x.context)['answer'], rows))

temp_val.to_csv('test_newsqa_answers.csv')

from evaluate import load

data = pd.read_csv('test_newsqa_answers.csv')
new_metric = load('bertscore')

precisions = []
f1scores = []

for _, row in data.iterrows():
    pred = row.predicted_answers
    golds = eval(row.answers)['text']

    max_precision = float('-inf')
    max_f1score = float('-inf')
    for gold in golds:
        score = new_metric.compute(references=[gold], predictions=[pred], lang='en')
        f1score = score['f1'][0]
        precision = score['precision'][0]
        if f1score > max_f1score:
            max_f1score = f1score
        if precision > max_precision:
            max_precision = precision

    precisions.append(max_precision)
    f1scores.append(max_f1score)
    print(max_precision, max_f1score)

data['f1_bert'] = f1scores
data['precisions_bert'] = precisions

data.to_csv('test_newsqa_answers_metrics.csv')

