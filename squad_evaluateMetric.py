import pandas as pd
from evaluate import load



data = pd.read_csv('predicted_answers_squad_val.csv')
new_metric = load('bleu')

scores = []


for _, row in data.iterrows():
    pred = row.predicted_answers
    golds = eval(row.answers)['text']

    max_score = float('-inf')
    score = new_metric.compute(references=[golds], predictions=[pred])
    
    scores.append(score['bleu'])

data['bleu'] = scores

data.to_csv('predicted_answers_squad_val.csv')

print(data.bleu.mean())


