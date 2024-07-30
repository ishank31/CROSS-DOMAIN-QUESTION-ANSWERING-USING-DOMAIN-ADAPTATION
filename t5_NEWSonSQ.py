from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import json

# Load T5 model pretrained on NEWSQA
tokenizer_newsqa = AutoTokenizer.from_pretrained("shivaniNK8/t5-small-finetuned-cnn-news")
model_newsqa = AutoModelForSeq2SeqLM.from_pretrained("shivaniNK8/t5-small-finetuned-cnn-news")

# Load the test_newsqa.json file
with open("test_newsqa.json", "r") as file:
    data_newsqa = json.load(file)

# Evaluate on NEWSQA
count = 0
for story in tqdm(data_newsqa["data"]):
    if count > 150:
        break
    text = story["text"]
    story_id = story["storyId"]
    questions = story["questions"]

    for question in questions:
        q = question["q"]
        inputs = f"question: {q} context: {text}"
        inputs_ids = tokenizer_newsqa(inputs, return_tensors="pt").input_ids
        outputs = model_newsqa.generate(inputs_ids)
        answer = tokenizer_newsqa.decode(outputs[0], skip_special_tokens=True)

        question["generated_answer"] = answer

    if count % 10 == 0:
        with open(f"t5NEWSQA_test_newsqa_with_generated_answers_{count}.json", "w") as file:
            json.dump(data_newsqa, file, indent=4)

    count += 1

with open("t5NEWSQA_test_newsqa_with_generated_answers_final.json", "w") as file:
    json.dump(data_newsqa, file, indent=4)
