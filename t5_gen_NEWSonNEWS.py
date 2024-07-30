from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import json

# Load model directly


tokenizer = AutoTokenizer.from_pretrained("shivaniNK8/t5-small-finetuned-cnn-news")
model = AutoModelForSeq2SeqLM.from_pretrained("shivaniNK8/t5-small-finetuned-cnn-news")

# Load the test_newsqa.json file
with open("test_newsqa.json", "r") as file:
    data = json.load(file)

# Calculate outputs for each story
count = 0
for story in tqdm(data["data"]):
    if count > 150:
        break
    text = story["text"]
    story_id = story["storyId"]
    questions = story["questions"]

    # Process each question
    for question in questions:
        q = question["q"]
        inputs = f"question: {q} context: {text}"
        inputs_ids = tokenizer(inputs, return_tensors="pt").input_ids
        outputs = model.generate(inputs_ids)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Add the generated answer to the question
        question["generated_answer"] = answer

    # Save the updated data every 10 stories
    if count % 10 == 0:
        with open(f"t5NEWSQA_test_newsqa_with_generated_answers_{count}.json", "w") as file:
            json.dump(data, file, indent=4)

    count += 1

# Save the final updated data to a new file

with open("t5NEWSQA_test_newsqa_with_generated_answers_final.json", "w") as file:
    json.dump(data, file, indent=4)