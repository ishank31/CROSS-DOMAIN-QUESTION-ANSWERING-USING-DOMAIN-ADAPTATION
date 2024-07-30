from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import json

# Load T5 model pretrained on SQuAD
tokenizer_squad = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
model_squad = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")

# Load the test SQuAD dataset
from datasets import load_dataset
squad_dataset = load_dataset('squad')

# Evaluate on SQuAD
count = 0
for example in tqdm(squad_dataset["test"]):
    if count > 150:
        break
    question = example["question"]
    context = example["context"]
    inputs = f"question: {question} context: {context}"
    inputs_ids = tokenizer_squad(inputs, return_tensors="pt").input_ids
    outputs = model_squad.generate(inputs_ids)
    answer = tokenizer_squad.decode(outputs[0], skip_special_tokens=True)

    example["generated_answer"] = answer

    if count % 10 == 0:
        with open(f"t5SQuAD_test_with_generated_answers_{count}.json", "w") as file:
            json.dump(squad_dataset, file, indent=4)

    count += 1

with open("t5SQuAD_test_with_generated_answers_final.json", "w") as file:
    json.dump(squad_dataset, file, indent=4)
