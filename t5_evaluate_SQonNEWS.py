from bert_score import score
import json
from tqdm import tqdm

# Load the JSON file
with open('test_newsqa_with_generated_answers_final.json', 'r') as f:
    data = json.load(f)

# Initialize counters
exact_match_count = 0
bertscore_sum = 0.0

# Iterate over the data and calculate evaluations
for item in tqdm(data['data']):
    text = item['text']
    for question in item['questions']:
        if 'generated_answer' in question and 'sourcerAnswers' in question['answers'][0]:
            generated_answer = question['generated_answer']
            sourcer_answer = question['answers'][0]['sourcerAnswers'][0]
            if 's' in sourcer_answer and 'e' in sourcer_answer:
                reference_answer = text[sourcer_answer['s']:sourcer_answer['e']]
                
                # Calculate exact match
                if generated_answer.lower() == reference_answer.lower():
                    exact_match_count += 1
                
                # Calculate BERTScore
                _, _, bertscore = score([generated_answer], [reference_answer], lang='en', verbose=False)
                bertscore_sum += bertscore.item()

# Calculate averages
total_questions = sum(len(item['questions']) for item in data['data'])
exact_match_avg = exact_match_count / total_questions if total_questions > 0 else 0
bertscore_avg = bertscore_sum / total_questions if total_questions > 0 else 0

print("Exact Match:", exact_match_avg)
print("BERTScore:", bertscore_avg)
