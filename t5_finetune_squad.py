from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

# Load the SQuAD dataset from the Hugging Face Hub
squad_dataset = load_dataset("squad")

# Define the preprocessing function
def preprocess_function(examples):
    inputs = [f"question: {q}  context: {c}" for q, c in zip(examples["question"], examples["context"])]
    targets = examples["answer_text"]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")

    labels = tokenizer(text_target=targets, max_length=128, truncation=True, padding="max_length")
    labels = [(-100 if label == tokenizer.pad_token_id else label) for label in labels.input_ids]

    model_inputs["labels"] = labels
    return model_inputs

# Load the pre-trained T5 small model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Preprocess the dataset
tokenized_squad = squad_dataset.map(preprocess_function, batched=True, remove_columns=squad_dataset.column_names)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./t5-squad",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
    load_best_model_at_end=True,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["validation"],
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./t5-squad-finetuned")
tokenizer.save_pretrained("./t5-squad-finetuned")

# Load the fine-tuned model
# model = T5ForConditionalGeneration.from_pretrained("./t5-squad-finetuned")
# tokenizer = T5Tokenizer.from_pretrained("./t5-squad-finetuned")