from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

# Load and filter the paleontology dataset
paleo_dataset = load_dataset("allenai/sciq", split="train")
paleo_dataset = paleo_dataset.filter(lambda x: any(kw in x['question'].lower() for kw in ["fossil", "dinosaur", "extinction"]))[:1000]  # Adjust size as needed

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small") 
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")

# Preprocessing function for paleontology questions
def preprocess_paleo_function(examples, tokenizer):
    inputs = f"Question: {examples['question']} Context: {examples['support']}"
    targets = examples['correct_answer']

    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess the dataset
paleo_dataset = paleo_dataset.map(lambda examples: preprocess_paleo_function(examples, tokenizer), batched=True)

# Define data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results_paleo_expert",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    save_strategy="no",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=paleo_dataset,
    data_collator=data_collator,
)

# Train the paleontology expert
trainer.train()