from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

# Load the bioinformatics dataset
bioinfo_dataset = load_dataset("bioasq", split="train[:1000]")  # Adjust dataset size as needed

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small") 
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")

# Preprocessing function for bioinformatics questions
def preprocess_bioinfo_function(examples, tokenizer):
    inputs = f"Question: {examples['question']} Context: {examples['context']}"
    targets = examples['answer']

    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess the dataset
bioinfo_dataset = bioinfo_dataset.map(lambda examples: preprocess_bioinfo_function(examples, tokenizer), batched=True)

# Define data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results_bioinfo_expert",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    save_strategy="no",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=bioinfo_dataset,
    data_collator=data_collator,
)

# Train the bioinformatics expert
trainer.train()