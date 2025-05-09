from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load a small transformer model
model_name = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load a small dataset (CNN/DailyMail for summarization)
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1000]")

# Preprocess data
def preprocess_function(examples):
    inputs = [ex for ex in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(examples["highlights"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    save_strategy="no",
    logging_steps=50,
)

# Train model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)
trainer.train()
