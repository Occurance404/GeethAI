import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling,
)

# --- Configuration ---
MODEL_NAME = "google/gemma-3-12b-it"
DATASET_FILE = "/home/system613-43/DoNootTouch/model/kavi_dataset.txt"
OUTPUT_DIR = "/home/system613-43/DoNootTouch/model/kavi_finetuned_model"
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 1 # Keep this low for large models
SAVE_STEPS = 500 # Save a checkpoint every 500 steps

def main():
    """
    Main function to run the fine-tuning process for the Kavi model.
    """
    print("--- Starting Kavi Model Fine-Tuning ---")

    # --- Step 1: Verify Dataset ---
    if not os.path.exists(DATASET_FILE) or os.path.getsize(DATASET_FILE) == 0:
        print(f"ERROR: Dataset file not found or is empty at: {DATASET_FILE}")
        print("Please run 'assemble_kavi_dataset.py' first to create the dataset.")
        return

    print(f"Using dataset: {DATASET_FILE}")
    print(f"Model will be saved to: {OUTPUT_DIR}")

    # --- Step 2: Load Tokenizer and Model ---
    print(f"Loading tokenizer and model for '{MODEL_NAME}'...")
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Set a padding token if it doesn't exist. EOS token is a good choice for GPT-like models.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the model
    # We use bfloat16 to reduce memory usage and speed up training on compatible GPUs.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto", # Automatically distribute the model across available GPUs
    )
    print("Model and tokenizer loaded successfully.")

    # --- Step 3: Prepare Dataset and Collator ---
    print("Preparing dataset for training...")
    
    # The TextDataset handles reading the text file and preparing it for the model.
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=DATASET_FILE,
        block_size=128, # The block size for text sequences
    )

    # The DataCollatorForLanguageModeling handles creating batches of data for training.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False, # We are doing Causal Language Modeling, not Masked Language Modeling
    )
    print(f"Dataset prepared with {len(train_dataset)} examples.")

    # --- Step 4: Configure Training ---
    print("Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        save_steps=SAVE_STEPS,
        save_total_limit=2, # Only keep the last 2 checkpoints
        prediction_loss_only=True,
        logging_dir='./logs',
        logging_steps=100,
    )

    # --- Step 5: Initialize and Run Trainer ---
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    print("--- Starting Training ---")
    trainer.train()
    print("--- Training Complete ---")

    # --- Step 6: Save the Final Model ---
    print("Saving the final fine-tuned model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
