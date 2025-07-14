import os
import torch
import argparse
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- Configuration ---
BASE_MODEL_NAME = "google/gemma-2-9b-it"  # A powerful, modern base model
DATASET_FILE = "lyrics_dataset.txt"
OUTPUT_DIR = "kavi_finetuned_model"
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 1 # Keep low for large models and LoRA
SAVE_STEPS = 500

def load_lyrics_dataset(file_path, tokenizer, is_dummy_run=False):
    """
    Loads the consolidated lyrics dataset from a text file.
    
    Args:
        file_path (str): The path to the lyrics_dataset.txt file.
        tokenizer: The model's tokenizer.
        is_dummy_run (bool): If True, only loads the first 30 entries.

    Returns:
        A Hugging Face Dataset object ready for training.
    """
    print(f"Loading dataset from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    # Split the text into individual lyrics using our separator
    # We also filter out any empty strings that might result from the split
    lyrics = [lyric.strip() for lyric in full_text.split("\n\n---\n\n") if lyric.strip()]
    
    if is_dummy_run:
        print("--- DUMMY RUN ENABLED: Using only the first 30 lyrics. ---")
        lyrics = lyrics[:30]

    if not lyrics:
        raise ValueError("No lyrics found in the dataset file. Please check the file and separator.")

    print(f"Loaded {len(lyrics)} lyrics for the dataset.")
    
    # Create a Hugging Face Dataset object
    dataset = Dataset.from_dict({"text": lyrics})
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
        
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset

def main(args):
    """
    Main function to run the fine-tuning process for the Kavi model using LoRA.
    """
    print("--- Starting Kavi Model Fine-Tuning (with LoRA) ---")

    # --- Step 1: Load Tokenizer and Model ---
    print(f"Loading tokenizer and base model for '{BASE_MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the model with quantization for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=None, # Can add BitsAndBytesConfig for 4-bit quantization if needed
    )
    print("Base model and tokenizer loaded successfully.")

    # --- Step 2: Configure LoRA ---
    print("Configuring model for LoRA (Parameter-Efficient Fine-Tuning)...")
    # First, prepare the model for k-bit training (even if not using it, it's good practice)
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,  # Rank of the update matrices. Higher rank means more parameters to train.
        lora_alpha=32,  # Alpha parameter for scaling.
        target_modules=["q_proj", "v_proj"],  # Target modules to apply LoRA to.
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    print("LoRA configured. Trainable parameters:")
    model.print_trainable_parameters()

    # --- Step 3: Prepare Dataset ---
    train_dataset = load_lyrics_dataset(DATASET_FILE, tokenizer, args.dummy_run)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # --- Step 4: Configure Training ---
    print("Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=50,
        report_to="none", # Disable wandb or other reporting for now
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

    # --- Step 6: Save the Final LoRA Adapters ---
    print("Saving the final fine-tuned LoRA adapters...")
    trainer.save_model(OUTPUT_DIR)
    # The tokenizer is saved automatically by the Trainer in this setup
    print(f"Model adapters saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the Kavi lyrics model with LoRA.")
    parser.add_argument(
        "--dummy_run",
        action="store_true",
        help="If set, runs the training on a small subset of the data for testing."
    )
    args = parser.parse_args()
    main(args)
