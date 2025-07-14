import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
MODEL_NAME = "google/gemma-3-12b-it"
# Set a specific cache directory within our project to keep things organized
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "huggingface_cache")

def main():
    """
    Downloads and caches a Hugging Face model and its tokenizer.
    """
    print("--- Starting Model Download ---")
    print(f"Model to download: {MODEL_NAME}")
    print(f"Models will be cached in: {CACHE_DIR}")

    # Create the cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)

    try:
        # --- Download Tokenizer ---
        print("Downloading tokenizer...")
        AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        print("Tokenizer downloaded successfully.")

        # --- Download Model ---
        print("Downloading model... (This may take a while)")
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        )
        print("Model downloaded successfully.")

    except Exception as e:
        print(f"\nAn error occurred during the download process: {e}")
        print("Please check your internet connection and Hugging Face credentials if required.")

    print("\n--- Download Complete ---")
    print(f"The model '{MODEL_NAME}' is now cached in the '{CACHE_DIR}' directory.")

if __name__ == "__main__":
    main()
