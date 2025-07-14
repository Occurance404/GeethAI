import torch
from transformers import pipeline
import glob
import os
from tqdm import tqdm
import librosa

# --- Configuration ---
MODEL_NAME = "openai/whisper-large-v3"
AUDIO_DIR = "Downloaded_Audio"
OUTPUT_FILE = "lyrics_dataset.txt"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SAMPLING_RATE = 16000
# This marker helps us track which files have been successfully processed.
PROGRESS_MARKER_PREFIX = "---Processed-File:"

def get_already_processed_files(output_file_path):
    """
    Reads the output file to find which files have already been transcribed
    by looking for the progress marker. This allows the script to be restartable.
    """
    processed_files = set()
    if not os.path.exists(output_file_path):
        return processed_files
    
    with open(output_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith(PROGRESS_MARKER_PREFIX):
                # Extract the filename from the marker line
                filename = line.strip().split(PROGRESS_MARKER_PREFIX)[1].strip()
                processed_files.add(filename)
    return processed_files

def main():
    """
    Generates a single, consolidated text file of lyrics.
    This script is restartable and will skip files that have already been processed.
    """
    print("Initializing script...")
    
    # --- Step 1: Setup the transcription pipeline ---
    print(f"Initializing pipeline with model: {MODEL_NAME} on device: {DEVICE}")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL_NAME,
        torch_dtype=torch.float16,
        device=DEVICE,
        model_kwargs={},
    )

    # --- Step 2: Find all audio files ---
    search_path = os.path.join(AUDIO_DIR, "**", "*.wav")
    all_audio_files = glob.glob(search_path, recursive=True)
    if not all_audio_files:
        print(f"No .wav files found in {AUDIO_DIR}. Exiting.")
        return

    # --- Step 3: Determine which files to skip (Resume Logic) ---
    processed_files = get_already_processed_files(OUTPUT_FILE)
    files_to_process = [f for f in all_audio_files if os.path.basename(f) not in processed_files]
    
    print(f"Found {len(all_audio_files)} total files.")
    if processed_files:
        print(f"Found {len(processed_files)} already processed files. Skipping them.")
    print(f"Processing {len(files_to_process)} new files.")

    if not files_to_process:
        print("No new files to process. Exiting.")
        return

    # --- Step 4: Process each new file ---
    for file_path in tqdm(files_to_process, desc="Transcribing files"):
        try:
            audio_input, _ = librosa.load(file_path, sr=SAMPLING_RATE, mono=True)

            output = pipe(
                audio_input,
                chunk_length_s=30,
                stride_length_s=5,
                generate_kwargs={"language": "telugu"},
                return_timestamps=False,
            )

            clean_text = output["text"].strip() if output and output.get("text") else ""

            if not clean_text:
                print(f"Warning: No transcription generated for {file_path}")
                continue

            # --- Step 5: Save progress immediately ---
            # We write the transcription and then a marker line with the filename.
            # This makes our process robust and restartable.
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                f.write(clean_text)
                f.write("\n\n---\n\n")
                f.write(f"{PROGRESS_MARKER_PREFIX} {os.path.basename(file_path)}\n")

        except Exception as e:
            print(f"ERROR processing file {file_path}: {e}")
            print("This file will be skipped. Progress on other files is saved.")
            continue

    print(f"Transcription generation complete. Dataset saved to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
