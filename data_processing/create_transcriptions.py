import torch
from transformers import pipeline
import glob
import json
import os
from tqdm import tqdm
import librosa

# --- Configuration ---
MODEL_NAME = "openai/whisper-base"
AUDIO_DIR = "Downloaded_Audio"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SAMPLING_RATE = 16000

def format_timestamp(seconds):
    """Converts seconds to HH:MM:SS.mmm format."""
    if seconds is None:
        return "00:00:00.000"
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{milliseconds:03}"

def main():
    """
    Generates timestamped transcriptions for all .wav files in the audio directory
    using a chunked pipeline for efficiency and handling long audio.
    """
    print(f"Initializing pipeline with model: {MODEL_NAME} on device: {DEVICE}")
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL_NAME,
        torch_dtype=torch.float16,
        device=DEVICE,
    )

    search_path = os.path.join(AUDIO_DIR, "**", "*.wav")
    audio_files = glob.glob(search_path, recursive=True)
    
    if not audio_files:
        print(f"No .wav files found in {AUDIO_DIR}. Exiting.")
        return

    print(f"Found {len(audio_files)} .wav files to process.")

    for file_path in tqdm(audio_files, desc="Processing files"):
        try:
            # Load the entire audio file, resample to 16kHz, and convert to mono
            audio_input, _ = librosa.load(file_path, sr=SAMPLING_RATE, mono=True)

            # Process the entire file at once; the pipeline handles chunking
            output = pipe(
                audio_input,
                chunk_length_s=30,  # Use 30s chunks, standard for Whisper
                stride_length_s=5,   # Add a 5s overlap for better context
                generate_kwargs={"language": "telugu"},
                return_timestamps="word",
            )

            if not output or not output.get("chunks"):
                print(f"Warning: No transcription generated for {file_path}")
                continue

            full_text = output["text"].strip()
            all_words = []
            for word in output["chunks"]:
                if word.get("timestamp"):
                    start_time, end_time = word["timestamp"]
                    # The pipeline might return None for timestamps of padding/special tokens
                    if start_time is not None and end_time is not None:
                        all_words.append({
                            "text": word["text"],
                            "start_time": format_timestamp(start_time),
                            "end_time": format_timestamp(end_time)
                        })

            if not all_words:
                print(f"Warning: No words with timestamps found for {file_path}")
                continue

            # The overall start and end times are simply the first and last word's times
            start_time_overall = all_words[0]["start_time"]
            end_time_overall = all_words[-1]["end_time"]

            transcription_data = {
                "title": os.path.splitext(os.path.basename(file_path))[0],
                "full_text": full_text,
                "segments": [{
                    "start_time": start_time_overall,
                    "end_time": end_time_overall,
                    "text": full_text
                }],
                "words": all_words
            }

            json_filename = os.path.splitext(file_path)[0] + ".json"
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(transcription_data, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

    print("Transcription generation complete.")

if __name__ == "__main__":
    main()