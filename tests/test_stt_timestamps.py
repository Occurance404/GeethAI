import torch
import os
import sys
import json
import re
import soundfile as sf
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# --- Step 1: Determine the audio file path ---
# Check if a file path is provided as a command-line argument
if len(sys.argv) > 1:
    audio_file_path = sys.argv[1]
    if not os.path.exists(audio_file_path):
        print(f"Error: The provided file path does not exist: {audio_file_path}")
        sys.exit(1)
else:
    # If no argument is provided, use the first .wav file found for testing
    print("No audio file provided. Using the first .wav file found for testing.")
    audio_dir = "/home/system613-43/DoNootTouch/model/Downloaded_Audio"
    first_song_id = os.listdir(audio_dir)[0]
    audio_file_path = None
    for root, _, files in os.walk(os.path.join(audio_dir, first_song_id)):
        for file in files:
            if file.endswith(".wav"):
                audio_file_path = os.path.join(root, file)
                break
        if audio_file_path:
            break

if not audio_file_path:
    print("No .wav audio file found to process.")
    sys.exit(1)

print(f"Using audio file for testing: {audio_file_path}")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    # Load processor and model
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to(device)

    # Load audio file
    audio_input, sample_rate = sf.read(audio_file_path)

    # Ensure audio is mono and resample to 16kHz as Whisper expects
    if audio_input.ndim > 1:
        audio_input = librosa.to_mono(audio_input.T)
    if sample_rate != 16000:
        audio_input = librosa.resample(audio_input, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    # Process audio input
    input_features = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)

    # Generate transcription, letting the model handle prompting internally
    output_dict = model.generate(
        input_features,
        return_timestamps=True,
        language="te",
        task="transcribe",
        return_dict_in_generate=True,
    )

    # Decode the sequences to get the transcription with timestamps
    transcription_with_timestamps = processor.batch_decode(
        output_dict["sequences"],
        skip_special_tokens=False,
        decode_with_timestamps=True,
    )

    print("\n--- Transcription Complete ---")

    # --- Process and save the timestamped lyrics as a JSON file ---
    if transcription_with_timestamps:
        full_text = transcription_with_timestamps[0]
        
        # Extract title from the audio file path
        song_title = os.path.splitext(os.path.basename(audio_file_path))[0]

        lines = []
        # Use regex to find all timestamp-text pairs
        matches = re.findall(r"<\|(.*?)\|>(.*?)(?=<\||$)", full_text)
        for i in range(len(matches)):
            start_time_str = matches[i][0]
            text = matches[i][1].strip()
            
            # The last timestamp is the end time of the last segment
            if i + 1 < len(matches):
                end_time_str = matches[i+1][0]
            else:
                # For the very last line, we don't have a following timestamp
                # We can either skip it or use a placeholder. Let's use the start time.
                end_time_str = start_time_str

            if text: # Only add lines that have text content
                lines.append({
                    "start_time": f"{float(start_time_str):.2f}",
                    "end_time": f"{float(end_time_str):.2f}",
                    "text": text
                })

        # Create the final JSON structure
        output_data = {
            "title": song_title,
            "full_lyrics": " ".join([line['text'] for line in lines]),
            "lines": lines
        }

        # Define the output JSON file path
        json_output_path = os.path.join(os.path.dirname(audio_file_path), "lyrics.json")

        # Save the data to the JSON file
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        
        print(f"\nSuccessfully generated and saved lyrics to: {json_output_path}")

    else:
        print("No transcription was generated.")

except Exception as e:
    print(f"An error occurred during transcription: {e}")