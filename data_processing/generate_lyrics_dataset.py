import os
import subprocess
import sys

# The root directory containing the song subdirectories
audio_dir = "/home/system613-43/DoNootTouch/model/Downloaded_Audio"
# The path to the transcription script we will call
transcription_script = "/home/system613-43/DoNootTouch/model/test_stt_timestamps.py"

print("Starting dataset generation...")
print(f"Looking for .wav files in: {audio_dir}")

# Find all .wav files recursively
wav_files_to_process = []
for root, _, files in os.walk(audio_dir):
    for file in files:
        if file.endswith(".wav"):
            wav_files_to_process.append(os.path.join(root, file))

total_files = len(wav_files_to_process)
print(f"Found {total_files} .wav files to process.\n")

# Process each file
for i, wav_path in enumerate(wav_files_to_process):
    print(f"--- Processing file {i+1}/{total_files} ---")
    print(f"Audio file: {wav_path}")

    # Check if the corresponding lyrics.json already exists
    json_output_path = os.path.join(os.path.dirname(wav_path), "lyrics.json")
    if os.path.exists(json_output_path):
        print(f"Lyrics already exist at: {json_output_path}. SKIPPING.")
        continue

    # Construct the command to run the transcription script
    command = [
        "python",
        transcription_script,
        wav_path
    ]

    try:
        # Execute the command
        # We use subprocess.run to wait for the script to complete
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True, # This will raise an exception if the script returns a non-zero exit code
            encoding='utf-8'
        )
        # Print the output from the script
        print(process.stdout)
        if process.stderr:
            print("Errors from script:")
            print(process.stderr)

    except subprocess.CalledProcessError as e:
        print(f"ERROR: The transcription script failed for {wav_path}.")
        print(f"Return code: {e.returncode}")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
    except Exception as e:
        print(f"An unexpected error occurred while processing {wav_path}: {e}")

print("\n--- Full Dataset Generation Complete ---")
