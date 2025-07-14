import os
import json
import sys

# The root directory containing the song subdirectories
audio_dir = "/home/system613-43/DoNootTouch/model/Downloaded_Audio"
# The name of the output file
output_file_name = "kavi_dataset.txt"
# The absolute path for the output file
output_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file_name)

print("Starting Kavi dataset assembly...")
print(f"Looking for lyrics.json files in: {audio_dir}")

# Find all lyrics.json files recursively
json_files_to_process = []
for root, _, files in os.walk(audio_dir):
    for file in files:
        if file == "lyrics.json":
            json_files_to_process.append(os.path.join(root, file))

total_files = len(json_files_to_process)
if total_files == 0:
    print("No 'lyrics.json' files found yet. The transcription script may still be running.")
    print("Run this script again after some files have been processed.")
    sys.exit(0)

print(f"Found {total_files} 'lyrics.json' files to process.\n")

# Use a set to store unique lyrics to avoid duplicates
unique_lyrics = set()

# Process each file
for i, json_path in enumerate(json_files_to_process):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "full_lyrics" in data and data["full_lyrics"]:
                lyrics_text = data["full_lyrics"].strip()
                if lyrics_text:
                    unique_lyrics.add(lyrics_text)
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {json_path}. Skipping.")
    except Exception as e:
        print(f"An unexpected error occurred while processing {json_path}: {e}")

# Write the unique lyrics to the output file
if not unique_lyrics:
    print("No valid lyrics were extracted. The output file will not be created.")
    sys.exit(0)

print(f"Extracted {len(unique_lyrics)} unique song lyrics.")
print(f"Writing dataset to: {output_file_path}")

with open(output_file_path, 'w', encoding='utf-8') as f:
    # Each song's lyrics are separated by two newlines
    f.write("\n\n".join(unique_lyrics))

print("\n--- Kavi Dataset Assembly Complete ---")