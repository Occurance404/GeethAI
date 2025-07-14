import os
import soundfile as sf
import librosa

# The root directory containing the song subdirectories
audio_dir = "/home/system613-43/DoNootTouch/model/Downloaded_Audio"
# The duration to trim from the beginning of each file, in seconds
trim_duration = 10

print(f"Starting audio preprocessing...")
print(f"Root directory: {audio_dir}")
print(f"Trimming first {trim_duration} seconds from each .wav file.\n")

# Keep track of processed files
processed_count = 0
error_count = 0

# Walk through all subdirectories of the root audio directory
for root, _, files in os.walk(audio_dir):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            print(f"Processing: {file_path}")

            try:
                # 1. Load the audio file to get data and sample rate
                audio_data, sample_rate = librosa.load(file_path, sr=None, mono=False)

                # 2. Calculate the number of samples to trim
                trim_samples = int(trim_duration * sample_rate)

                # 3. Check if the audio is long enough to be trimmed
                if len(audio_data) > trim_samples:
                    # 4. Trim the audio by slicing the numpy array
                    trimmed_audio = audio_data[trim_samples:]

                    # 5. Overwrite the original file with the trimmed audio
                    sf.write(file_path, trimmed_audio, sample_rate)
                    print(f"  -> Successfully trimmed and saved.")
                    processed_count += 1
                else:
                    print(f"  -> SKIPPED: Audio is shorter than {trim_duration} seconds.")

            except Exception as e:
                print(f"  -> ERROR: Could not process file. Reason: {e}")
                error_count += 1

print(f"\n--- Preprocessing Complete ---")
print(f"Successfully processed: {processed_count} files")
print(f"Errors encountered:   {error_count} files")
