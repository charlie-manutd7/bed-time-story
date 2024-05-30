import os
import pandas as pd
from pathlib import Path


def prepare_metadata(audio_dir, output_path):
    metadata = []
    for wav_file in Path(audio_dir).glob("*.wav"):
        # Assuming the file name contains the transcript, e.g., "sample1_Hello this is a test.wav"
        transcript = wav_file.stem.split("_")[1]
        metadata.append([wav_file.stem, transcript, wav_file])

    # Save metadata to a CSV file
    metadata_df = pd.DataFrame(
        metadata, columns=["id", "transcript", "wav_path"])
    metadata_df.to_csv(output_path, index=False, header=False)


if __name__ == "__main__":
    prepare_metadata("voice_samples", "filelists/metadata.csv")
