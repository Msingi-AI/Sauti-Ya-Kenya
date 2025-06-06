import os
import zipfile
import shutil
from pathlib import Path

def zip_clips_directory(clips_dir="cv-corpus-21.0-delta-2025-03-14/sw/clips", 
                       output_dir="upload_ready"):
    os.makedirs(output_dir, exist_ok=True)
    
    shutil.copy2("cv-corpus-21.0-delta-2025-03-14/sw/validated_sentences.tsv", 
                 os.path.join(output_dir, "validated_sentences.tsv"))
    shutil.copy2("cv-corpus-21.0-delta-2025-03-14/sw/clip_durations.tsv",
                 os.path.join(output_dir, "clip_durations.tsv"))
    
    audio_files = list(Path(clips_dir).glob("*.mp3"))
    
    target_size = 200 * 1024 * 1024
    total_files = len(audio_files)
    files_per_zip = max(1, total_files // ((os.path.getsize(clips_dir) // target_size) + 1))
    
    for i in range(0, total_files, files_per_zip):
        batch_files = audio_files[i:i + files_per_zip]
        zip_name = f"clips_part_{i//files_per_zip + 1}.zip"
        zip_path = os.path.join(output_dir, zip_name)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in batch_files:
                zipf.write(file, os.path.join("clips", file.name))
    
    print("\nPreparation complete!")
    print("Upload these files to Colab:")
    print("1. validated_sentences.tsv")
    print("2. clip_durations.tsv")
    print("3. clips_part_*.zip files")
    print(f"\nFiles are ready in the '{output_dir}' directory")

if __name__ == "__main__":
    zip_clips_directory()
