"""
Data collection utilities for Kenyan Swahili TTS
"""
import argparse
import json
import logging
import os
import sounddevice as sd
import soundfile as sf
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Optional, Dict, List
import wave
import numpy as np
from datetime import datetime

class AudioRecorder:
    """Audio recording with real-time monitoring"""
    def __init__(self, sample_rate: int = 22050, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = False
        self.audio_data = []
        
    def callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_data.extend(indata.copy())
            
    def start_recording(self):
        self.recording = True
        self.audio_data = []
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.callback
        )
        self.stream.start()
        
    def stop_recording(self) -> np.ndarray:
        self.recording = False
        self.stream.stop()
        self.stream.close()
        return np.concatenate(self.audio_data)

class DataCollectionGUI:
    """GUI for collecting Swahili speech data"""
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.recorder = AudioRecorder()
        self.current_prompt = ""
        self.recording = False
        self.prompts = self.load_prompts()
        self.prompt_index = 0
        self.metadata = {}
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "wavs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)
        
        self.setup_gui()
        
    def load_prompts(self) -> List[str]:
        """Load recording prompts"""
        prompt_file = os.path.join(self.output_dir, "prompts.txt")
        if os.path.exists(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        return []
        
    def setup_gui(self):
        """Set up the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Sauti ya Kenya - Data Collection")
        self.root.geometry("800x600")
        
        # Speaker info frame
        info_frame = ttk.LabelFrame(self.root, text="Speaker Information")
        info_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(info_frame, text="Speaker ID:").grid(row=0, column=0, padx=5, pady=5)
        self.speaker_id = ttk.Entry(info_frame)
        self.speaker_id.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(info_frame, text="Gender:").grid(row=0, column=2, padx=5, pady=5)
        self.gender = ttk.Combobox(info_frame, values=["Male", "Female", "Other"])
        self.gender.grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(info_frame, text="Age:").grid(row=1, column=0, padx=5, pady=5)
        self.age = ttk.Entry(info_frame)
        self.age.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(info_frame, text="Dialect:").grid(row=1, column=2, padx=5, pady=5)
        self.dialect = ttk.Entry(info_frame)
        self.dialect.grid(row=1, column=3, padx=5, pady=5)
        
        # Prompt display
        prompt_frame = ttk.LabelFrame(self.root, text="Recording Prompt")
        prompt_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.prompt_text = scrolledtext.ScrolledText(prompt_frame, wrap=tk.WORD, height=5)
        self.prompt_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Controls
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=5, pady=5)
        
        self.record_button = ttk.Button(
            control_frame, 
            text="Start Recording",
            command=self.toggle_recording
        )
        self.record_button.pack(side="left", padx=5)
        
        self.next_button = ttk.Button(
            control_frame,
            text="Next Prompt",
            command=self.next_prompt
        )
        self.next_button.pack(side="left", padx=5)
        
        self.save_button = ttk.Button(
            control_frame,
            text="Save Session",
            command=self.save_session
        )
        self.save_button.pack(side="right", padx=5)
        
        # Status
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_label = ttk.Label(self.root, textvariable=self.status_var)
        status_label.pack(pady=5)
        
        self.update_prompt()
        
    def toggle_recording(self):
        """Toggle recording state"""
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        """Start recording audio"""
        self.recording = True
        self.record_button.config(text="Stop Recording")
        self.status_var.set("Recording...")
        self.recorder.start_recording()
        
    def stop_recording(self):
        """Stop recording and save audio"""
        self.recording = False
        self.record_button.config(text="Start Recording")
        self.status_var.set("Processing...")
        
        # Get audio data
        audio_data = self.recorder.stop_recording()
        
        # Save audio file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"speaker_{self.speaker_id.get()}_{timestamp}.wav"
        filepath = os.path.join(self.output_dir, "wavs", filename)
        
        sf.write(filepath, audio_data, self.recorder.sample_rate)
        
        # Save metadata
        metadata = {
            "speaker_id": self.speaker_id.get(),
            "gender": self.gender.get(),
            "age": self.age.get(),
            "dialect": self.dialect.get(),
            "text": self.current_prompt,
            "audio_file": filename,
            "timestamp": timestamp
        }
        
        metadata_file = os.path.join(
            self.output_dir, 
            "metadata",
            f"{filename}.json"
        )
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        self.status_var.set("Recording saved!")
        
    def update_prompt(self):
        """Update the display prompt"""
        if self.prompt_index < len(self.prompts):
            self.current_prompt = self.prompts[self.prompt_index]
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert("1.0", self.current_prompt)
        else:
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert("1.0", "No more prompts!")
            
    def next_prompt(self):
        """Move to next prompt"""
        self.prompt_index += 1
        self.update_prompt()
        
    def save_session(self):
        """Save the recording session"""
        self.status_var.set("Session saved!")
        
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

def main():
    parser = argparse.ArgumentParser(description="Collect Swahili speech data")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save recordings"
    )
    args = parser.parse_args()
    
    app = DataCollectionGUI(args.output_dir)
    app.run()

if __name__ == "__main__":
    main()
