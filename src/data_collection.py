import tkinter as tk
from tkinter import ttk, messagebox
import sounddevice as sd
import soundfile as sf
import numpy as np
import os

import json
import threading
from datetime import datetime
from pathlib import Path
import queue
import wave
from typing import Optional, Dict, List

class AudioRecorder:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.recording = False
        self.audio_data = []
        self.stream = None
        self.audio_queue = queue.Queue()

    def callback(self, indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        if self.recording:
            self.audio_queue.put(indata.copy())

    def start_recording(self):
        self.recording = True
        self.audio_data = []
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.callback
        )
        self.stream.start()

    def stop_recording(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        while not self.audio_queue.empty():
            self.audio_data.append(self.audio_queue.get())

        if not self.audio_data:
            return np.array([])

        return np.concatenate(self.audio_data, axis=0)

class DataCollectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kenyan Swahili TTS Data Collection")
        self.root.geometry("800x600")

        self.recorder = AudioRecorder()
        self.current_recording = None
        self.metadata = {}
        
        self.data_dir = Path("data")
        self.recordings_dir = self.data_dir / "recordings"
        self.metadata_file = self.data_dir / "metadata.json"
        self.create_directories()
        self.load_metadata()

        self.setup_ui()

    def create_directories(self):
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.metadata_file.exists():
            with open(self.metadata_file, "w") as f:
                json.dump({}, f)

    def load_metadata(self):
        try:
            with open(self.metadata_file, "r") as f:
                self.metadata = json.load(f)
        except json.JSONDecodeError:
            self.metadata = {}

    def save_metadata(self):
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def setup_ui(self):
        speaker_frame = ttk.LabelFrame(self.root, text="Speaker Information", padding=10)
        speaker_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(speaker_frame, text="Speaker ID:").grid(row=0, column=0, sticky="w")
        self.speaker_id = ttk.Entry(speaker_frame)
        self.speaker_id.grid(row=0, column=1, sticky="ew")

        ttk.Label(speaker_frame, text="Gender:").grid(row=1, column=0, sticky="w")
        self.gender = ttk.Combobox(speaker_frame, values=["Male", "Female", "Other"])
        self.gender.grid(row=1, column=1, sticky="ew")

        ttk.Label(speaker_frame, text="Age:").grid(row=2, column=0, sticky="w")
        self.age = ttk.Entry(speaker_frame)
        self.age.grid(row=2, column=1, sticky="ew")

        ttk.Label(speaker_frame, text="Dialect:").grid(row=3, column=0, sticky="w")
        self.dialect = ttk.Entry(speaker_frame)
        self.dialect.grid(row=3, column=1, sticky="ew")

        text_frame = ttk.LabelFrame(self.root, text="Recording Text", padding=10)
        text_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.text_input = tk.Text(text_frame, height=5)
        self.text_input.pack(fill="both", expand=True)

        controls_frame = ttk.Frame(self.root, padding=10)
        controls_frame.pack(fill="x", padx=10, pady=5)

        self.record_button = ttk.Button(
            controls_frame,
            text="Start Recording",
            command=self.toggle_recording
        )
        self.record_button.pack(side="left", padx=5)

        self.play_button = ttk.Button(
            controls_frame,
            text="Play Recording",
            command=self.play_recording,
            state="disabled"
        )
        self.play_button.pack(side="left", padx=5)

        self.save_button = ttk.Button(
            controls_frame,
            text="Save Recording",
            command=self.save_recording,
            state="disabled"
        )
        self.save_button.pack(side="left", padx=5)

        status_frame = ttk.Frame(self.root, padding=10)
        status_frame.pack(fill="x", padx=10, pady=5)

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var
        )
        self.status_label.pack(fill="x")

        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(
            status_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress.pack(fill="x", pady=5)

    def toggle_recording(self):
        if not self.recorder.recording:
            if not self.validate_inputs():
                return

            self.recorder.start_recording()
            self.record_button.configure(text="Stop Recording")
            self.status_var.set("Recording...")
            self.play_button.configure(state="disabled")
            self.save_button.configure(state="disabled")
            
            self.update_progress()
        else:
            self.current_recording = self.recorder.stop_recording()
            self.record_button.configure(text="Start Recording")
            self.status_var.set("Recording stopped")
            self.play_button.configure(state="normal")
            self.save_button.configure(state="normal")
            self.progress_var.set(0)

    def update_progress(self):
        if self.recorder.recording:
            current = self.progress_var.get()
            if current >= 100:
                current = 0
            self.progress_var.set(current + 1)
            self.root.after(100, self.update_progress)

    def validate_inputs(self):
        if not self.speaker_id.get():
            messagebox.showerror("Error", "Please enter a Speaker ID")
            return False
        if not self.gender.get():
            messagebox.showerror("Error", "Please select a Gender")
            return False
        if not self.age.get():
            messagebox.showerror("Error", "Please enter an Age")
            return False
        if not self.dialect.get():
            messagebox.showerror("Error", "Please enter a Dialect")
            return False
        if not self.text_input.get("1.0", "end-1c").strip():
            messagebox.showerror("Error", "Please enter text to record")
            return False
        return True

    def play_recording(self):
        if self.current_recording is not None:
            try:
                sd.play(self.current_recording, self.recorder.sample_rate)
                sd.wait()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to play recording: {e}")

    def save_recording(self):
        if self.current_recording is None:
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.speaker_id.get()}_{timestamp}.wav"
            filepath = self.recordings_dir / filename

            with wave.open(str(filepath), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.recorder.sample_rate)
                wf.writeframes((self.current_recording * 32767).astype(np.int16))

            metadata_entry = {
                "speaker_id": self.speaker_id.get(),
                "gender": self.gender.get(),
                "age": self.age.get(),
                "dialect": self.dialect.get(),
                "text": self.text_input.get("1.0", "end-1c").strip(),
                "timestamp": timestamp,
                "sample_rate": self.recorder.sample_rate,
                "duration": len(self.current_recording) / self.recorder.sample_rate
            }

            self.metadata[filename] = metadata_entry
            self.save_metadata()

            self.current_recording = None
            self.play_button.configure(state="disabled")
            self.save_button.configure(state="disabled")
            self.text_input.delete("1.0", "end")
            
            messagebox.showinfo("Success", "Recording saved successfully!")
            self.status_var.set("Ready")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save recording: {e}")

def main():
    root = tk.Tk()
    app = DataCollectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
