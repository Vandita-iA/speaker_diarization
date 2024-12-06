import os
import time
import whisper
import requests
import urllib3
import pandas as pd
from pydub import AudioSegment
from pyannote.audio import Pipeline
# Load your MP3 file

def download_audio_file(link, file_name):
    try:
        urllib3.disable_warnings()
        response = requests.get(link, verify=False, stream=True)
        if response.status_code == 200:
            with open(file_name, 'wb') as f:
                f.write(response.content)
            print(f"Audio file saved: {file_name}")
            return file_name
        else:
            print(f"Failed to download file: {link}")
            return None
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        return None
    

def delete_audio_file(file_name):
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"Audio file deleted: {file_name}")
    else:
        print(f"File not found: {file_name}")

    if os.path.exists(audio_path):
        os.remove(audio_path)
        print(f"Converted WAV file deleted")
    else:
        print(f"Converted WAV file not found")


file_name = f"audio_temp2_{int(time.time())}.mp3"
link = "https://media.plivo.com/v1/Account/MAYWQZMDLMMTY0YZHINJ/Recording/0eaa572d-3756-490d-b9ab-b7647887360f.mp3"
file_name = download_audio_file(link, file_name)

start_time2 = time.time()
audio = AudioSegment.from_file(file_name, format="mp3")

# Convert to WAV format with mono channel and 16 kHz sample rate
audio = audio.set_channels(1).set_frame_rate(16000)
audio.export("converted_file2.wav", format="wav")
# Path to your converted WAV file
audio_path = "converted_file2.wav"

# Load pre-trained diarization pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")

# Run diarization
diarization = pipeline(audio_path, min_speakers=3, max_speakers=5)

model = whisper.load_model("small")

# Perform transcription
transcription = model.transcribe("converted_file2.wav", word_timestamps=True)
segments = transcription["segments"]

# Combine speaker labels with transcript
output = []

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f} - {turn.end:.1f}: Speaker {speaker}")


# Output speaker segments
try:
    current_speaker = None
    current_text = ""

    for segment in segments:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]

        # Find the speaker from diarization
        matched_speakers = []
        max_overlap = 0
        selected_speaker = None

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Calculate overlap between transcription and diarization segments
            overlap_start = max(start_time, turn.start)
            overlap_end = min(end_time, turn.end)
            overlap_duration = max(0, overlap_end - overlap_start)

            if overlap_duration > 0:
                matched_speakers.append((speaker, overlap_duration))
                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    selected_speaker = speaker

        # Assign text to the current speaker or start a new block
        if selected_speaker:
            if selected_speaker == current_speaker:
                current_text += f" {text}"  # Append to the current speaker's text
            else:
                # Add the completed block for the previous speaker
                if current_speaker is not None:
                    output.append(f"{current_speaker}: {current_text.strip()}\n")
                # Start a new block for the new speaker
                current_speaker = selected_speaker
                current_text = text
        else:
            # Handle unmatched segments (optional)
            if current_speaker is not None:
                output.append(f"{current_speaker}: {current_text.strip()}")
            current_speaker = "Unknown Speaker"
            current_text = text

    # Add the final block of text
    if current_text:
        output.append(f"{current_speaker}: {current_text.strip()}")

    for line in output:
        print(line)

    transcript = " ".join(output)

    end_time2 = time.time()
    print(f"total time taken = {(end_time2-start_time2)/60} minutes")
    
    with open("progress_log.txt", "a") as log_file:
        log_file.write(f"Processed: {link} in {(end_time2-start_time2)/60} minutes using {model} model\n\n")
        log_file.write(f"Transcript: {transcript}\n\n")
        log_file.write(f"Time taken: {(end_time2-start_time2)/60} minutes\n\n\n")


except Exception as e:
    print(f"Error processing {link}: {e}")
finally:
    # Clean up temporary files
    delete_audio_file(file_name)
