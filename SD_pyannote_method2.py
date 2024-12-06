import os
import requests
import time
import whisper
from pydub import AudioSegment
from pyannote.audio import Pipeline

# Function to download the audio file
def download_audio_file(link, file_name):
    try:
        response = requests.get(link, stream=True)
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

# Function to delete temporary audio files
def delete_audio_file(file_name):
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"Audio file deleted: {file_name}")

# Function to save the final transcription to an Excel file
def save_transcriptions_to_excel(transcriptions, output_file, rec_link):
    import pandas as pd
    # Add recording link to each entry
    for entry in transcriptions:
        entry["recording_link"] = rec_link
    df = pd.DataFrame(transcriptions)
    df.to_excel(output_file, index=False)
    print(f"Transcriptions saved to {output_file}")

# Function to split audio by diarization intervals
def split_audio_by_diarization(audio_path, diarization, buffer=0.1):
    audio = AudioSegment.from_file(audio_path)
    segments = []
    for idx, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        start_ms = max(0, int((turn.start - buffer) * 1000))  # Apply buffer, avoid negative times
        end_ms = int((turn.end + buffer) * 1000)  # Extend the end time
        segment_audio = audio[start_ms:end_ms]
        segment_path = f"segment_{idx}_speaker_{speaker}.wav"
        segment_audio.export(segment_path, format="wav")
        segments.append({"path": segment_path, "start": turn.start, "end": turn.end, "speaker": speaker})
    return segments

# Function to transcribe each audio segment
def transcribe_segments(segments, whisper_model):
    transcriptions = []
    for segment in segments:
        try:
            print(f"Transcribing segment: {segment['path']} (Speaker: {segment['speaker']})")
            transcription = whisper_model.transcribe(segment["path"], word_timestamps=True)
            if "words" not in transcription or not transcription["words"]:
                transcriptions.append({
                    "speaker": segment["speaker"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": transcription["text"].strip()
                })
                continue
            speaker_text = []
            for word in transcription["words"]:
                if segment["start"] <= word["start"] <= segment["end"]:
                    speaker_text.append(word["text"])
            transcriptions.append({
                "speaker": segment["speaker"],
                "start": segment["start"],
                "end": segment["end"],
                "text": " ".join(speaker_text).strip()
            })
        except Exception as e:
            print(f"Error transcribing segment {segment['path']}: {e}")
            continue
    return transcriptions

# Function to smooth boundaries and combine overlapping segments
def smooth_boundaries(transcriptions):
    smoothed = []
    current_segment = None
    for trans in transcriptions:
        if current_segment is None:
            current_segment = trans
        else:
            if current_segment["end"] >= trans["start"]:
                current_segment["text"] += f" {trans['text']}"
                current_segment["end"] = trans["end"]
            else:
                smoothed.append(current_segment)
                current_segment = trans
    if current_segment:
        smoothed.append(current_segment)
    return smoothed

# Main processing loop
def process_audio(reclinks, whisper_model, diarization_pipeline):
    for idx, rec_link in enumerate(reclinks):
        print(f"Processing: {rec_link}")
        file_name = f"audio_temp_{int(time.time())}.mp3"
        file_name = download_audio_file(rec_link, file_name)
        if not file_name:
            continue

        try:
            audio = AudioSegment.from_file(file_name, format="mp3")
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export("converted_file.wav", format="wav")
            audio_path = "converted_file.wav"
            diarization = diarization_pipeline(audio_path, min_speakers=3, max_speakers=5)
            segments = split_audio_by_diarization(audio_path, diarization)
            transcriptions = transcribe_segments(segments, whisper_model)
            smoothed_transcriptions = smooth_boundaries(transcriptions)
            output_file = f"SD_pyannote_method2_single_run_output{idx + 1}.xlsx"
            save_transcriptions_to_excel(smoothed_transcriptions, output_file, rec_link)
        except Exception as e:
            print(f"Error processing {rec_link}: {e}")
        finally:
            delete_audio_file(file_name)
            delete_audio_file("converted_file.wav")
            for segment in segments:
                if os.path.exists(segment["path"]):
                    delete_audio_file(segment["path"])

# Main script
if __name__ == "__main__":
    reclinks = [
'https://media.plivo.com/v1/Account/MAYWQZMDLMMTY0YZHINJ/Recording/b28824ce-02f8-4885-9d2e-243d57f6d7a7.mp3'
    ]
    print("Loading Whisper model...")
    whisper_model = whisper.load_model("small")
    print("Loading diarization pipeline...")
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="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")
    process_audio(reclinks, whisper_model, diarization_pipeline)
