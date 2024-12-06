import os
import time
import whisper
import requests
import urllib3
import pandas as pd
from openpyxl import load_workbook
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


def append_to_excel(file_path, data):
    # Check if file exists
    if not os.path.exists(file_path):
        # If file doesn't exist, create it with the column headers
        headers = list(data.keys())
        df = pd.DataFrame(columns=headers)
        df.to_excel(file_path, index=False)
    
    # Open the file in append mode and add the new data
    workbook = load_workbook(file_path)
    sheet = workbook.active
    row = [data[key] for key in data.keys()]  # Convert the data dictionary to a list of values
    sheet.append(row)  # Append the new row
    workbook.save(file_path)
    workbook.close()


input_file = "Transcripts_with_cleaned_data_250.xlsx"
df = pd.read_excel(input_file)
reclinks = df["rec_link"]
output_file = "Processed_Transcripts.xlsx"


if os.path.exists(output_file):
    output_data = pd.read_excel(output_file)
    processed_links = set(output_data["rec_link"])
else:
    output_data = pd.DataFrame(columns=["rec_link", "transcript"])
    processed_links = set()

model = whisper.load_model("small")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="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")

for rec_link in reclinks:
    
    # Skip already processed records
    if rec_link in processed_links:
        continue
    
    print(f"Processing: {rec_link}")
    file_name = f"audio_temp_{int(time.time())}.mp3"
    # link = "https://media.plivo.com/v1/Account/MAYWQZMDLMMTY0YZHINJ/Recording/0eaa572d-3756-490d-b9ab-b7647887360f.mp3"
    file_name = download_audio_file(rec_link, file_name)

    if not file_name:
        continue

    try:

        start_time2 = time.time()
        audio = AudioSegment.from_file(file_name, format="mp3")

        # Convert to WAV format with mono channel and 16 kHz sample rate
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export("converted_file.wav", format="wav")

        # Load pre-trained diarization pipeline

        # v2.1_speaker_diariz_run1 access token = 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

        # Path to your converted WAV file
        audio_path = "converted_file.wav"
        # Run diarization
        diarization = pipeline(audio_path, min_speakers=2, max_speakers=5)


        # Perform transcription
        transcription = model.transcribe("converted_file.wav", word_timestamps=True)
        segments = transcription["segments"]

        # Combine speaker labels with transcript
        output = []

        # diarization_result = []
        # for turn, _, speaker in diarization.itertracks(yield_label=True):
        #     diarization_result.append(f"{turn.start:.1f} - {turn.end:.1f}: Speaker {speaker}\n")



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

        transcript = " ".join(output)
        # Save to Excel file
        append_to_excel(output_file, {"rec_link": rec_link, "transcript": transcript})
        print("Processing complete.")

        end_time2 = time.time()
        print(f"total time taken = {(end_time2-start_time2)/60} minutes")
        
        with open("progress_log.txt", "a") as log_file:
            log_file.write(f"Processed: {rec_link} in {(end_time2-start_time2)/60} minutes using {model} model\n\n")
            log_file.write(f"Transcript: {transcript}\n\n")
            log_file.write(f"Time taken: {(end_time2-start_time2)/60} minutes\n\n\n")


    except Exception as e:
        print(f"Error processing {rec_link}: {e}")
    finally:
        # Clean up temporary files
        delete_audio_file(file_name)














