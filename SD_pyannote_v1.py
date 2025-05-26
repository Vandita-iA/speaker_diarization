import os
import time
import urllib3
import logging
import whisper
import psycopg2
import requests
import warnings
from pydub import AudioSegment
from urllib.parse import urlparse
from pyannote.audio import Pipeline

# Suppress SSL & warning logs and add timme stamps for each log
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings ("ignore", message= ".*: degrees of freedom is <= 0.*", category=UserWarning)
logging.getLogger("speechbrain").setLevel(logging.WARNING)
logging.getLogger("speechbrain").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ──────────────── Configuration ────────────────
# Credentials and settings for PostgreSQL database

DB_HOST = ''					# your DB host
DB_PORT = ''					# your DB port 
DB_NAME = ''					# your DB name
DB_USER = ''					# your DB user
DB_PASSWORD = ''				# your DB password

# Add your huggingface token here
use_auth_token = ""

# Input Query
input_query = """
              SELECT DISTINCT "Rec. Link"
              FROM table
              WHERE diarized_transcript IS NULL
              AND "Rec. Link" IS NOT NULL;
              """
# Output Query
output_query = """
               UPDATE table
               SET diarized_transcript = %s
               WHERE "Rec. Link" = %s
               """
# ──────────────── Database Connection ────────────────
def get_db_connection():
    """
    Returns a PostgreSQL database connection using global config values.
    """
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        database=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )


# ──────────────── HTTP Downloader ────────────────
def download_http_file(url, download_to_base):
    """
    Downloads an audio file via HTTP and infers the file extension from content-type.
    """
    try:
        logging.info(f"📥 Downloading (HTTP): {url}")
        r = requests.get(url, verify = False,stream=True)
        r.raise_for_status()

        # Determine file extension from content-type
        content_type = r.headers.get('Content-Type', '').lower()
        if 'mpeg' in content_type:
            ext = 'mp3'
        elif 'wav' in content_type or 'wave' in content_type:
            ext = 'wav'
        else:
            logging.warning(f"⚠️ Unknown content type: {content_type}")
            return None, None

        file_path = f"{download_to_base}.{ext}"
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

        return file_path, ext
    except Exception as e:
        logging.error(f"⚠️ HTTP download error: {e}")
        return None, None


# ──────────────── Delete Temporary Files ────────────────
def delete_audio_file(*files):
    """
    Deletes given temporary files if they exist.
    """
    for f in files:
        try:
            if f and os.path.exists(f):
                os.remove(f)
        except Exception as e:
            logging.warning(f"⚠️ File cleanup issue: {e}")


# ──────────────── Model Loading ────────────────

# Load Whisper model for transcriptio
model = whisper.load_model(
    "small", 
    device = "cpu"
    )

# Load PyAnnote diarization pipeline (requires Hugging Face auth token)
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token
    )


# ──────────────── Identify Full URL from Rec Link ────────────────
def resolve_rec_link(rec_link):
    """
    Determines the audio source type (SharePoint or other) and constructs a full download URL.
    """
    rec_link = rec_link.strip()

    if (rec_link.startswith('https')) and (rec_link.endswith('.mp3')):
        # Add the specified prefix and suffix
        modified_link = rec_link
        return "not-voitekk", modified_link
            
    elif  (rec_link.startswith('https')) and (rec_link.endswith('.wav')):
        # Add the specified prefix and suffix
        modified_link = rec_link
        return "not-voitekk", modified_link
            
    elif (not rec_link.startswith('https')) and (not rec_link.endswith('.mp3')):
        # Add the specified prefix and suffix
        modified_link = 'https://192.168.1.99/newdumprecords/' + rec_link + '.mp3'
        return "voitek", modified_link

    elif (not rec_link.startswith('https')) and (rec_link.endswith('.mp3')):
        # Add the specified prefix
        modified_link = 'https://192.168.1.99/newdumprecords/' + rec_link 
        return "voitek", modified_link

    elif (rec_link.startswith('https')) and (not rec_link.endswith('.mp3')):
        # Add the specified prefix
        modified_link = rec_link +'.mp3'
        return "voitek", modified_link

    else:
        modified_link = rec_link
    print(modified_link)       


# ──────────────── Main Logic ────────────────

with get_db_connection() as conn:
    with conn.cursor() as cursor:
        # Fetch unique recording links that haven't been transcribed yet
        cursor.execute(input_query)

        rec_links = [row[0] for row in cursor.fetchall()]
        print(f"Number of rec links to be processed are {len(rec_links)}: ")

        for rec_link in rec_links:
            try:
                method, url = resolve_rec_link(rec_link)
                file_base = f"audio_temp_{int(time.time())}"
                output_audio_file = "converted.wav"

                # Download the audio
                if file_path:
                    file_path, ext = download_http_file(url, file_base)
                else:
                    continue

                # Convert to wav
                if ext == "mp3":
                    audio = AudioSegment.from_file(file_path, format="mp3")
                    audio.set_channels(1).set_frame_rate(16000).export(output_audio_file, format="wav")
                elif ext == "wav":
                    os.rename(file_path, output_audio_file)
                else:
                    logging.warning("⚠️ Unsupported audio format.")
                    continue

                # Perform Diarization + Transcription
                diarization = pipeline(output_audio_file, min_speakers=2, max_speakers=6)
                transcription = model.transcribe(output_audio_file, word_timestamps=True, fp16=False)

                segments = transcription["segments"]
                output = []
                current_speaker = None
                current_text = ""

                for segment in segments:
                    start_time = segment["start"]
                    end_time = segment["end"]
                    text = segment["text"]
                    selected_speaker = None
                    max_overlap = 0

                    # Determine most overlapping speaker turn
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        overlap = max(0, min(end_time, turn.end) - max(start_time, turn.start))
                        if overlap > max_overlap:
                            max_overlap = overlap
                            selected_speaker = speaker

                    # Accumulate text per speake
                    if selected_speaker == current_speaker:
                        current_text += f" {text}"
                    else:
                        if current_speaker is not None:
                            output.append(f"{current_speaker}: {current_text.strip()}")
                        current_speaker = selected_speaker
                        current_text = text

                if current_text:
                    output.append(f"{current_speaker}: {current_text.strip()}")

                final_transcript = " ".join(output)

                # Save transcript to database
                cursor.execute(output_query, (final_transcript, rec_link))
                conn.commit()
                logging.info("✅ Transcript saved.")

            except Exception as e:
                logging.error(f"❌ Error processing {rec_link}: {e}")
            finally:
                delete_audio_file(file_path, output_audio_file)
