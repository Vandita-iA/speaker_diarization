# SD_pyannote_v1

A script to run speaker‐diarization (via **pyannote**) on your audio streams, backed by a database for storing/querying results, and using a gated Hugging Face model.

---

## Table of Contents

- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Configuration](#configuration)  
  - [1. Environment Variables (.env)](#1-environment-variables-env)  
  - [2. In‐Script Placeholders](#2-in-script-placeholders)  
  - [3. Hugging Face Token & Gated Model](#3-hugging-face-token--gated-model)  
- [Usage](#usage)  
- [all-requirements.txt](#all-requirements.txt)  
- [Project Structure](#project-structure)  
- [License](#license)  

---

## Prerequisites

- **Python 3.8+**  
- [`pip`](https://pip.pypa.io/en/stable/) (bundled with Python)  
- (Recommended) [virtualenv](https://virtualenv.pypa.io/) or built-in `venv`

---

## Installation

1. **Clone or download** this repo:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo

2. **Create & activate** a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r all-requirements.txt

---

## Configuration

Before running the script, you’ll need to supply:
1. **Database connection details**
   Create a .env file in the project root with the following keys:
   ```bash
   DB_HOST=your_db_host
   DB_PORT=5432
   DB_NAME=your_db_name
   DB_USER=your_db_user
   DB_PASSWORD=your_db_password
   use_auth_token=your_huggingface_token

3. **SQL queries** to fetch and update your audio‐metadata and diarized transcripts respectively
   Open SD_pyannote_v1.py and look for the following markers—replace the sample values or placeholder comments with your real parameters:
   ```bash
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
   
4. **Hugging Face auth token** for gated model access:

   [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   You need to agree to share your contact information to access this model. This repository is publicly accessible, but you have to accept the conditions to access its files and content.

---

## Usage

With your virtualenv active and .env configured, simply run:
    ```bash
    python SD_pyannote_v1.py

If you need to customize (e.g. point to a file instead of a DB), edit the corresponding placeholder blocks in SD_pyannote_v1.py.

---

## requirements.txt
All Python dependencies are pinned here. To regenerate after adding/updating:

    ```bash
    pip freeze > requirements.txt

---

## Project Structure
     ```bash
     your-repo/
     ├── SD_pyannote_v1.py         # Main script
     ├── all-requirements.txt      # Python dependencies
     └── README.md                 # This file
