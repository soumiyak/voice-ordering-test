from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import numpy as np
import soundfile as sf
import noisereduce as nr
import openai
from dotenv import load_dotenv
import os
import pandas as pd
import re
import subprocess
from collections import defaultdict
from io import StringIO
from pathlib import Path

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load default menu
menu_df = pd.DataFrame()
default_menu_path = "menu.csv"
if os.path.exists(default_menu_path):
    menu_df = pd.read_csv(default_menu_path)
    print(f"✅ Loaded default menu: {default_menu_path}")
else:
    print("⚠ Default menu.csv not found. Please upload one via /upload-csv/")

@app.post("/upload-csv/")
async def upload_csv(csvFile: UploadFile = File(...)):
    try:
        contents = await csvFile.read()
        decoded_csv = contents.decode("utf-8")

        # Save to disk
        with open("menu.csv", "w", encoding="utf-8") as f:
            f.write(decoded_csv)

        # Load to memory
        global menu_df
        menu_df = pd.read_csv(StringIO(decoded_csv))

        return {
            "status": "success",
            "message": f"CSV file '{csvFile.filename}' uploaded successfully.",
            "rows": len(menu_df),
            "columns": list(menu_df.columns)
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

SUPPORTED_AUDIO_TYPES = {
    "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/webm",
    "audio/ogg", "audio/x-m4a", "audio/mp4", "audio/flac", "audio/aac",
    "audio/x-aiff"
}

SUPPORTED_VIDEO_TYPES = {
    "video/mp4", "video/webm", "video/x-m4v", "video/x-msvideo"
}

def denoise_audio(data, sr=16000):
    try:
        noise_sample = data[:sr] if len(data) >= sr else data
        if np.all(noise_sample == 0):
            return data
        y_nr = nr.reduce_noise(y=data, sr=sr, y_noise=noise_sample)
        return y_nr
    except Exception:
        return data

# def convert_audio_to_wav(input_path, output_path):
#     cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path]
#     subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def convert_audio_to_wav(input_path, output_path):
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("FFmpeg conversion error:", result.stderr.decode())
        raise RuntimeError("FFmpeg audio conversion failed.")

# def extract_audio_from_video(input_path, output_path):
#     cmd = ["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "mp3", "-ar", "16000", "-ac", "1", output_path]
#     subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def extract_audio_from_video(input_path, output_path):
    cmd = ["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "mp3", "-ar", "16000", "-ac", "1", output_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("FFmpeg video audio extraction error:", result.stderr.decode())
        raise RuntimeError("FFmpeg video audio extraction failed.")

def transcribe_audio(path):
    try:
        with open(path, "rb") as audio_file:
            response = openai.Audio.transcribe(
                model="gpt-4o-mini-transcribe",
                file=audio_file
            )
        return response["text"]
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return ""

def gpt_menu_match(transcribed_text, menu_df):
    menu_str = ", ".join(
    f"{row['item_id']} | {row['item_name']}" for _, row in menu_df.iterrows()
    )
    prompt = f"""
Extract the food items and their quantities from the following spoken order:

"{transcribed_text}"

Only match items from this menu:
{menu_str}

Instructions:
- If the input is in a non-English language (e.g., Hindi, Tamil, Telugu, Urdu), translate it to English before processing.
- Ignore filler or address words like "bhaiya", "anna", "dada", "boss", or similar – they are not part of the order.
- Identify food items and map them to the closest valid menu item, even if the names are partially spoken or mispronounced.
  Examples: 
  "Spoken Order: hot crispy chicken" → Original Menu item: "Hot & Crispy Chicken Bucket (6 pcs)", 
  "Spoken Order: paneer popper" → "Original Menu item: Paneer Poppers", 
  "Spoken Order: ginger burger" → "Original Menu item: Veg Zinger Burger", 
  "Spoken Order: grilled zinger" → "Original Menu item: Tandoori Zinger Burger"
  "Spoken Order: chicken sixty five" → "Original Menu item: chicken 65"
- Use numeric digits for quantities (e.g., "two" → 2). If quantity is not stated, assume 1.
- Do not return "NaN" or leave blanks — always return numeric values.
- If the same item appears multiple times, combine them and total the quantity.
- If the user stutters or changes their mind mid-sentence (e.g., says "aaannnwwww one dosa... annnn one onion dosa"), discard incomplete or earlier mentions and keep only the final, most specific version ("1 Onion Dosa"). Ignore false starts, repetitions, or abandoned phrases.
- Only include items that exist in the menu. Do not hallucinate and invent or guess items not found.
- Output should be a single clean, comma-separated list (no bullet points or extra text), in the following format:
  item_id / id | item_name | quantity
  Example: 101 | Veg Zinger Burger | 1, 103 | Hot & Crispy Chicken Bucket (6 pcs) | 2, 110 | Pepsi 500 ml | 1
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception:
        return ""


def standardize_menu_columns(menu_df):
    print("Inside")
    df = menu_df.copy()
    df.columns = [col.strip().lower() for col in df.columns]

    # Define mapping
    column_mapping = {}
    for col in df.columns:
        if col in ["id", "item id", "menu id"]:
            column_mapping[col] = "item_id"
        elif col in ["name", "dish name", "item name", "dish_name", "Name", "Item Name"]:
            column_mapping[col] = "item_name"
        elif col in ["price", "cost", "Price", "Cost"]:
            column_mapping[col] = "price"


    df.rename(columns=column_mapping, inplace=True)

    # Check for required columns
    required_cols = ["item_id", "item_name", "price"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError("Menu must contain item_id, item_name, and price columns.")

    # Return only what's needed for parsing
    return df[required_cols]

def parse_order(text, menu_df):
    print("inside parse_order")

    import re
    from collections import defaultdict

    print(menu_df.head())

    # menu_df = standardize_menu_columns(menu_df)

    response = gpt_menu_match(text, menu_df)
    print("GPT Response:",response)
    if not response:
        return [], ""
    

    df = menu_df.copy()  
    parsed_order = []

    # Parse entries like: 101 | Veg Zinger Burger | 1
    entries = [e.strip() for e in response.split(",") if e.strip()]
    for entry in entries:
        parts = [p.strip() for p in entry.split("|")]
        if len(parts) == 3:
            try:
                item_id = int(parts[0])
                item_name = parts[1]
                quantity = int(parts[2])

                row = df[df["item_id"] == item_id]
                if not row.empty:
                    menu_item = str(row.iloc[0]["item_name"])
                    if item_name.lower() in menu_item.lower():
                        parsed_order.append({
                            "Id": item_id,
                            "Item": item_name,
                            "Quantity": quantity
                    })
            except ValueError:
                continue  # skip rows with parsing errors

    return parsed_order, response

@app.post("/process/")
async def process_audio_file(file: UploadFile = File(...)):
    global menu_df  # ✅ Ensure you're using the global variable
    print("File received:", file.filename)
    print(f"Received content type: {file.content_type}")

    # print("File received:", file.filename)
    try:
        ext = Path(file.filename).suffix.lower()

        AUDIO_EXTENSIONS = {".wav", ".mp3", ".webm", ".ogg", ".m4a", ".mp4", ".flac", ".aac", ".aiff"}
        VIDEO_EXTENSIONS = {".mp4", ".webm", ".m4v", ".avi"}

        if ext in VIDEO_EXTENSIONS:
            suffix = ext if ext else ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as raw:
                raw.write(await file.read())
                raw_path = raw.name
            audio_path = raw_path + "_audio.mp3"
            extract_audio_from_video(raw_path, audio_path)

        elif ext in AUDIO_EXTENSIONS:
            suffix = ext if ext else ".webm"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as raw:
                raw.write(await file.read())
                raw_path = raw.name
            audio_path = raw_path

        else:
            return {"error": f"Unsupported file extension: {ext}"}

        # Convert to WAV
        converted_path = audio_path + "_converted.wav"
        convert_audio_to_wav(audio_path, converted_path)

        # Read and denoise
        data, samplerate = sf.read(converted_path)
        print(f"Loaded audio shape: {data.shape}, Sample rate: {samplerate}")

        if data.ndim > 1:
            data = data[:, 0]
        denoised = denoise_audio(data, samplerate)

        # Save temp denoised file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            sf.write(temp_audio.name, denoised, samplerate)

            # Transcribe
            transcription = transcribe_audio(temp_audio.name)
            print("Original Transcription:", transcription)

            if menu_df.empty:
                return {"error": "No menu available. Please upload a menu CSV or ensure menu.csv is in server."}

            # Before parsing, standardize menu columns
            menu_df = standardize_menu_columns(menu_df)

            # print(menu_df.head())
            parsed_order, response_text = parse_order(transcription, menu_df)

        return {
            "transcription": response_text,
            "order": parsed_order
        }

    except Exception as e:
        return {"error": f"Unexpected error: {e}"}
    
    