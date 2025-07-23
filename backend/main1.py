# # # # import threading
# # # # import queue
# # # # import numpy as np
# # # # import sounddevice as sd
# # # # from faster_whisper import WhisperModel
# # # # from fastapi import FastAPI, UploadFile, File
# # # # from fastapi.middleware.cors import CORSMiddleware
# # # # import tempfile, os, subprocess, re
# # # # import pandas as pd
# # # # import soundfile as sf
# # # # import noisereduce as nr
# # # # from dotenv import load_dotenv
# # # # from pathlib import Path
# # # # from io import StringIO
# # # # from collections import defaultdict
# # # # import openai

# # # # # ───── Setup ───────────────────────────────────────────────────────────────── #
# # # # load_dotenv()
# # # # openai.api_key = os.getenv("OPENAI_API_KEY")
# # # # app = FastAPI()

# # # # app.add_middleware(
# # # #     CORSMiddleware,
# # # #     allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
# # # # )

# # # # # ───── Global State ───────────────────────────────────────────────────────── #
# # # # menu_df = pd.DataFrame()
# # # # live_transcript = ""
# # # # audio_queue = queue.Queue()
# # # # audio_buffer = []

# # # # model = WhisperModel("small", device="cpu", compute_type="float32")

# # # # # ───── Load menu if available ─────────────────────────────────────────────── #
# # # # menu_path = "menu.csv"
# # # # if os.path.exists(menu_path):
# # # #     menu_df = pd.read_csv(menu_path)
# # # #     print("✅ Loaded menu.csv")
# # # # else:
# # # #     print("⚠️ menu.csv not found")

# # # # # ───── Whisper Live Audio Recorder ───────────────────────────────────────── #
# # # # def audio_callback(indata, frames, time, status):
# # # #     if status:
# # # #         print("Stream status:", status)
# # # #     audio_queue.put(indata.copy())

# # # # def recorder():
# # # #     with sd.InputStream(samplerate=16000, channels=1, callback=audio_callback, blocksize=int(16000 * 0.5)):
# # # #         print("🎤 Whisper real-time listening started...")
# # # #         while True:
# # # #             sd.sleep(100)

# # # # def transcriber():
# # # #     global audio_buffer, live_transcript
# # # #     frames_per_chunk = int(16000 * 4)  # 2 seconds

# # # #     while True:
# # # #         block = audio_queue.get()
# # # #         audio_buffer.append(block)
# # # #         total_frames = sum(len(b) for b in audio_buffer)

# # # #         if total_frames >= frames_per_chunk:
# # # #             audio_data = np.concatenate(audio_buffer)[:frames_per_chunk].flatten().astype(np.float32)
# # # #             audio_buffer.clear()

# # # #             segments, _ = model.transcribe(audio_data, language="en", beam_size=5)
# # # #             for segment in segments:
# # # #                 print("🗣️", segment.text)
# # # #                 live_transcript = segment.text  # Save latest transcript

# # # # # Start background threads
# # # # threading.Thread(target=recorder, daemon=True).start()
# # # # threading.Thread(target=transcriber, daemon=True).start()

# # # # # ───── Routes ─────────────────────────────────────────────────────────────── #
# # # # @app.get("/live-transcription/")
# # # # async def get_live_transcription():
# # # #     return {"transcription": live_transcript}

# # # # @app.post("/upload-csv/")
# # # # async def upload_csv(csvFile: UploadFile = File(...)):
# # # #     try:
# # # #         contents = await csvFile.read()
# # # #         decoded = contents.decode("utf-8")
# # # #         with open("menu.csv", "w", encoding="utf-8") as f:
# # # #             f.write(decoded)

# # # #         global menu_df
# # # #         menu_df = pd.read_csv(StringIO(decoded))
# # # #         return {"status": "success", "message": f"{csvFile.filename} uploaded", "rows": len(menu_df)}
# # # #     except Exception as e:
# # # #         return {"status": "error", "message": str(e)}

# # # # def denoise_audio(data, sr=16000):
# # # #     try:
# # # #         noise_sample = data[:sr] if len(data) >= sr else data
# # # #         if np.all(noise_sample == 0): return data
# # # #         return nr.reduce_noise(y=data, sr=sr, y_noise=noise_sample)
# # # #     except: return data

# # # # def convert_audio_to_wav(input_path, output_path):
# # # #     subprocess.run(["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path],
# # # #                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

# # # # def extract_audio_from_video(input_path, output_path):
# # # #     subprocess.run(["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "mp3", "-ar", "16000", "-ac", "1", output_path],
# # # #                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

# # # # def transcribe_audio(path):
# # # #     try:
# # # #         with open(path, "rb") as f:
# # # #             resp = openai.Audio.transcribe(model="gpt-4o-mini-transcribe", file=f)
# # # #         return resp["text"]
# # # #     except Exception as e:
# # # #         print("Transcription error:", e)
# # # #         return ""

# # # # def gpt_menu_match(transcribed_text, menu_df):
# # # #     menu_items = ", ".join(menu_df["Item Name"].astype(str).tolist())
# # # #     prompt = f"""Extract food items and quantities from this order: "{transcribed_text}"
# # # # Menu: {menu_items}
# # # # Use digits. Ignore filler words. Match partial names to menu. Output like "1 Item A, 2 Item B" only."""

# # # #     try:
# # # #         resp = openai.ChatCompletion.create(
# # # #             model="gpt-4o-mini",
# # # #             messages=[{"role": "user", "content": prompt}],
# # # #             temperature=0
# # # #         )
# # # #         return resp['choices'][0]['message']['content'].strip()
# # # #     except Exception:
# # # #         return ""

# # # # def parse_order(text, menu_df):
# # # #     parsed = gpt_menu_match(text, menu_df)
# # # #     if not parsed: return [], ""

# # # #     df = menu_df.copy()
# # # #     df.columns = [col.lower().strip() for col in df.columns]
# # # #     id_col = next((c for c in df.columns if "id" in c), None)
# # # #     name_col = next((c for c in df.columns if "name" in c), None)
# # # #     price_col = next((c for c in df.columns if "price" in c), None)

# # # #     order = defaultdict(lambda: {"Id": "", "Quantity": 0, "Rate": 0, "Total": 0})
# # # #     entries = [e.strip() for e in parsed.split(",") if e.strip()]
# # # #     for entry in entries:
# # # #         match = re.match(r"(\d+)\s+(.+)", entry)
# # # #         if match:
# # # #             qty = int(match.group(1))
# # # #             item = match.group(2).strip()
# # # #             row = df[df[name_col].str.lower() == item.lower()]
# # # #             if not row.empty:
# # # #                 rate = int(row.iloc[0][price_col])
# # # #                 item_id = int(row.iloc[0][id_col])
# # # #                 order[item]["Id"] = item_id
# # # #                 order[item]["Quantity"] += qty
# # # #                 order[item]["Rate"] = rate
# # # #                 order[item]["Total"] += qty * rate

# # # #     parsed_list = [
# # # #         {"Item": k, "Id": v["Id"], "Quantity": v["Quantity"], "Rate": v["Rate"], "Total": v["Total"]}
# # # #         for k, v in order.items()
# # # #     ]
# # # #     total = sum(v["Total"] for v in order.values())
# # # #     if parsed_list:
# # # #         parsed_list.append({"Item": "Total Amount", "Id": "", "Quantity": "", "Rate": "", "Total": total})

# # # #     return parsed_list, parsed

# # # # @app.post("/process/")
# # # # async def process_audio_file(file: UploadFile = File(...)):
# # # #     try:
# # # #         ext = Path(file.filename).suffix.lower()
# # # #         AUDIO = {".wav", ".mp3", ".webm", ".ogg", ".m4a", ".mp4", ".flac", ".aac", ".aiff"}
# # # #         VIDEO = {".mp4", ".webm", ".m4v", ".avi"}

# # # #         if ext in VIDEO:
# # # #             with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
# # # #                 temp_file.write(await file.read())
# # # #                 video_path = temp_file.name
# # # #             audio_path = video_path + "_audio.mp3"
# # # #             extract_audio_from_video(video_path, audio_path)
# # # #         elif ext in AUDIO:
# # # #             with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
# # # #                 temp_file.write(await file.read())
# # # #                 audio_path = temp_file.name
# # # #         else:
# # # #             return {"error": f"Unsupported extension: {ext}"}

# # # #         wav_path = audio_path + "_converted.wav"
# # # #         convert_audio_to_wav(audio_path, wav_path)

# # # #         data, sr = sf.read(wav_path)
# # # #         if data.ndim > 1: data = data[:, 0]
# # # #         clean_audio = denoise_audio(data, sr)

# # # #         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
# # # #             sf.write(tmp.name, clean_audio, sr)
# # # #             transcript = transcribe_audio(tmp.name)

# # # #         if menu_df.empty:
# # # #             return {"error": "No menu found. Upload menu.csv first."}

# # # #         order, parsed_text = parse_order(transcript, menu_df)
# # # #         return {"transcription": parsed_text, "order": order}

# # # #     except Exception as e:
# # # #         return {"error": f"Processing failed: {str(e)}"}
# # # # Combined FastAPI backend with both WebSocket live transcription and upload-based transcription

# # # # import os
# # # # import re
# # # # import queue
# # # # import threading
# # # # import tempfile
# # # # import asyncio
# # # # import subprocess
# # # # from pathlib import Path
# # # # from collections import defaultdict
# # # # from io import StringIO
# # # # import numpy as np
# # # # import pandas as pd
# # # # import sounddevice as sd
# # # # import soundfile as sf
# # # # import noisereduce as nr
# # # # from dotenv import load_dotenv
# # # # from faster_whisper import WhisperModel
# # # # from fastapi import FastAPI, WebSocket, UploadFile, File
# # # # from fastapi.middleware.cors import CORSMiddleware
# # # # import openai

# # # # # Load environment variables
# # # # load_dotenv()
# # # # openai.api_key = os.getenv("OPENAI_API_KEY")

# # # # # FastAPI App
# # # # app = FastAPI()

# # # # app.add_middleware(
# # # #     CORSMiddleware,
# # # #     allow_origins=["*"],
# # # #     allow_credentials=True,
# # # #     allow_methods=["*"],
# # # #     allow_headers=["*"],
# # # # )

# # # # # Global variables for live transcription
# # # # samplerate = 16000
# # # # block_duration = 0.5
# # # # chunk_duration = 2
# # # # channels = 1
# # # # frames_per_block = int(samplerate * block_duration)
# # # # frames_per_chunk = int(samplerate * chunk_duration)
# # # # audio_queue = queue.Queue()
# # # # audio_buffer = []
# # # # model = WhisperModel("small", device="cpu", compute_type="float32")
# # # # connected_websockets = []

# # # # # Load default menu
# # # # menu_df = pd.DataFrame()
# # # # default_menu_path = "menu.csv"
# # # # if os.path.exists(default_menu_path):
# # # #     menu_df = pd.read_csv(default_menu_path)
# # # #     print(f"✅ Loaded default menu: {default_menu_path}")
# # # # else:
# # # #     print("⚠️ Default menu.csv not found. Please upload one via /upload-csv/")

# # # # # Recorder and Callback

# # # # def audio_callback(indata, frames, time, status):
# # # #     if status:
# # # #         print("Stream status:", status)
# # # #     audio_queue.put(indata.copy())

# # # # def recorder():
# # # #     with sd.InputStream(samplerate=samplerate, channels=channels,
# # # #                         callback=audio_callback, blocksize=frames_per_block):
# # # #         print("🎤 Listening... Press Ctrl+C to stop.")
# # # #         while True:
# # # #             sd.sleep(100)

# # # # def start_recorder():
# # # #     threading.Thread(target=recorder, daemon=True).start()

# # # # @app.on_event("startup")
# # # # async def startup_event():
# # # #     start_recorder()

# # # # # WebSocket live transcription
# # # # @app.websocket("/ws/transcribe")
# # # # async def websocket_endpoint(websocket: WebSocket):
# # # #     await websocket.accept()
# # # #     connected_websockets.append(websocket)
# # # #     try:
# # # #         global audio_buffer
# # # #         while True:
# # # #             block = await asyncio.to_thread(audio_queue.get)
# # # #             audio_buffer.append(block)
# # # #             total_frames = sum(len(b) for b in audio_buffer)
# # # #             if total_frames >= frames_per_chunk:
# # # #                 audio_data = np.concatenate(audio_buffer)[:frames_per_chunk]
# # # #                 audio_buffer = []
# # # #                 audio_data = audio_data.flatten().astype(np.float32)
# # # #                 segments, _ = model.transcribe(audio_data, language="en", beam_size=5)
# # # #                 for segment in segments:
# # # #                     text = segment.text
# # # #                     for ws in connected_websockets:
# # # #                         await ws.send_text(text)
# # # #     except Exception as e:
# # # #         print("WebSocket closed:", e)
# # # #     finally:
# # # #         connected_websockets.remove(websocket)

# # # # # Menu Upload
# # # # @app.post("/upload-csv/")
# # # # async def upload_csv(csvFile: UploadFile = File(...)):
# # # #     try:
# # # #         contents = await csvFile.read()
# # # #         decoded_csv = contents.decode("utf-8")
# # # #         with open("menu.csv", "w", encoding="utf-8") as f:
# # # #             f.write(decoded_csv)
# # # #         global menu_df
# # # #         menu_df = pd.read_csv(StringIO(decoded_csv))
# # # #         return {
# # # #             "status": "success",
# # # #             "message": f"CSV file '{csvFile.filename}' uploaded successfully.",
# # # #             "rows": len(menu_df),
# # # #             "columns": list(menu_df.columns)
# # # #         }
# # # #     except Exception as e:
# # # #         return {"status": "error", "message": str(e)}

# # # # # Utility Functions
# # # # SUPPORTED_AUDIO_TYPES = {
# # # #     "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/webm",
# # # #     "audio/ogg", "audio/x-m4a", "audio/mp4", "audio/flac", "audio/aac",
# # # #     "audio/x-aiff"
# # # # }

# # # # def denoise_audio(data, sr=16000):
# # # #     try:
# # # #         noise_sample = data[:sr] if len(data) >= sr else data
# # # #         if np.all(noise_sample == 0):
# # # #             return data
# # # #         y_nr = nr.reduce_noise(y=data, sr=sr, y_noise=noise_sample)
# # # #         return y_nr
# # # #     except Exception:
# # # #         return data

# # # # def convert_audio_to_wav(input_path, output_path):
# # # #     cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path]
# # # #     subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

# # # # def extract_audio_from_video(input_path, output_path):
# # # #     cmd = ["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "mp3", "-ar", "16000", "-ac", "1", output_path]
# # # #     subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

# # # # def transcribe_audio(path):
# # # #     try:
# # # #         with open(path, "rb") as audio_file:
# # # #             response = openai.Audio.transcribe(model="gpt-4o-mini-transcribe", file=audio_file)
# # # #         return response["text"]
# # # #     except Exception as e:
# # # #         print(f"❌ Transcription failed: {e}")
# # # #         return ""

# # # # def gpt_menu_match(transcribed_text, menu_df):
# # # #     menu_str = ", ".join(menu_df["Item Name"].astype(str).tolist())
# # # #     prompt = f"""
# # # # Extract the food items and their quantities from the following spoken order:

# # # # "{transcribed_text}"

# # # # Only match items from this menu:
# # # # {menu_str}
# # # # Instructions:
# # # # - If the input is in a non-English language (e.g., Hindi, Tamil, Telugu, Urdu), translate it to English before processing.
# # # # - Ignore filler or address words like "bhaiya", "anna", "dada", "boss", or similar – they are not part of the order.
# # # # - Identify food items and map them to the closest valid menu item, even if the names are partially spoken or mispronounced.
# # # #   Examples: 
# # # #   "Spoken Order: hot crispy chicken" → Original Menu item: "Hot & Crispy Chicken Bucket (6 pcs)", 
# # # #   "Spoken Order: paneer popper" → "Original Menu item: Paneer Poppers", 
# # # #   "Spoken Order: ginger burger" → "Original Menu item: Veg Zinger Burger", 
# # # #   "Spoken Order: grilled zinger" → "Original Menu item: Tandoori Zinger Burger"
# # # # - Use numeric digits for quantities (e.g., "two" → 2). If quantity is not stated, assume 1.
# # # # - Do not return "NaN" or leave blanks — always return numeric values.
# # # # - If the same item appears multiple times, combine them and total the quantity.
# # # # - If the user stutters or changes their mind mid-sentence (e.g., says "aaannnwwww one dosa... annnn one onion dosa"), discard incomplete or earlier mentions and keep only the final, most specific version ("1 Onion Dosa"). Ignore false starts, repetitions, or abandoned phrases.
# # # # - Only include items that exist in the menu. Do not hallucinate and invent or guess items not found.
# # # # - Output should be a single clean, comma-separated list (no bullet points or extra text), e.g.:
# # # #   1 Veg Zinger Burger, 1 Hot & Crispy Chicken Bucket (6 pcs), 1 Pepsi 500 ml
# # # # - Format: "1 Item A, 2 Item B"
# # # # """
# # # #     try:
# # # #         response = openai.ChatCompletion.create(
# # # #             model="gpt-4o-mini",
# # # #             messages=[{"role": "user", "content": prompt}],
# # # #             temperature=0
# # # #         )
# # # #         return response['choices'][0]['message']['content'].strip()
# # # #     except Exception:
# # # #         return ""

# # # # def parse_order(text, menu_df):
# # # #     response = gpt_menu_match(text, menu_df)
# # # #     if not response:
# # # #         return [], ""
# # # #     df = menu_df.copy()
# # # #     df.columns = [col.lower().strip() for col in df.columns]
# # # #     id_col = next((col for col in df.columns if col in ["id", "item id"]), None)
# # # #     name_col = next((col for col in df.columns if "item" in col and "name" in col), None)
# # # #     price_col = next((col for col in df.columns if "price" in col), None)
# # # #     if not (id_col and name_col and price_col):
# # # #         raise ValueError("Required columns not found")
# # # #     order = defaultdict(lambda: {"Id": "", "Quantity": 0, "Rate": 0, "Total": 0})
# # # #     entries = [e.strip() for e in response.split(",") if e.strip()]
# # # #     for entry in entries:
# # # #         match = re.match(r"(\d+)\s+(.+)", entry)
# # # #         if match:
# # # #             quantity = int(match.group(1))
# # # #             item_name = match.group(2).strip()
# # # #             row = df[df[name_col].str.lower() == item_name.lower()]
# # # #             if not row.empty:
# # # #                 rate = int(row.iloc[0][price_col])
# # # #                 item_id = int(row.iloc[0][id_col])
# # # #                 order[item_name]["Id"] = item_id
# # # #                 order[item_name]["Quantity"] += quantity
# # # #                 order[item_name]["Rate"] = rate
# # # #                 order[item_name]["Total"] += rate * quantity
# # # #     parsed = [
# # # #         {
# # # #             "Item": k,
# # # #             "Id": int(v["Id"]),
# # # #             "Quantity": int(v["Quantity"]),
# # # #             "Rate": int(v["Rate"]),
# # # #             "Total": int(v["Total"])
# # # #         }
# # # #         for k, v in order.items()
# # # #     ]
# # # #     total = sum(v["Total"] for v in order.values())
# # # #     if parsed:
# # # #         parsed.append({"Item": "Total Amount", "Id": "", "Quantity": "", "Rate": "", "Total": total})
# # # #     return parsed, response

# # # # # Audio Upload
# # # # @app.post("/process/")
# # # # async def process_audio_file(file: UploadFile = File(...)):
# # # #     print("File received:", file.filename)
# # # #     try:
# # # #         ext = Path(file.filename).suffix.lower()
# # # #         AUDIO_EXTENSIONS = {".wav", ".mp3", ".webm", ".ogg", ".m4a", ".mp4", ".flac", ".aac", ".aiff"}
# # # #         VIDEO_EXTENSIONS = {".mp4", ".webm", ".m4v", ".avi"}
# # # #         if ext in VIDEO_EXTENSIONS:
# # # #             suffix = ext if ext else ".mp4"
# # # #             with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as raw:
# # # #                 raw.write(await file.read())
# # # #                 raw_path = raw.name
# # # #             audio_path = raw_path + "_audio.mp3"
# # # #             extract_audio_from_video(raw_path, audio_path)
# # # #         elif ext in AUDIO_EXTENSIONS:
# # # #             suffix = ext if ext else ".webm"
# # # #             with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as raw:
# # # #                 raw.write(await file.read())
# # # #                 raw_path = raw.name
# # # #             audio_path = raw_path
# # # #         else:
# # # #             return {"error": f"Unsupported file extension: {ext}"}
# # # #         converted_path = audio_path + "_converted.wav"
# # # #         convert_audio_to_wav(audio_path, converted_path)
# # # #         data, samplerate = sf.read(converted_path)
# # # #         if data.ndim > 1:
# # # #             data = data[:, 0]
# # # #         denoised = denoise_audio(data, samplerate)
# # # #         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
# # # #             sf.write(temp_audio.name, denoised, samplerate)
# # # #             transcription = transcribe_audio(temp_audio.name)
# # # #             if menu_df.empty:
# # # #                 return {"error": "No menu available. Please upload a menu CSV or ensure menu.csv is in server."}
# # # #             parsed_order, response_text = parse_order(transcription, menu_df)
# # # #         return {"transcription": response_text, "order": parsed_order}
# # # #     except Exception as e:
# # # #         return {"error": f"Unexpected error: {e}"}

# # # from fastapi import FastAPI, UploadFile, File
# # # from fastapi.middleware.cors import CORSMiddleware
# # # import tempfile
# # # import numpy as np
# # # import soundfile as sf
# # # import noisereduce as nr
# # # import openai
# # # from dotenv import load_dotenv
# # # import os
# # # import pandas as pd
# # # import re
# # # import subprocess
# # # from collections import defaultdict
# # # from io import StringIO
# # # from pathlib import Path

# # # # Load environment variables
# # # load_dotenv()
# # # openai.api_key = os.getenv("OPENAI_API_KEY")

# # # app = FastAPI()

# # # # CORS Middleware
# # # app.add_middleware(
# # #     CORSMiddleware,
# # #     allow_origins=["*"],
# # #     allow_credentials=True,
# # #     allow_methods=["*"],
# # #     allow_headers=["*"],
# # # )

# # # # Load default menu
# # # menu_df = pd.DataFrame()
# # # default_menu_path = "menu.csv"
# # # if os.path.exists(default_menu_path):
# # #     menu_df = pd.read_csv(default_menu_path)
# # #     print(f"✅ Loaded default menu: {default_menu_path}")
# # # else:
# # #     print("⚠️ Default menu.csv not found. Please upload one via /upload-csv/")

# # # @app.post("/upload-csv/")
# # # async def upload_csv(csvFile: UploadFile = File(...)):
# # #     try:
# # #         contents = await csvFile.read()
# # #         decoded_csv = contents.decode("utf-8")

# # #         # Save to disk
# # #         with open("menu.csv", "w", encoding="utf-8") as f:
# # #             f.write(decoded_csv)

# # #         # Load to memory
# # #         global menu_df
# # #         menu_df = pd.read_csv(StringIO(decoded_csv))

# # #         return {
# # #             "status": "success",
# # #             "message": f"CSV file '{csvFile.filename}' uploaded successfully.",
# # #             "rows": len(menu_df),
# # #             "columns": list(menu_df.columns)
# # #         }

# # #     except Exception as e:
# # #         return {"status": "error", "message": str(e)}

# # # SUPPORTED_AUDIO_TYPES = {
# # #     "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/webm",
# # #     "audio/ogg", "audio/x-m4a", "audio/mp4", "audio/flac", "audio/aac",
# # #     "audio/x-aiff"
# # # }

# # # SUPPORTED_VIDEO_TYPES = {
# # #     "video/mp4", "video/webm", "video/x-m4v", "video/x-msvideo"
# # # }

# # # def denoise_audio(data, sr=16000):
# # #     try:
# # #         noise_sample = data[:sr] if len(data) >= sr else data
# # #         if np.all(noise_sample == 0):
# # #             return data
# # #         y_nr = nr.reduce_noise(y=data, sr=sr, y_noise=noise_sample)
# # #         return y_nr
# # #     except Exception:
# # #         return data

# # # def convert_audio_to_wav(input_path, output_path):
# # #     cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path]
# # #     subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

# # # def extract_audio_from_video(input_path, output_path):
# # #     cmd = ["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "mp3", "-ar", "16000", "-ac", "1", output_path]
# # #     subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

# # # def transcribe_audio(path):
# # #     try:
# # #         with open(path, "rb") as audio_file:
# # #             response = openai.Audio.transcribe(
# # #                 model="gpt-4o-mini-transcribe",
# # #                 file=audio_file
# # #             )
# # #         return response["text"]
# # #     except Exception as e:
# # #         print(f"❌ Transcription failed: {e}")
# # #         return ""

# # # def gpt_menu_match(transcribed_text, menu_df):
# # #     menu_str = ", ".join(menu_df["Item Name"].astype(str).tolist())
# # #     prompt = f"""
# # # Extract the food items and their quantities from the following spoken order:

# # # "{transcribed_text}"

# # # Only match items from this menu:
# # # {menu_str}

# # # Instructions:
# # # - If the input is in a non-English language (e.g., Hindi, Tamil, Telugu, Urdu), translate it to English before processing.
# # # - Ignore filler or address words like "bhaiya", "anna", "dada", "boss", or similar – they are not part of the order.
# # # - Identify food items and map them to the closest valid menu item, even if the names are partially spoken or mispronounced.
# # #   Examples: 
# # #   "Spoken Order: hot crispy chicken" → Original Menu item: "Hot & Crispy Chicken Bucket (6 pcs)", 
# # #   "Spoken Order: paneer popper" → "Original Menu item: Paneer Poppers", 
# # #   "Spoken Order: ginger burger" → "Original Menu item: Veg Zinger Burger", 
# # #   "Spoken Order: grilled zinger" → "Original Menu item: Tandoori Zinger Burger"
# # # - Use numeric digits for quantities (e.g., "two" → 2). If quantity is not stated, assume 1.
# # # - Do not return "NaN" or leave blanks — always return numeric values.
# # # - If the same item appears multiple times, combine them and total the quantity.
# # # - If the user stutters or changes their mind mid-sentence (e.g., says "aaannnwwww one dosa... annnn one onion dosa"), discard incomplete or earlier mentions and keep only the final, most specific version ("1 Onion Dosa"). Ignore false starts, repetitions, or abandoned phrases.
# # # - Only include items that exist in the menu. Do not hallucinate and invent or guess items not found.
# # # - Output should be a single clean, comma-separated list (no bullet points or extra text), e.g.:
# # #   1 Veg Zinger Burger, 1 Hot & Crispy Chicken Bucket (6 pcs), 1 Pepsi 500 ml
# # # - Format: "1 Item A, 2 Item B"
# # # """

# # #     try:
# # #         response = openai.ChatCompletion.create(
# # #             model="gpt-4o-mini",
# # #             messages=[{"role": "user", "content": prompt}],
# # #             temperature=0
# # #         )
# # #         return response['choices'][0]['message']['content'].strip()
# # #     except Exception:
# # #         return ""

# # # def parse_order(text, menu_df):
# # #     import re
# # #     from collections import defaultdict

# # #     response = gpt_menu_match(text, menu_df)
# # #     if not response:
# # #         return [], ""

# # #     # Normalize column names
# # #     df = menu_df.copy()
# # #     df.columns = [col.lower().strip() for col in df.columns]

# # #     # Map flexible column names
# # #     id_col = next((col for col in df.columns if col in ["id", "item id"]), None)
# # #     name_col = next((col for col in df.columns if "item" in col and "name" in col), None)
# # #     price_col = next((col for col in df.columns if "price" in col), None)

# # #     if not (id_col and name_col and price_col):
# # #         raise ValueError("Required columns (Id, Item Name, Price) not found in the menu.")

# # #     order = defaultdict(lambda: {"Id": "", "Quantity": 0, "Rate": 0, "Total": 0})
# # #     entries = [e.strip() for e in response.split(",") if e.strip()]

# # #     for entry in entries:
# # #         match = re.match(r"(\d+)\s+(.+)", entry)
# # #         if match:
# # #             quantity = int(match.group(1))
# # #             item_name = match.group(2).strip()
# # #             row = df[df[name_col].str.lower() == item_name.lower()]
# # #             if not row.empty:
# # #                 rate = int(row.iloc[0][price_col])
# # #                 item_id = int(row.iloc[0][id_col])
# # #                 order[item_name]["Id"] = item_id
# # #                 order[item_name]["Quantity"] += quantity
# # #                 order[item_name]["Rate"] = rate
# # #                 order[item_name]["Total"] += rate * quantity

# # #     parsed = [
# # #         {
# # #             "Item": k,
# # #             "Id": int(v["Id"]),
# # #             "Quantity": int(v["Quantity"]),
# # #             "Rate": int(v["Rate"]),
# # #             "Total": int(v["Total"])
# # #         }
# # #         for k, v in order.items()
# # #     ]
# # #     total = sum(v["Total"] for v in order.values())
# # #     if parsed:
# # #         parsed.append({
# # #             "Item": "Total Amount",
# # #             "Id": "",
# # #             "Quantity": "",
# # #             "Rate": "",
# # #             "Total": total
# # #         })

# # #     return parsed, response

# # # @app.post("/process/")
# # # async def process_audio_file(file: UploadFile = File(...)):
# # #     print("File received:", file.filename)
# # #     try:
# # #         ext = Path(file.filename).suffix.lower()

# # #         AUDIO_EXTENSIONS = {".wav", ".mp3", ".webm", ".ogg", ".m4a", ".mp4", ".flac", ".aac", ".aiff"}
# # #         VIDEO_EXTENSIONS = {".mp4", ".webm", ".m4v", ".avi"}

# # #         if ext in VIDEO_EXTENSIONS:
# # #             suffix = ext if ext else ".mp4"
# # #             with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as raw:
# # #                 raw.write(await file.read())
# # #                 raw_path = raw.name
# # #             audio_path = raw_path + "_audio.mp3"
# # #             extract_audio_from_video(raw_path, audio_path)

# # #         elif ext in AUDIO_EXTENSIONS:
# # #             suffix = ext if ext else ".webm"
# # #             with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as raw:
# # #                 raw.write(await file.read())
# # #                 raw_path = raw.name
# # #             audio_path = raw_path

# # #         else:
# # #             return {"error": f"Unsupported file extension: {ext}"}

# # #         # Convert to WAV
# # #         converted_path = audio_path + "_converted.wav"
# # #         convert_audio_to_wav(audio_path, converted_path)

# # #         # Read and denoise
# # #         data, samplerate = sf.read(converted_path)
# # #         if data.ndim > 1:
# # #             data = data[:, 0]
# # #         denoised = denoise_audio(data, samplerate)

# # #         # Save temp denoised file
# # #         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
# # #             sf.write(temp_audio.name, denoised, samplerate)

# # #             # Transcribe
# # #             transcription = transcribe_audio(temp_audio.name)

# # #             if menu_df.empty:
# # #                 return {"error": "No menu available. Please upload a menu CSV or ensure menu.csv is in server."}

# # #             parsed_order, response_text = parse_order(transcription, menu_df)

# # #         return {
# # #             "transcription": response_text,
# # #             "order": parsed_order
# # #         }

# # #     except Exception as e:
# # #         return {"error": f"Unexpected error: {e}"}
# # from fastapi import FastAPI, UploadFile, File
# # from fastapi.middleware.cors import CORSMiddleware
# # import tempfile
# # import numpy as np
# # import soundfile as sf
# # import noisereduce as nr
# # import openai
# # from dotenv import load_dotenv
# # import os
# # import pandas as pd
# # import re
# # import subprocess
# # from collections import defaultdict
# # from io import StringIO
# # from pathlib import Path
# # import string
# # # Load environment variables
# # load_dotenv()
# # openai.api_key = os.getenv("OPENAI_API_KEY")

# # app = FastAPI()

# # # CORS Middleware
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # Load default menu
# # menu_df = pd.DataFrame()
# # default_menu_path = "menu.csv"
# # if os.path.exists(default_menu_path):
# #     menu_df = pd.read_csv(default_menu_path)
# #     print(f"✅ Loaded default menu: {default_menu_path}")
# # else:
# #     print("⚠️ Default menu.csv not found. Please upload one via /upload-csv/")

# # @app.post("/upload-csv/")
# # async def upload_csv(csvFile: UploadFile = File(...)):
# #     try:
# #         contents = await csvFile.read()
# #         decoded_csv = contents.decode("utf-8")

# #         # Save to disk
# #         with open("menu.csv", "w", encoding="utf-8") as f:
# #             f.write(decoded_csv)

# #         # Load to memory
# #         global menu_df
# #         menu_df = pd.read_csv(StringIO(decoded_csv))

# #         return {
# #             "status": "success",
# #             "message": f"CSV file '{csvFile.filename}' uploaded successfully.",
# #             "rows": len(menu_df),
# #             "columns": list(menu_df.columns)
# #         }

# #     except Exception as e:
# #         return {"status": "error", "message": str(e)}

# # SUPPORTED_AUDIO_TYPES = {
# #     "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/webm",
# #     "audio/ogg", "audio/x-m4a", "audio/mp4", "audio/flac", "audio/aac",
# #     "audio/x-aiff"
# # }

# # SUPPORTED_VIDEO_TYPES = {
# #     "video/mp4", "video/webm", "video/x-m4v", "video/x-msvideo"
# # }

# # def denoise_audio(data, sr=16000):
# #     try:
# #         noise_sample = data[:sr] if len(data) >= sr else data
# #         if np.all(noise_sample == 0):
# #             return data
# #         y_nr = nr.reduce_noise(y=data, sr=sr, y_noise=noise_sample)
# #         return y_nr
# #     except Exception:
# #         return data

# # def convert_audio_to_wav(input_path, output_path):
# #     cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path]
# #     subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

# # def extract_audio_from_video(input_path, output_path):
# #     cmd = ["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "mp3", "-ar", "16000", "-ac", "1", output_path]
# #     subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

# # def transcribe_audio(path):
# #     try:
# #         with open(path, "rb") as audio_file:
# #             response = openai.Audio.transcribe(
# #                 model="gpt-4o-mini-transcribe",
# #                 file=audio_file
# #             )
# #         return response["text"]
# #     except Exception as e:
# #         print(f"❌ Transcription failed: {e}")
# #         return ""

# # def gpt_menu_match(transcribed_text, menu_df):
# #     menu_str = ", ".join(menu_df["Item Name"].astype(str).tolist())
# #     prompt = f"""
# # You are an AI voice assistant for a restaurant in Singapore. Your job is to extract food items and their quantities from a spoken order transcript.

# # Customer transcript:
# # \"\"\"{transcribed_text}\"\"\"

# # Menu items:
# # {menu_str}

# # Instructions:
# # 1. The customer may speak in English, Hindi, Tamil, Malay, or mix. Translate if necessary before processing.
# # 2. Match only the items listed in the menu. If a generic term is spoken (e.g., "vada"), choose the closest match from the menu (e.g., "Sambar Vada").
# # 3. Dish names may be incomplete or mispronounced. Identify the most likely correct dish using contextual understanding.
# # 4. Quantities can appear before or after the dish name (e.g., "2 dosa" or "dosa 2"). Use numeric digits in the result (e.g., "2").
# # 5. If quantity is not specified, default to 1.
# # 6. Combine duplicate dishes and sum the quantities.
# # 7. Ignore filler words or polite phrases like “bhaiya”, “anna”, “boss”, etc.
# # 8. Identify food items and map them to the closest valid menu item from the menu, even if the names are partially spoken, mispronounced, or translated. Use contextual understanding to resolve dish variations or abbreviations.
# #     Examples:
# #     - "Spoken Order: goreng rice" → Original Menu item: "Nasi Goreng"
# #     - "Spoken Order: egg prata" → Original Menu item: "Egg Paratha / Muttai Parotta"
# #     - "Spoken Order: sambhar wada" → Original Menu item: "Sambar Vada"
# #     - "Spoken Order: broth rice" → Original Menu item: "Koali Soaru (Chicken Broth Rice)"
# #     - "Spoken Order: cheese tikka" → Original Menu item: "Paneer Tikka"

# # 9. If a dish name is repeated or corrected mid-sentence, keep the last and most complete version.
# # 10. Do not hallucinate. If a dish is not in the menu, ignore it.
# # 11. Output must be in English only. Return item names exactly as they appear in the menu list (even if the spoken order used Hindi, Tamil, or any other script).
# # 12. Do not return dish names in Hindi or Tamil script — always use the English name.
# # 13. Output should be a single clean, comma-separated string like:
# #     1 Nasi Goreng, 1 Masala Dosa, 2 Sambar Vada

# # 14. Format: "<quantity> <matched dish name from menu>"
# # 15. Do not include any bullet points, headings, or extra explanation. Return only the clean string.
# # """

# #     try:
# #         response = openai.ChatCompletion.create(
# #             model="gpt-4o-mini",
# #             messages=[{"role": "user", "content": prompt}],
# #             temperature=0
# #         )
# #         return response['choices'][0]['message']['content'].strip()
# #     except Exception:
# #         return ""

# # def parse_order(text, menu_df):
# #     response = gpt_menu_match(text, menu_df)
# #     print("GPT RESPONSE:", response)
# #     if not response:
# #         return [], "", [], 0.0

# #     df = menu_df.copy()
# #     df.columns = [col.lower().strip() for col in df.columns]

# #     id_col = next((col for col in df.columns if col in ["id", "item id"]), None)
# #     name_col = next((col for col in df.columns if "item" in col and "name" in col), None)
# #     price_col = next((col for col in df.columns if "price" in col), None)

# #     if not (id_col and name_col and price_col):
# #         raise ValueError("Required columns (Id, Item Name, Price) not found in the menu.")

# #     # Build a lookup with all clean variants of item names
# #     menu_lookup = {}
# #     for _, row in df.iterrows():
# #         full_name = row[name_col].strip()
# #         lower_name = full_name.lower()
    
# #         # Create clean variants from slashes, parentheses, and base
# #         parts = re.split(r"/|\(|\)", lower_name)
# #         variants = [lower_name] + [p.strip() for p in parts if p.strip()]
    
# #         for variant in variants:
# #             cleaned_key = variant.translate(str.maketrans('', '', string.punctuation)).strip()
# #             if cleaned_key not in menu_lookup:
# #                 menu_lookup[cleaned_key] = {
# #                     "Id": int(row[id_col]),
# #                     "Rate": int(row[price_col]),
# #                     "Name": full_name  # original for display
# #                 }

# #     order = defaultdict(lambda: {"Id": "", "Quantity": 0, "Rate": 0, "Total": 0})
# #     unmatched_items = []
# #     entries = [e.strip() for e in response.split(",") if e.strip()]
# #     total_transcribed = len(entries)
# #     total_matched = 0

# #     for entry in entries:
# #         match = re.match(r"(\d+)\s+(.+)", entry.strip())
# #         if match:
# #             quantity = int(match.group(1))
# #             item_raw = match.group(2).strip().lower()
# #             item_name = item_raw.translate(str.maketrans('', '', string.punctuation)).strip()

# #             if item_name in menu_lookup:
# #                 item = menu_lookup[item_name]
# #                 order[item["Name"]]["Id"] = item["Id"]
# #                 order[item["Name"]]["Quantity"] += quantity
# #                 order[item["Name"]]["Rate"] = item["Rate"]
# #                 order[item["Name"]]["Total"] += item["Rate"] * quantity
# #                 total_matched += 1
# #             else:
# #                 unmatched_items.append(item_raw)
# #                 print(f"❌ No match for: {item_raw}")

# #     parsed = [
# #         {
# #             "Item": k,
# #             "Id": int(v["Id"]),
# #             "Quantity": int(v["Quantity"]),
# #             "Rate": int(v["Rate"]),
# #             "Total": int(v["Total"])
# #         }
# #         for k, v in order.items()
# #     ]

# #     total = sum(v["Total"] for v in order.values())
# #     if parsed:
# #         parsed.append({
# #             "Item": "Total Amount",
# #             "Id": "",
# #             "Quantity": "",
# #             "Rate": "",
# #             "Total": total
# #         })

# #     # Accuracy metric
# #     match_accuracy = round((total_matched / total_transcribed) * 100, 2) if total_transcribed > 0 else 0.0
# #     print(f"🔎 Match accuracy: {match_accuracy}% ({total_matched}/{total_transcribed})")
# #     print("Parsed order:", parsed)

# #     return parsed, response, unmatched_items, match_accuracy

# # @app.post("/process/")
# # async def process_audio_file(file: UploadFile = File(...)):
# #     print("File received:", file.filename)
# #     try:
# #         ext = Path(file.filename).suffix.lower()

# #         AUDIO_EXTENSIONS = {".wav", ".mp3", ".webm", ".ogg", ".m4a", ".mp4", ".flac", ".aac", ".aiff"}
# #         VIDEO_EXTENSIONS = {".mp4", ".webm", ".m4v", ".avi"}

# #         if ext in VIDEO_EXTENSIONS:
# #             suffix = ext if ext else ".mp4"
# #             with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as raw:
# #                 raw.write(await file.read())
# #                 raw_path = raw.name
# #             audio_path = raw_path + "_audio.mp3"
# #             extract_audio_from_video(raw_path, audio_path)

# #         elif ext in AUDIO_EXTENSIONS:
# #             suffix = ext if ext else ".webm"
# #             with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as raw:
# #                 raw.write(await file.read())
# #                 raw_path = raw.name
# #             audio_path = raw_path

# #         else:
# #             return {"error": f"Unsupported file extension: {ext}"}

# #         # Convert to WAV
# #         converted_path = audio_path + "_converted.wav"
# #         convert_audio_to_wav(audio_path, converted_path)

# #         # Read and denoise
# #         data, samplerate = sf.read(converted_path)
# #         if data.ndim > 1:
# #             data = data[:, 0]
# #         denoised = denoise_audio(data, samplerate)

# #         # Save temp denoised file
# #         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
# #             sf.write(temp_audio.name, denoised, samplerate)

# #             # Transcribe
# #             transcription = transcribe_audio(temp_audio.name)

# #         if menu_df.empty:
# #             return {"error": "No menu available. Please upload a menu CSV or ensure menu.csv is in server."}

# #         # ✅ FIX: corrected transcription variable name
# #         parsed_order, response, unmatched_items, match_accuracy = parse_order(transcription, menu_df)

# #         # print("Transcript:", transcription)
# #         # print("GPT Response:", response)
# #         # print("Parsed Order:", parsed_order)
# #         # print("Unmatched Items:", unmatched_items)
# #         # print("Match Accuracy:", match_accuracy)


# #         return {
# #             "transcription": transcription,
# #             "gpt_response": response,
# #             "order": parsed_order,
# #             "unmatched": unmatched_items,
# #             "accuracy": match_accuracy
# #         }

# #     except Exception as e:
# #         return {"error": f"Unexpected error: {e}"}
        
# # from fastapi import Request
# # from datetime import datetime
# # import json

# # @app.post("/feedback/")
# # async def receive_feedback(request: Request):
# #     data = await request.json()
# #     feedback_value = data.get("feedback", "")
# #     gpt_response = data.get("gpt_response", "")
# #     final_order = data.get("final_order", [])

# #     # Timestamp
# #     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# #     # Log entry
# #     log_entry = {
# #         "timestamp": timestamp,
# #         "feedback": feedback_value,
# #         "gpt_response": gpt_response,
# #         "final_order": final_order
# #     }

# #     # Append to log file
# #     with open("session_feedback.log", "a") as f:
# #         f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

# #     return {"message": "Feedback received"}


# # # Instructions:
# # # - If the input is in a non-English language (e.g., Hindi, Tamil, Telugu, Urdu, malay), translate it to English before processing.
# # # - Customers may speak in different languages or accents — including English, Hindi, Tamil, and Malay. Understand their pronunciation and return the   best-matching menu item.
# # # - Quantities can appear before or after the dish name (e.g., "2 Pav Bhaji", "Pav Bhaji 2") — both are valid.
# # # - Ignore filler or address words like "bhaiya", "anna", "dada", "boss", or similar – they are not part of the order.
# # # - Identify food items and map them to the closest valid menu item, even if the names are partially spoken or mispronounced.
# # #   Examples: 
# # #   "Spoken Order: hot crispy chicken" → Original Menu item: "Hot & Crispy Chicken Bucket (6 pcs)", 
# # #   "Spoken Order: paneer popper" → "Original Menu item: Paneer Poppers", 
# # #   "Spoken Order: ginger burger" → "Original Menu item: Veg Zinger Burger", 
# # #   "Spoken Order: grilled zinger" → "Original Menu item: Tandoori Zinger Burger"
# # # - Use numeric digits for quantities (e.g., "two" → 2). If quantity is not stated, assume 1.
# # # - If a quantity is mentioned **after** a dish name (e.g., "butter chicken 2"), associate it correctly.
# # # - If no quantity is mentioned, assume it is 1.
# # # - Do not return "NaN" or leave blanks — always return numeric values.
# # # - If the same item appears multiple times, combine them and total the quantity.
# # # - If the user stutters or changes their mind mid-sentence (e.g., says "aaannnwwww one dosa... annnn one onion dosa"), discard incomplete or earlier mentions and keep only the final, most specific version ("1 Onion Dosa"). Ignore false starts, repetitions, or abandoned phrases.
# # # - Only include items that exist in the menu. Do not hallucinate and invent or guess items not found.
# # # - Output should be strictly in english with a single clean, comma-separated list (no bullet points or extra text), e.g.:
# # #   1 Veg Zinger Burger, 1 Hot & Crispy Chicken Bucket (6 pcs), 1 Pepsi 500 ml.
# # # - Format: "1 Item A, 2 Item B"

# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# import tempfile
# import numpy as np
# import soundfile as sf
# import noisereduce as nr
# import openai
# from dotenv import load_dotenv
# import os
# import pandas as pd
# import re
# import subprocess
# from collections import defaultdict
# from io import StringIO
# from pathlib import Path
# import string
# # Load environment variables
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# app = FastAPI()

# # CORS Middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load default menu
# menu_df = pd.DataFrame()
# default_menu_path = "menu.csv"
# if os.path.exists(default_menu_path):
#     menu_df = pd.read_csv(default_menu_path)
#     print(f"✅ Loaded default menu: {default_menu_path}")
# else:
#     print("⚠️ Default menu.csv not found. Please upload one via /upload-csv/")

# @app.post("/upload-csv/")
# async def upload_csv(csvFile: UploadFile = File(...)):
#     try:
#         contents = await csvFile.read()
#         decoded_csv = contents.decode("utf-8")

#         # Save to disk
#         with open("menu.csv", "w", encoding="utf-8") as f:
#             f.write(decoded_csv)

#         # Load to memory
#         global menu_df
#         menu_df = pd.read_csv(StringIO(decoded_csv))

#         return {
#             "status": "success",
#             "message": f"CSV file '{csvFile.filename}' uploaded successfully.",
#             "rows": len(menu_df),
#             "columns": list(menu_df.columns)
#         }

#     except Exception as e:
#         return {"status": "error", "message": str(e)}

# SUPPORTED_AUDIO_TYPES = {
#     "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/webm",
#     "audio/ogg", "audio/x-m4a", "audio/mp4", "audio/flac", "audio/aac",
#     "audio/x-aiff"
# }

# SUPPORTED_VIDEO_TYPES = {
#     "video/mp4", "video/webm", "video/x-m4v", "video/x-msvideo"
# }

# def denoise_audio(data, sr=16000):
#     try:
#         noise_sample = data[:sr] if len(data) >= sr else data
#         if np.all(noise_sample == 0):
#             return data
#         y_nr = nr.reduce_noise(y=data, sr=sr, y_noise=noise_sample)
#         return y_nr
#     except Exception:
#         return data

# def convert_audio_to_wav(input_path, output_path):
#     cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path]
#     subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

# def extract_audio_from_video(input_path, output_path):
#     cmd = ["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "mp3", "-ar", "16000", "-ac", "1", output_path]
#     subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

# def transcribe_audio(path):
#     try:
#         with open(path, "rb") as audio_file:
#             response = openai.Audio.transcribe(
#                 model="gpt-4o-mini-transcribe",
#                 file=audio_file
#             )
#         return response["text"]
#     except Exception as e:
#         print(f"❌ Transcription failed: {e}")
#         return ""

# def gpt_menu_match(transcribed_text, menu_df):
#     menu_str = ", ".join(menu_df["Item Name"].astype(str).tolist())
#     prompt = f"""
# You are an AI voice assistant for a restaurant in Singapore. Your job is to extract food items and their quantities from a spoken order transcript.

# Customer transcript:
# \"\"\"{transcribed_text}\"\"\"

# Menu items:
# {menu_str}

# Instructions:
# 1. The customer may speak in English, Hindi, Tamil, Malay, or mix. Translate if necessary before processing.
# 2. Match only the items listed in the menu. If a generic term is spoken (e.g., "vada"), choose the closest match from the menu (e.g., "Sambar Vada").
# 3. Dish names may be incomplete or mispronounced. Identify the most likely correct dish using contextual understanding.
# 4. Quantities can appear before or after the dish name (e.g., "2 dosa" or "dosa 2"). Use numeric digits in the result (e.g., "2").
# 5. If quantity is not specified, default to 1.
# 6. Combine duplicate dishes and sum the quantities.
# 7. Ignore filler words or polite phrases like “bhaiya”, “anna”, “boss”, etc.
# 8. Identify food items and map them to the closest valid menu item from the menu, even if the names are partially spoken, mispronounced, or translated. Use contextual understanding to resolve dish variations or abbreviations.
#     Examples:
#     - "Spoken Order: goreng rice" → Original Menu item: "Nasi Goreng"
#     - "Spoken Order: egg prata" → Original Menu item: "Egg Paratha / Muttai Parotta"
#     - "Spoken Order: sambhar wada" → Original Menu item: "Sambar Vada"
#     - "Spoken Order: broth rice" → Original Menu item: "Koali Soaru (Chicken Broth Rice)"
#     - "Spoken Order: cheese tikka" → Original Menu item: "Paneer Tikka"

# 9. If a dish name is repeated or corrected mid-sentence, keep the last and most complete version.
# 10. Do not hallucinate. If a dish is not in the menu, ignore it.
# 11. Output must be in English only. Return item names exactly as they appear in the menu list (even if the spoken order used Hindi, Tamil, or any other script).
# 12. Do not return dish names in Hindi or Tamil script — always use the English name.
# 13. Output should be a single clean, comma-separated string like:
#     1 Nasi Goreng, 1 Masala Dosa, 2 Sambar Vada

# 14. Format: "<quantity> <matched dish name from menu>"
# 15. Do not include any bullet points, headings, or extra explanation. Return only the clean string.
# """

#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0
#         )
#         return response['choices'][0]['message']['content'].strip()
#     except Exception:
#         return ""

# def parse_order(text, menu_df):
#     response = gpt_menu_match(text, menu_df)
#     print("GPT RESPONSE:", response)
#     if not response:
#         return [], "", [], 0.0

#     df = menu_df.copy()
#     df.columns = [col.lower().strip() for col in df.columns]

#     id_col = next((col for col in df.columns if col in ["id", "item id"]), None)
#     name_col = next((col for col in df.columns if "item" in col and "name" in col), None)
#     price_col = next((col for col in df.columns if "price" in col), None)

#     if not (id_col and name_col and price_col):
#         raise ValueError("Required columns (Id, Item Name, Price) not found in the menu.")

#     # Build a lookup with all clean variants of item names
#     menu_lookup = {}
#     for _, row in df.iterrows():
#         full_name = row[name_col].strip()
#         lower_name = full_name.lower()
    
#         # Create clean variants from slashes, parentheses, and base
#         parts = re.split(r"/|\(|\)", lower_name)
#         variants = [lower_name] + [p.strip() for p in parts if p.strip()]
    
#         for variant in variants:
#             cleaned_key = variant.translate(str.maketrans('', '', string.punctuation)).strip()
#             if cleaned_key not in menu_lookup:
#                 menu_lookup[cleaned_key] = {
#                     "Id": int(row[id_col]),
#                     "Rate": int(row[price_col]),
#                     "Name": full_name  # original for display
#                 }

#     order = defaultdict(lambda: {"Id": "", "Quantity": 0, "Rate": 0, "Total": 0})
#     unmatched_items = []
#     entries = [e.strip() for e in response.split(",") if e.strip()]
#     total_transcribed = len(entries)
#     total_matched = 0

#     for entry in entries:
#         match = re.match(r"(\d+)\s+(.+)", entry.strip())
#         if match:
#             quantity = int(match.group(1))
#             item_raw = match.group(2).strip().lower()
#             item_name = item_raw.translate(str.maketrans('', '', string.punctuation)).strip()

#             if item_name in menu_lookup:
#                 item = menu_lookup[item_name]
#                 order[item["Name"]]["Id"] = item["Id"]
#                 order[item["Name"]]["Quantity"] += quantity
#                 order[item["Name"]]["Rate"] = item["Rate"]
#                 order[item["Name"]]["Total"] += item["Rate"] * quantity
#                 total_matched += 1
#             else:
#                 unmatched_items.append(item_raw)
#                 print(f"❌ No match for: {item_raw}")

#     parsed = [
#         {
#             "Item": k,
#             "Id": int(v["Id"]),
#             "Quantity": int(v["Quantity"]),
#             "Rate": int(v["Rate"]),
#             "Total": int(v["Total"])
#         }
#         for k, v in order.items()
#     ]

#     total = sum(v["Total"] for v in order.values())
#     if parsed:
#         parsed.append({
#             "Item": "Total Amount",
#             "Id": "",
#             "Quantity": "",
#             "Rate": "",
#             "Total": total
#         })

#     # Accuracy metric
#     match_accuracy = round((total_matched / total_transcribed) * 100, 2) if total_transcribed > 0 else 0.0
#     print(f"🔎 Match accuracy: {match_accuracy}% ({total_matched}/{total_transcribed})")
#     print("Parsed order:", parsed)

#     return parsed, response, unmatched_items, match_accuracy

# # @app.post("/process/")
# # async def process_audio_file(file: UploadFile = File(...)):
# #     print("File received:", file.filename)
# #     try:
# #         ext = Path(file.filename).suffix.lower()

# #         AUDIO_EXTENSIONS = {".wav", ".mp3", ".webm", ".ogg", ".m4a", ".mp4", ".flac", ".aac", ".aiff"}
# #         VIDEO_EXTENSIONS = {".mp4", ".webm", ".m4v", ".avi"}

# #         if ext in VIDEO_EXTENSIONS:
# #             suffix = ext if ext else ".mp4"
# #             with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as raw:
# #                 raw.write(await file.read())
# #                 raw_path = raw.name
# #             audio_path = raw_path + "_audio.mp3"
# #             extract_audio_from_video(raw_path, audio_path)

# #         elif ext in AUDIO_EXTENSIONS:
# #             suffix = ext if ext else ".webm"
# #             with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as raw:
# #                 raw.write(await file.read())
# #                 raw_path = raw.name
# #             audio_path = raw_path

# #         else:
# #             return {"error": f"Unsupported file extension: {ext}"}

# #         # Convert to WAV
# #         converted_path = audio_path + "_converted.wav"
# #         convert_audio_to_wav(audio_path, converted_path)

# #         # Read and denoise
# #         data, samplerate = sf.read(converted_path)
# #         if data.ndim > 1:
# #             data = data[:, 0]
# #         denoised = denoise_audio(data, samplerate)

# #         # Save temp denoised file
# #         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
# #             sf.write(temp_audio.name, denoised, samplerate)

# #             # Transcribe
# #             transcription = transcribe_audio(temp_audio.name)

# #         if menu_df.empty:
# #             return {"error": "No menu available. Please upload a menu CSV or ensure menu.csv is in server."}

# #         # ✅ FIX: corrected transcription variable name
# #         parsed_order, response, unmatched_items, match_accuracy = parse_order(transcription, menu_df)

# #         # print("Transcript:", transcription)
# #         # print("GPT Response:", response)
# #         # print("Parsed Order:", parsed_order)
# #         # print("Unmatched Items:", unmatched_items)
# #         # print("Match Accuracy:", match_accuracy)


# #         return {
# #             "transcription": transcription,
# #             "gpt_response": response,
# #             "order": parsed_order,
# #             "unmatched": unmatched_items,
# #             "accuracy": match_accuracy
# #         }

# #     except Exception as e:
# #         return {"error": f"Unexpected error: {e}"}
        
# # from fastapi import Request
# # from datetime import datetime
# # import json
# from fastapi import UploadFile, File, Form, Request
# from pathlib import Path
# import os, tempfile, shutil
# import json
# from datetime import datetime
# from uuid import uuid4

# # @app.post("/process/")
# # async def process_audio_file(file: UploadFile = File(...), session_id: str = Form(...)):
# #     print("File received:", file.filename, "Session ID:", session_id)

# #     try:
# #         ext = Path(file.filename).suffix.lower()
# #         AUDIO_EXTENSIONS = {".wav", ".mp3", ".webm", ".ogg", ".m4a", ".mp4", ".flac", ".aac", ".aiff"}
# #         VIDEO_EXTENSIONS = {".mp4", ".webm", ".m4v", ".avi"}

# #         session_dir = os.path.join("audio", session_id)
# #         os.makedirs(session_dir, exist_ok=True)

# #         # Save raw uploaded file to session folder
# #         suffix = ext if ext else ".webm"
# #         filename = f"{uuid4().hex}{ext}"  # 🔥 generate unique name
# #         raw_path = os.path.join(session_dir, filename)
# #         with open(raw_path, "wb") as out_file:
# #             out_file.write(await file.read())

# #         # Convert video to audio if needed
# #         if ext in VIDEO_EXTENSIONS:
# #             audio_path = raw_path + "_audio.mp3"
# #             extract_audio_from_video(raw_path, audio_path)
# #         elif ext in AUDIO_EXTENSIONS:
# #             audio_path = raw_path
# #         else:
# #             return {"error": f"Unsupported file extension: {ext}"}

# #         # Convert to WAV
# #         converted_path = audio_path + "_converted.wav"
# #         convert_audio_to_wav(audio_path, converted_path)

# #         # Denoise
# #         data, samplerate = sf.read(converted_path)
# #         if data.ndim > 1:
# #             data = data[:, 0]
# #         denoised = denoise_audio(data, samplerate)

# #         # Save denoised audio to temp WAV file
# #         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
# #             sf.write(temp_audio.name, denoised, samplerate)
# #             transcription = transcribe_audio(temp_audio.name)

# #         if menu_df.empty:
# #             return {"error": "No menu available."}

# #         parsed_order, response, unmatched_items, match_accuracy = parse_order(transcription, menu_df)

# #         return {
# #             "session_id": session_id,
# #             "transcription": transcription,
# #             "gpt_response": response,
# #             "order": parsed_order,
# #             "unmatched": unmatched_items,
# #             "accuracy": match_accuracy
# #         }

# #     except Exception as e:
# #         return {"error": f"Unexpected error: {e}"}
# from uuid import uuid4
# import os, tempfile
# from pathlib import Path
# import soundfile as sf
# SESSION_STATE = {}  

# @app.post("/process/")
# async def process_audio_file(file: UploadFile = File(...), session_id: str = Form(...)):
#     print("File received:", file.filename, "Session ID:", session_id)

#     try:
#         ext = Path(file.filename).suffix.lower()
#         AUDIO_EXTENSIONS = {".wav", ".mp3", ".webm", ".ogg", ".m4a", ".mp4", ".flac", ".aac", ".aiff"}
#         VIDEO_EXTENSIONS = {".mp4", ".webm", ".m4v", ".avi"}

#         session_dir = os.path.join("audio", session_id)
#         os.makedirs(session_dir, exist_ok=True)

#         # Generate unique file name
#         base_name = uuid4().hex
#         raw_temp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
#         raw_temp.write(await file.read())
#         raw_temp.close()

#         # Extract or pass through audio
#         if ext in VIDEO_EXTENSIONS:
#             audio_path = raw_temp.name + "_audio.mp3"
#             extract_audio_from_video(raw_temp.name, audio_path)
#         elif ext in AUDIO_EXTENSIONS:
#             audio_path = raw_temp.name
#         else:
#             os.unlink(raw_temp.name)
#             return {"error": f"Unsupported file extension: {ext}"}

#         # Convert to WAV
#         converted_path = audio_path + "_converted.wav"
#         convert_audio_to_wav(audio_path, converted_path)

#         # Denoise
#         data, samplerate = sf.read(converted_path)
#         if data.ndim > 1:
#             data = data[:, 0]
#         denoised = denoise_audio(data, samplerate)

#         # Save only the denoised WAV file to session folder
#         final_denoised_path = os.path.join(session_dir, f"{base_name}_denoised.wav")
#         sf.write(final_denoised_path, denoised, samplerate)

#         # Clean up all intermediate temp files
#         for path in [raw_temp.name, audio_path, converted_path]:
#             if os.path.exists(path):
#                 os.remove(path)

#         # Transcribe
#         transcription = transcribe_audio(final_denoised_path)

#         if menu_df.empty:
#             return {"error": "No menu available."}
#         if session_id not in SESSION_STATE:
#             SESSION_STATE[session_id] = {"responses": []}

#         parsed_order, response, unmatched_items, match_accuracy = parse_order(transcription, menu_df)
#         SESSION_STATE[session_id]["responses"].append(response)

#         return {
#             "session_id": session_id,
#             "transcription": transcription,
#             "gpt_response": response,
#             "order": parsed_order,
#             "unmatched": unmatched_items,
#             "accuracy": match_accuracy
#         }

#     except Exception as e:
#         return {"error": f"Unexpected error: {e}"}

# @app.post("/feedback/")
# async def receive_feedback(request: Request):
#     data = await request.json()
#     session_id = data.get("session_id")
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     audio_files = []
#     if session_id:
#         session_dir = os.path.join("audio", session_id)
#         if os.path.exists(session_dir):
#             audio_files = os.listdir(session_dir)

#     feedback_log = {
#         "timestamp": timestamp,
#         "session_id": session_id,
#         "feedback": data.get("feedback", ""),
#         "issues": data.get("issues", []),
#         "gpt_response": SESSION_STATE.get(session_id, {}).get("responses", []),
#         "final_order": data.get("final_order", []),
#         "audio_files": audio_files
#     }

#     print("\nReceived Feedback:")
#     print(json.dumps(feedback_log, indent=4))

#     with open("session_feedback.log", "a") as f:
#         f.write(json.dumps(feedback_log) + "\n")

#     return {"message": "Feedback received"}

# # @app.post("/feedback/")
# # async def receive_feedback(request: Request):
    
# #     data = await request.json()
# #     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# #     feedback_log = {
# #         "timestamp": timestamp,
# #         "feedback": data.get("feedback", ""),
# #         "issues": data.get("issues", []),
# #         "gpt_response": data.get("gpt_response", ""),
# #         "final_order": data.get("final_order", [])
# #     }
# #     print("\nReceived Feedback:")
# #     print(json.dumps(feedback_log, indent=4))  # pretty print

# #     with open("session_feedback.log", "a") as f:
# #         f.write(json.dumps(feedback_log) + "\n")

# #     return {"message": "Feedback received"}


# # Instructions:
# # - If the input is in a non-English language (e.g., Hindi, Tamil, Telugu, Urdu, malay), translate it to English before processing.
# # - Customers may speak in different languages or accents — including English, Hindi, Tamil, and Malay. Understand their pronunciation and return the   best-matching menu item.
# # - Quantities can appear before or after the dish name (e.g., "2 Pav Bhaji", "Pav Bhaji 2") — both are valid.
# # - Ignore filler or address words like "bhaiya", "anna", "dada", "boss", or similar – they are not part of the order.
# # - Identify food items and map them to the closest valid menu item, even if the names are partially spoken or mispronounced.
# #   Examples: 
# #   "Spoken Order: hot crispy chicken" → Original Menu item: "Hot & Crispy Chicken Bucket (6 pcs)", 
# #   "Spoken Order: paneer popper" → "Original Menu item: Paneer Poppers", 
# #   "Spoken Order: ginger burger" → "Original Menu item: Veg Zinger Burger", 
# #   "Spoken Order: grilled zinger" → "Original Menu item: Tandoori Zinger Burger"
# # - Use numeric digits for quantities (e.g., "two" → 2). If quantity is not stated, assume 1.
# # - If a quantity is mentioned **after** a dish name (e.g., "butter chicken 2"), associate it correctly.
# # - If no quantity is mentioned, assume it is 1.
# # - Do not return "NaN" or leave blanks — always return numeric values.
# # - If the same item appears multiple times, combine them and total the quantity.
# # - If the user stutters or changes their mind mid-sentence (e.g., says "aaannnwwww one dosa... annnn one onion dosa"), discard incomplete or earlier mentions and keep only the final, most specific version ("1 Onion Dosa"). Ignore false starts, repetitions, or abandoned phrases.
# # - Only include items that exist in the menu. Do not hallucinate and invent or guess items not found.
# # - Output should be strictly in english with a single clean, comma-separated list (no bullet points or extra text), e.g.:
# #   1 Veg Zinger Burger, 1 Hot & Crispy Chicken Bucket (6 pcs), 1 Pepsi 500 ml.
# # - Format: "1 Item A, 2 Item B"


# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# import tempfile
# import numpy as np
# import soundfile as sf
# import noisereduce as nr
# import openai
# from dotenv import load_dotenv
# import os
# import pandas as pd
# import re
# import subprocess
# from collections import defaultdict
# from io import StringIO
# from pathlib import Path
# import string
# # Load environment variables
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# app = FastAPI()

# # CORS Middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load default menu
# menu_df = pd.DataFrame()
# default_menu_path = "menu.csv"
# if os.path.exists(default_menu_path):
#     menu_df = pd.read_csv(default_menu_path)
#     print(f"✅ Loaded default menu: {default_menu_path}")
# else:
#     print("⚠️ Default menu.csv not found. Please upload one via /upload-csv/")

# @app.post("/upload-csv/")
# async def upload_csv(csvFile: UploadFile = File(...)):
#     try:
#         contents = await csvFile.read()
#         decoded_csv = contents.decode("utf-8")

#         # Save to disk
#         with open("menu.csv", "w", encoding="utf-8") as f:
#             f.write(decoded_csv)

#         # Load to memory
#         global menu_df
#         menu_df = pd.read_csv(StringIO(decoded_csv))

#         return {
#             "status": "success",
#             "message": f"CSV file '{csvFile.filename}' uploaded successfully.",
#             "rows": len(menu_df),
#             "columns": list(menu_df.columns)
#         }

#     except Exception as e:
#         return {"status": "error", "message": str(e)}

# SUPPORTED_AUDIO_TYPES = {
#     "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/webm",
#     "audio/ogg", "audio/x-m4a", "audio/mp4", "audio/flac", "audio/aac",
#     "audio/x-aiff"
# }

# SUPPORTED_VIDEO_TYPES = {
#     "video/mp4", "video/webm", "video/x-m4v", "video/x-msvideo"
# }

# def denoise_audio(data, sr=16000):
#     try:
#         noise_sample = data[:sr] if len(data) >= sr else data
#         if np.all(noise_sample == 0):
#             return data
#         y_nr = nr.reduce_noise(y=data, sr=sr, y_noise=noise_sample)
#         return y_nr
#     except Exception:
#         return data

# def convert_audio_to_wav(input_path, output_path):
#     cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path]
#     subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

# def extract_audio_from_video(input_path, output_path):
#     cmd = ["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "mp3", "-ar", "16000", "-ac", "1", output_path]
#     subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

# def transcribe_audio(path):
#     try:
#         with open(path, "rb") as audio_file:
#             response = openai.Audio.transcribe(
#                 model="gpt-4o-mini-transcribe",
#                 file=audio_file
#             )
#         return response["text"]
#     except Exception as e:
#         print(f"❌ Transcription failed: {e}")
#         return ""

# def gpt_menu_match(transcribed_text, menu_df):
#     menu_str = ", ".join(menu_df["Item Name"].astype(str).tolist())
#     prompt = f"""
# You are an AI voice assistant for a restaurant in Singapore. Your job is to extract food items and their quantities from a spoken order transcript.

# Customer transcript:
# \"\"\"{transcribed_text}\"\"\"

# Menu items:
# {menu_str}

# Instructions:
# 1. First, translate the entire transcript to English 
# 2. Then extract the food items and their quantities from the translated order.
# 3. Match only the items listed in the menu. If a generic term is spoken (e.g., "vada"), choose the closest match from the menu (e.g., "Sambar Vada").
# 4. Dish names may be incomplete or mispronounced. Identify the most likely correct dish using contextual understanding.
# 5. Quantities can appear before or after the dish name (e.g., "2 dosa" or "dosa 2"). Use numeric digits in the result (e.g., "2").
# 6. If quantity is not specified, default to 1.
# 7. Combine duplicate dishes and sum the quantities.
# 8. Ignore filler words or polite phrases like “bhaiya”, “anna”, “boss”, etc.
# 9. Identify food items and map them to the closest valid menu item from the menu, even if the names are partially spoken, mispronounced, or translated. Use contextual understanding to resolve dish variations or abbreviations.
#     Examples:
#     - "Spoken Order: goreng rice" → Original Menu item: "Nasi Goreng"
#     - "Spoken Order: egg prata" → Original Menu item: "Egg Paratha / Muttai Parotta"
#     - "Spoken Order: sambhar wada" → Original Menu item: "Sambar Vada"
#     - "Spoken Order: broth rice" → Original Menu item: "Koali Soaru (Chicken Broth Rice)"
#     - "Spoken Order: cheese tikka" → Original Menu item: "Paneer Tikka"

# 10. If a dish name is repeated or corrected mid-sentence, keep the last and most complete version.
# 11. Do not hallucinate. If a dish is not in the menu, ignore it.
# 12. Output must be in English only. Return item names exactly as they appear in the menu list (even if the spoken order used Hindi, Tamil, or any other script).
# 13. Do not return dish names in Hindi or Tamil script — always use the English name.
# 14. Output should be a single clean, comma-separated list (no bullet points or extra text), in the following format:
#   ` item_name | quantity`
#   Example: ` Veg Zinger Burger | 1,  Hot & Crispy Chicken Bucket (6 pcs) | 2, Pepsi 500 ml | 1`

# """

#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0
#         )
#         return response['choices'][0]['message']['content'].strip()
#     except Exception:
#         return ""

# def parse_order(text, menu_df):
#     response = gpt_menu_match(text, menu_df)
#     print("GPT RESPONSE:", response)
#     if not response:
#         return [], "", [], 0.0

#     df = menu_df.copy()
#     df.columns = [col.lower().strip() for col in df.columns]

#     id_col = next((col for col in df.columns if col in ["id", "item id"]), None)
#     name_col = next((col for col in df.columns if "item" in col and "name" in col), None)
#     price_col = next((col for col in df.columns if "price" in col), None)

#     if not (id_col and name_col and price_col):
#         raise ValueError("Required columns (Id, Item Name, Price) not found in the menu.")

#     # Build a lookup with all clean variants of item names
#     menu_lookup = {}
#     for _, row in df.iterrows():
#         full_name = row[name_col].strip()
#         lower_name = full_name.lower()
    
#         # Create clean variants from slashes, parentheses, and base
#         parts = re.split(r"/|\(|\)", lower_name)
#         variants = [lower_name] + [p.strip() for p in parts if p.strip()]
    
#         for variant in variants:
#             cleaned_key = variant.translate(str.maketrans('', '', string.punctuation)).strip()
#             if cleaned_key not in menu_lookup:
#                 menu_lookup[cleaned_key] = {
#                     "Id": int(row[id_col]),
#                     "Rate": int(row[price_col]),
#                     "Name": full_name  # original for display
#                 }

#     order = defaultdict(lambda: {"Id": "", "Quantity": 0, "Rate": 0, "Total": 0})
#     unmatched_items = []
#     entries = [e.strip() for e in response.split(",") if e.strip()]
#     total_transcribed = len(entries)
#     total_matched = 0

#     for entry in entries:
#         if "|" in entry:
#             item_part, qty_part = map(str.strip, entry.split("|", 1))
#             item_raw = item_part.lower()
#             item_name = item_raw.translate(str.maketrans('', '', string.punctuation)).strip()

#             try:
#                 quantity = int(qty_part)
#             except ValueError:
#                 quantity = 1  # fallback in case of bad quantity

#             if item_name in menu_lookup:
#                 item = menu_lookup[item_name]
#                 order[item["Name"]]["Id"] = item["Id"]
#                 order[item["Name"]]["Quantity"] += quantity
#                 order[item["Name"]]["Rate"] = item["Rate"]
#                 order[item["Name"]]["Total"] += item["Rate"] * quantity
#                 total_matched += 1
#             else:
#                 unmatched_items.append(item_raw)
#                 print(f"❌ No match for: {item_raw}")
#         else:
#             # ❌ If format is invalid or missing '|', add entire entry as unmatched
#             unmatched_items.append(entry.strip())
#             print(f"❌ Bad format or missing '|': {entry.strip()}")

#     parsed = [
#         {
#             "Item": k,
#             "Id": int(v["Id"]),
#             "Quantity": int(v["Quantity"]),
#             "Rate": int(v["Rate"]),
#             "Total": int(v["Total"])
#         }
#         for k, v in order.items()
#     ]

#     total = sum(v["Total"] for v in order.values())
#     if parsed:
#         parsed.append({
#             "Item": "Total Amount",
#             "Id": "",
#             "Quantity": "",
#             "Rate": "",
#             "Total": total
#         })

#     # Accuracy metric
#     match_accuracy = round((total_matched / total_transcribed) * 100, 2) if total_transcribed > 0 else 0.0
#     print(f"🔎 Match accuracy: {match_accuracy}% ({total_matched}/{total_transcribed})")
#     print("Parsed order:", parsed)

#     return parsed, response, unmatched_items, match_accuracy

# from fastapi import UploadFile, File, Form, Request
# from pathlib import Path
# import os, tempfile, shutil
# import json
# from datetime import datetime
# from uuid import uuid4
# import soundfile as sf
# SESSION_STATE = {}  

# @app.post("/process/")
# async def process_audio_file(file: UploadFile = File(...), session_id: str = Form(...)):
#     print("File received:", file.filename, "Session ID:", session_id)

#     try:
#         ext = Path(file.filename).suffix.lower()
#         AUDIO_EXTENSIONS = {".wav", ".mp3", ".webm", ".ogg", ".m4a", ".mp4", ".flac", ".aac", ".aiff"}
#         VIDEO_EXTENSIONS = {".mp4", ".webm", ".m4v", ".avi"}

#         session_dir = os.path.join("audio", session_id)
#         os.makedirs(session_dir, exist_ok=True)

#         # Generate unique file name
#         base_name = uuid4().hex
#         raw_temp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
#         raw_temp.write(await file.read())
#         raw_temp.close()

#         # Extract or pass through audio
#         if ext in VIDEO_EXTENSIONS:
#             audio_path = raw_temp.name + "_audio.mp3"
#             extract_audio_from_video(raw_temp.name, audio_path)
#         elif ext in AUDIO_EXTENSIONS:
#             audio_path = raw_temp.name
#         else:
#             os.unlink(raw_temp.name)
#             return {"error": f"Unsupported file extension: {ext}"}

#         # Convert to WAV
#         converted_path = audio_path + "_converted.wav"
#         convert_audio_to_wav(audio_path, converted_path)

#         # Denoise
# # Save the original uploaded audio (converted to WAV) to session folder
#         final_original_path = os.path.join(session_dir, f"{base_name}_original.wav")
#         shutil.copyfile(converted_path, final_original_path)

#         # Denoise and use only for transcription (not saved)
#         data, samplerate = sf.read(converted_path)
#         if data.ndim > 1:
#             data = data[:, 0]
#         denoised = denoise_audio(data, samplerate)

#         # Save to a temporary file just for transcription
#         final_denoised_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
#         sf.write(final_denoised_path, denoised, samplerate)

#         # Clean up all intermediate temp files
#         for path in [raw_temp.name, audio_path, converted_path]:
#             if os.path.exists(path):
#                 os.remove(path)

#         # Transcribe
#         transcription = transcribe_audio(final_denoised_path)

#         if menu_df.empty:
#             return {"error": "No menu available."}
#         # if session_id not in SESSION_STATE:
#         #     SESSION_STATE[session_id] = {"responses": []}

#         # parsed_order, response, unmatched_items, match_accuracy = parse_order(transcription, menu_df)
#         # SESSION_STATE[session_id]["responses"].append(response)
#         if session_id not in SESSION_STATE:
#             SESSION_STATE[session_id] = {
#                 "responses": [],
#                 "transcription": []
#             }
        
#         parsed_order, response, unmatched_items, match_accuracy = parse_order(transcription, menu_df)
#         SESSION_STATE[session_id]["responses"].append(response)
#         SESSION_STATE[session_id]["transcriptions"].append(transcription)

#         return {
#             "session_id": session_id,
#             "transcription": transcription,
#             "gpt_response": response,
#             "order": parsed_order,
#             "unmatched": unmatched_items,
#             "accuracy": match_accuracy
#         }

#     except Exception as e:
#         return {"error": f"Unexpected error: {e}"}

# @app.post("/feedback/")
# async def receive_feedback(request: Request):
#     data = await request.json()
#     session_id = data.get("session_id")
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     audio_files = []
#     if session_id:
#         session_dir = os.path.join("audio", session_id)
#         if os.path.exists(session_dir):
#             audio_files = os.listdir(session_dir)

#     feedback_log = {
#         "timestamp": timestamp,
#         "session_id": session_id,
#         "transcription": SESSION_STATE.get(session_id, {}).get("transcription", []),
#         "feedback": data.get("feedback", ""),
#         "issues": data.get("issues", []),
#         "gpt_response": SESSION_STATE.get(session_id, {}).get("responses", []),
#         "final_order": data.get("final_order", []),
#         "audio_files": audio_files
#     }

#     print("\nReceived Feedback:")
#     print(json.dumps(feedback_log, indent=4))

#     with open("session_feedback.log", "a") as f:
#         f.write(json.dumps(feedback_log) + "\n")

#     return {"message": "Feedback received"}



# # Instructions:
# # - If the input is in a non-English language (e.g., Hindi, Tamil, Telugu, Urdu, malay), translate it to English before processing.
# # - Customers may speak in different languages or accents — including English, Hindi, Tamil, and Malay. Understand their pronunciation and return the   best-matching menu item.
# # - Quantities can appear before or after the dish name (e.g., "2 Pav Bhaji", "Pav Bhaji 2") — both are valid.
# # - Ignore filler or address words like "bhaiya", "anna", "dada", "boss", or similar – they are not part of the order.
# # - Identify food items and map them to the closest valid menu item, even if the names are partially spoken or mispronounced.
# #   Examples: 
# #   "Spoken Order: hot crispy chicken" → Original Menu item: "Hot & Crispy Chicken Bucket (6 pcs)", 
# #   "Spoken Order: paneer popper" → "Original Menu item: Paneer Poppers", 
# #   "Spoken Order: ginger burger" → "Original Menu item: Veg Zinger Burger", 
# #   "Spoken Order: grilled zinger" → "Original Menu item: Tandoori Zinger Burger"
# # - Use numeric digits for quantities (e.g., "two" → 2). If quantity is not stated, assume 1.
# # - If a quantity is mentioned **after** a dish name (e.g., "butter chicken 2"), associate it correctly.
# # - If no quantity is mentioned, assume it is 1.
# # - Do not return "NaN" or leave blanks — always return numeric values.
# # - If the same item appears multiple times, combine them and total the quantity.
# # - If the user stutters or changes their mind mid-sentence (e.g., says "aaannnwwww one dosa... annnn one onion dosa"), discard incomplete or earlier mentions and keep only the final, most specific version ("1 Onion Dosa"). Ignore false starts, repetitions, or abandoned phrases.
# # - Only include items that exist in the menu. Do not hallucinate and invent or guess items not found.
# # - Output should be strictly in english with a single clean, comma-separated list (no bullet points or extra text), e.g.:
# #   1 Veg Zinger Burger, 1 Hot & Crispy Chicken Bucket (6 pcs), 1 Pepsi 500 ml.
# # - Format: "1 Item A, 2 Item B"
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
import string
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
    print("⚠️ Default menu.csv not found. Please upload one via /upload-csv/")

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

def convert_audio_to_wav(input_path, output_path):
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def extract_audio_from_video(input_path, output_path):
    cmd = ["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "mp3", "-ar", "16000", "-ac", "1", output_path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

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
    menu_str = ", ".join(menu_df["Item Name"].astype(str).tolist())
    prompt = f"""
You are an AI voice assistant for a restaurant in Singapore. Your job is to extract food items and their quantities from a spoken order transcript.

Customer transcript:
\"\"\"{transcribed_text}\"\"\"

Menu items:
{menu_str}

Instructions:
1. The customer may speak in English, Hindi, Tamil, Malay, or a mix. Translate if necessary before processing.
2. Match only the items listed in the menu. If a generic or vague term is spoken (e.g., "vada"), select the closest match from the menu (e.g., "Sambar Vada").
3. Dish names may be incomplete or mispronounced. Identify the most likely correct dish using contextual understanding.
4. Quantities can appear before or after the dish name (e.g., "2 dosa" or "dosa 2"). Always use numeric digits (e.g., "2").
5. If quantity is not specified, assume it is 1.
6. Combine duplicate dishes and sum the quantities.
7. Ignore filler or polite words like “bhaiya”, “anna”, “boss”, “please”, etc.
8. Match spoken items to the closest valid menu item using contextual understanding, even if the names are abbreviated, partially spoken, or mispronounced.
   Examples:
   - "Spoken Order: goreng rice" → Matched Menu Item: "Nasi Goreng"
   - "Spoken Order: egg prata" → "Egg Paratha / Muttai Parotta"
   - "Spoken Order: sambhar wada" → "Sambar Vada"
   - "Spoken Order: broth rice" → "Koali Soaru (Chicken Broth Rice)"
   - "Spoken Order: cheese tikka" → "Paneer Tikka"
9. If the customer repeats or corrects an item mid-sentence, use only the last and most complete version.
10. Do not hallucinate or invent items. If the item is not in the menu, ignore it completely.
11. Output should only include items that exist in the menu. Do not output the full menu under any condition — especially if the audio is blank, noisy, or from a movie.
12. If no valid food items are identified in the spoken text, return an empty item list. Do not guess.
13. Output must be in English only. Return item names exactly as they appear in the menu list.
14. Do not return dish names in Hindi, Tamil, or any other script — only use the English names from the menu.
15. Format the output exactly like this (no extra text or explanation):

✅ Format:
Translated Transcript: <translated_text_here>
Matched Items: item_id | item_name | quantity, item_id | item_name | quantity, ...

✅ Example:
Translated Transcript: I would like one Veg Zinger Burger and two Pepsi
Matched Items: 101 | Veg Zinger Burger | 1, 110 | Pepsi 500 ml | 2
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        output = response['choices'][0]['message']['content'].strip()

        # Parse the translated text and matched items from the response
        translated_text = ""
        matched_items = ""
        for line in output.splitlines():
            if line.lower().startswith("translated transcript:"):
                translated_text = line.split(":", 1)[1].strip()
            elif line.lower().startswith("matched items:"):
                matched_items = line.split(":", 1)[1].strip()

        return translated_text, matched_items
    except Exception as e:
        return "", ""

# def parse_order(text, menu_df):
#     translated_text, response = gpt_menu_match(text, menu_df)  # ⬅️ Updated here
#     print("GPT RESPONSE:", response)
#     if not response:
#         return [], "", [], 0.0, translated_text  # ⬅️ Also return translated_text even if empty

#     df = menu_df.copy()
#     df.columns = [col.lower().strip() for col in df.columns]

#     id_col = next((col for col in df.columns if col in ["id", "item id"]), None)
#     name_col = next((col for col in df.columns if "item" in col and "name" in col), None)
#     price_col = next((col for col in df.columns if "price" in col), None)

#     if not (id_col and name_col and price_col):
#         raise ValueError("Required columns (Id, Item Name, Price) not found in the menu.")

#     # Build a lookup with all clean variants of item names
#     menu_lookup = {}
#     for _, row in df.iterrows():
#         full_name = row[name_col].strip()
#         lower_name = full_name.lower()
    
#         # Create clean variants from slashes, parentheses, and base
#         parts = re.split(r"/|\(|\)", lower_name)
#         variants = [lower_name] + [p.strip() for p in parts if p.strip()]
    
#         for variant in variants:
#             cleaned_key = variant.translate(str.maketrans('', '', string.punctuation)).strip()
#             if cleaned_key not in menu_lookup:
#                 menu_lookup[cleaned_key] = {
#                     "Id": int(row[id_col]),
#                     "Rate": int(row[price_col]),
#                     "Name": full_name  # original for display
#                 }

#     order = defaultdict(lambda: {"Id": "", "Quantity": 0, "Rate": 0, "Total": 0})
#     unmatched_items = []
#     entries = [e.strip() for e in response.split(",") if e.strip()]
#     total_transcribed = len(entries)
#     total_matched = 0

#     # for entry in entries:
#     #     if "|" in entry:
#     #         item_part, qty_part = map(str.strip, entry.split("|", 1))
#     #         item_raw = item_part.lower()
#     #         item_name = item_raw.translate(str.maketrans('', '', string.punctuation)).strip()

#     #         try:
#     #             quantity = int(qty_part)
#     #         except ValueError:
#     #             quantity = 1  # fallback in case of bad quantity

#     #         if item_name in menu_lookup:
#     #             item = menu_lookup[item_name]
#     #             order[item["Name"]]["Id"] = item["Id"]
#     #             order[item["Name"]]["Quantity"] += quantity
#     #             order[item["Name"]]["Rate"] = item["Rate"]
#     #             order[item["Name"]]["Total"] += item["Rate"] * quantity
#     #             total_matched += 1
#     #         else:
#     #             unmatched_items.append(item_raw)
#     #             print(f"❌ No match for: {item_raw}")
#     #     else:
#     #         # ❌ If format is invalid or missing '|', add entire entry as unmatched
#     #         unmatched_items.append(entry.strip())
#     #         print(f"❌ Bad format or missing '|': {entry.strip()}")
#     for entry in entries:
#         parts = [p.strip() for p in entry.split("|")]
#         if len(parts) == 3:
#             item_id_str, item_name, qty_str = parts
#             item_name_key = item_name.lower().translate(str.maketrans('', '', string.punctuation)).strip()

#             try:
#                 item_id = int(item_id_str)
#                 quantity = int(qty_str)
#             except ValueError:
#                 quantity = 1  # fallback

#             if item_name_key in menu_lookup:
#                 item = menu_lookup[item_name_key]
#                 order[item["Name"]]["Id"] = item_id  # use GPT's item_id
#                 order[item["Name"]]["Quantity"] += quantity
#                 order[item["Name"]]["Rate"] = item["Rate"]
#                 order[item["Name"]]["Total"] += item["Rate"] * quantity
#                 total_matched += 1
#             else:
#                 unmatched_items.append(item_name)
#                 print(f"❌ No match for: {item_name}")
#         else:
#             unmatched_items.append(entry.strip())
#             print(f"❌ Bad format or missing fields: {entry.strip()}")

#     parsed = [
#         {
#             "Item": k,
#             "Id": int(v["Id"]),
#             "Quantity": int(v["Quantity"]),
#             "Rate": int(v["Rate"]),
#             "Total": int(v["Total"])
#         }
#         for k, v in order.items()
#     ]

#     total = sum(v["Total"] for v in order.values())
#     if parsed:
#         parsed.append({
#             "Item": "Total Amount",
#             "Id": "",
#             "Quantity": "",
#             "Rate": "",
#             "Total": total
#         })

#     # Accuracy metric
#     match_accuracy = round((total_matched / total_transcribed) * 100, 2) if total_transcribed > 0 else 0.0
#     print(f"🔎 Match accuracy: {match_accuracy}% ({total_matched}/{total_transcribed})")
#     print("Parsed order:", parsed)

#     return parsed, response, unmatched_items, match_accuracy, translated_text
def parse_order(text, menu_df):
    translated_text, response = gpt_menu_match(text, menu_df)
    print("GPT RESPONSE:", response)
    if not response:
        return [], "", [], 0.0, translated_text

    df = menu_df.copy()
    df.columns = [col.lower().strip() for col in df.columns]

    id_col = next((col for col in df.columns if col in ["id", "item id", "item_id"]), None)
    name_col = next((col for col in df.columns if "item" in col and "name" in col), None)
    price_col = next((col for col in df.columns if "price" in col), None)

    if not (id_col and name_col and price_col):
        raise ValueError("Required columns (Id, Item Name, Price) not found in the menu.")

    # Build a lookup using clean item names (to match GPT item_name)
    menu_lookup = {}
    for _, row in df.iterrows():
        full_name = row[name_col].strip()
        lower_name = full_name.lower()
        parts = re.split(r"/|\(|\)", lower_name)
        variants = [lower_name] + [p.strip() for p in parts if p.strip()]
        for variant in variants:
            cleaned_key = variant.translate(str.maketrans('', '', string.punctuation)).strip()
            if cleaned_key not in menu_lookup:
                menu_lookup[cleaned_key] = {
                    "Id": int(row[id_col]),
                    "Rate": int(row[price_col]),
                    "Name": full_name
                }

    order = defaultdict(lambda: {"Id": "", "Quantity": 0, "Rate": 0, "Total": 0})
    unmatched_items = []
    entries = [e.strip() for e in response.split(",") if e.strip()]
    total_transcribed = len(entries)
    total_matched = 0

    for entry in entries:
        parts = [p.strip() for p in entry.split("|")]
        if len(parts) == 3:
            try:
                item_id = int(parts[0])
                item_name = parts[1]
                quantity = int(parts[2])
            except ValueError:
                print(f"⚠️ Skipping entry due to value error: {entry}")
                continue

            # Clean the item name for lookup
            lookup_key = item_name.lower().translate(str.maketrans('', '', string.punctuation)).strip()

            if lookup_key in menu_lookup:
                menu_item = menu_lookup[lookup_key]
                order[menu_item["Name"]]["Id"] = item_id  # <-- Use GPT's item_id
                order[menu_item["Name"]]["Quantity"] += quantity
                order[menu_item["Name"]]["Rate"] = menu_item["Rate"]
                order[menu_item["Name"]]["Total"] += menu_item["Rate"] * quantity
                total_matched += 1
            else:
                unmatched_items.append(item_name)
                print(f"❌ No match for: {item_name}")
        else:
            unmatched_items.append(entry.strip())
            print(f"❌ Bad format or missing '|': {entry.strip()}")

    parsed = [
        {
            "Item": k,
            "Id": int(v["Id"]),
            "Quantity": int(v["Quantity"]),
            "Rate": int(v["Rate"]),
            "Total": int(v["Total"])
        }
        for k, v in order.items()
    ]

    total = sum(v["Total"] for v in order.values())
    if parsed:
        parsed.append({
            "Item": "Total Amount",
            "Id": "",
            "Quantity": "",
            "Rate": "",
            "Total": total
        })

    match_accuracy = round((total_matched / total_transcribed) * 100, 2) if total_transcribed > 0 else 0.0
    print(f"🔎 Match accuracy: {match_accuracy}% ({total_matched}/{total_transcribed})")
    print("Parsed order:", parsed)

    return parsed, response, unmatched_items, match_accuracy, translated_text

from fastapi import UploadFile, File, Form, Request
from pathlib import Path
import os, tempfile, shutil
import json
from datetime import datetime
from uuid import uuid4
import soundfile as sf
SESSION_STATE = {}  

@app.post("/process/")
async def process_audio_file(file: UploadFile = File(...), session_id: str = Form(...)):
    print("File received:", file.filename, "Session ID:", session_id)

    try:
        ext = Path(file.filename).suffix.lower()
        AUDIO_EXTENSIONS = {".wav", ".mp3", ".webm", ".ogg", ".m4a", ".mp4", ".flac", ".aac", ".aiff"}
        VIDEO_EXTENSIONS = {".mp4", ".webm", ".m4v", ".avi"}

        session_dir = os.path.join("audio", session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Generate unique file name
        base_name = uuid4().hex
        raw_temp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        raw_temp.write(await file.read())
        raw_temp.close()

        # Extract or pass through audio
        if ext in VIDEO_EXTENSIONS:
            audio_path = raw_temp.name + "_audio.mp3"
            extract_audio_from_video(raw_temp.name, audio_path)
        elif ext in AUDIO_EXTENSIONS:
            audio_path = raw_temp.name
        else:
            os.unlink(raw_temp.name)
            return {"error": f"Unsupported file extension: {ext}"}

        # Convert to WAV
        converted_path = audio_path + "_converted.wav"
        convert_audio_to_wav(audio_path, converted_path)

# Save the original uploaded audio (converted to WAV) to session folder
        final_original_path = os.path.join(session_dir, f"{base_name}_original.wav")
        shutil.copyfile(converted_path, final_original_path)

        # Denoise and use only for transcription (not saved)
        data, samplerate = sf.read(converted_path)
        if data.ndim > 1:
            data = data[:, 0]
        denoised = denoise_audio(data, samplerate)

        # Save to a temporary file just for transcription
        final_denoised_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sf.write(final_denoised_path, denoised, samplerate)
        # Clean up all intermediate temp files
        for path in [raw_temp.name, audio_path, converted_path]:
            if os.path.exists(path):
                os.remove(path)

        # Transcribe
        transcription = transcribe_audio(final_denoised_path)
        if menu_df.empty:
            return {"error": "No menu available."}
        if session_id not in SESSION_STATE:
            SESSION_STATE[session_id] = {
                "responses": [],
                "transcriptions": [],  # ✅ Add this line to store multiple transcriptions
                "translations": [],  # ✅ Add this

            }

        # parsed_order, response, unmatched_items, match_accuracy = parse_order(transcription, menu_df)
        parsed_order, response, unmatched_items, match_accuracy, translated_text = parse_order(transcription, menu_df)

        # ✅ Append the current transcription
        SESSION_STATE[session_id]["transcriptions"].append(transcription)
        SESSION_STATE[session_id]["responses"].append(response)
        SESSION_STATE[session_id]["translations"].append(translated_text)  # ✅ Store translation
        return {
            "session_id": session_id,
            "transcription": SESSION_STATE[session_id]["transcriptions"],
            "translated_transcript": SESSION_STATE[session_id]["translations"],  # ✅ Now included
            "gpt_response": SESSION_STATE[session_id]["responses"],
            "order": parsed_order,
            "unmatched": unmatched_items,
            "accuracy": match_accuracy
        }

    except Exception as e:
        return {"error": f"Unexpected error: {e}"}

@app.post("/feedback/")
async def receive_feedback(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    audio_files = []
    if session_id:
        session_dir = os.path.join("audio", session_id)
        if os.path.exists(session_dir):
            audio_files = os.listdir(session_dir)

    feedback_log = {
        "timestamp": timestamp,
        "session_id": session_id,
        "feedback": data.get("feedback", ""),
        "issues": data.get("issues", []),
        "transcriptions": SESSION_STATE.get(session_id, {}).get("transcriptions", []),  # ✅ Add this
        "translated_transcripts": SESSION_STATE.get(session_id, {}).get("translations", []),  # ✅ Add this
        "gpt_response": SESSION_STATE.get(session_id, {}).get("responses", []),
        "final_order": data.get("final_order", []),
        "audio_files": audio_files
    }

    print("\nReceived Feedback:")
    print(json.dumps(feedback_log, indent=4))

    with open("session_feedback.log", "a") as f:
        f.write(json.dumps(feedback_log) + "\n")

    return {"message": "Feedback received"}



# Instructions:
# - If the input is in a non-English language (e.g., Hindi, Tamil, Telugu, Urdu, malay), translate it to English before processing.
# - Customers may speak in different languages or accents — including English, Hindi, Tamil, and Malay. Understand their pronunciation and return the   best-matching menu item.
# - Quantities can appear before or after the dish name (e.g., "2 Pav Bhaji", "Pav Bhaji 2") — both are valid.
# - Ignore filler or address words like "bhaiya", "anna", "dada", "boss", or similar – they are not part of the order.
# - Identify food items and map them to the closest valid menu item, even if the names are partially spoken or mispronounced.
#   Examples: 
#   "Spoken Order: hot crispy chicken" → Original Menu item: "Hot & Crispy Chicken Bucket (6 pcs)", 
#   "Spoken Order: paneer popper" → "Original Menu item: Paneer Poppers", 
#   "Spoken Order: ginger burger" → "Original Menu item: Veg Zinger Burger", 
#   "Spoken Order: grilled zinger" → "Original Menu item: Tandoori Zinger Burger"
# - Use numeric digits for quantities (e.g., "two" → 2). If quantity is not stated, assume 1.
# - If a quantity is mentioned **after** a dish name (e.g., "butter chicken 2"), associate it correctly.
# - If no quantity is mentioned, assume it is 1.
# - Do not return "NaN" or leave blanks — always return numeric values.
# - If the same item appears multiple times, combine them and total the quantity.
# - If the user stutters or changes their mind mid-sentence (e.g., says "aaannnwwww one dosa... annnn one onion dosa"), discard incomplete or earlier mentions and keep only the final, most specific version ("1 Onion Dosa"). Ignore false starts, repetitions, or abandoned phrases.
# - Only include items that exist in the menu. Do not hallucinate and invent or guess items not found.
# - Output should be strictly in english with a single clean, comma-separated list (no bullet points or extra text), e.g.:
#   1 Veg Zinger Burger, 1 Hot & Crispy Chicken Bucket (6 pcs), 1 Pepsi 500 ml.
# - Format: "1 Item A, 2 Item B"
