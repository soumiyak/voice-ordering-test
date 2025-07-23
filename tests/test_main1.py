import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def send_audio(filename, mime=None, session_id="test-session"):
    folder = os.path.join(os.path.dirname(__file__), "..", "sample_audios")
    path = os.path.join(folder, filename)
    assert os.path.exists(path), f"File not found: {filename}"

    if not mime:
        if filename.endswith(".wav"):
            mime = "audio/wav"
        elif filename.endswith(".ogg"):
            mime = "audio/ogg"
        elif filename.endswith(".mp3"):
            mime = "audio/mpeg"
        else:
            mime = "application/octet-stream"

    with open(path, "rb") as f:
        response = client.post(
            "/process/",
            files={"file": (filename, f, mime)},
            data={"session_id": session_id}
        )
    return response

def check_key(data, key):
    assert key in data, f"Missing key in response: {key}. Full response: {data}"

# ✅ TC01 - Valid input test
def test_valid_input():
    response = send_audio("sample.wav")
    data = response.json()
    print("Transcription:", data.get("transcription", ""))
    print("GPT Response:", data.get("gpt_response", ""))
    print("Unmatched Items:", data.get("unmatched", []))
    print("Parsed Order:", data.get("order", []))
    assert response.status_code == 200
    check_key(data, "order")
    assert len(data["order"]) > 0

# ✅ TC02 - Partial input (adjusted expectation)
def test_partial_input():
    response = send_audio("partial_item.ogg")
    data = response.json()
    print("Transcription:", data.get("transcription", ""))
    print("GPT Response:", data.get("gpt_response", ""))
    print("Unmatched Items:", data.get("unmatched", []))
    print("Parsed Order:", data.get("order", []))
    assert response.status_code == 200
    check_key(data, "order")
    assert isinstance(data["order"], list)

# ✅ TC03 - Background noise
def test_background_noise():
    response = send_audio("background_noise.ogg")
    data = response.json()
    assert response.status_code == 200
    check_key(data, "transcription")

# ✅ TC04 - Silence
def test_silence_input():
    response = send_audio("silence.ogg")
    data = response.json()
    assert response.status_code == 200
    check_key(data, "order")
    assert data["order"] == []

# ✅ TC05 - Invalid item
def test_invalid_item():
    response = send_audio("invalid_item.ogg")
    data = response.json()
    assert response.status_code == 200
    check_key(data, "unmatched")
    assert len(data["unmatched"]) > 0

# ✅ TC06 - Mixed items
def test_mixed_items():
    response = send_audio("mixed_items.ogg")
    data = response.json()
    print("Transcription:", data.get("transcription", ""))
    print("GPT Response:", data.get("gpt_response", ""))
    print("Unmatched Items:", data.get("unmatched", []))
    print("Parsed Order:", data.get("order", []))
    assert response.status_code == 200
    check_key(data, "order")
    assert any(i.get("Item") for i in data["order"] if isinstance(i, dict))

# ✅ TC07 - Mumbled input (GPT hallucination issue)
def test_mumbled_input():
    response = send_audio("nonsense.ogg")
    data = response.json()
    assert response.status_code == 200
    check_key(data, "order")
    assert isinstance(data["order"], list)

# ✅ TC08 - Non-English / quantity input (check list of strings)
def test_non_english_input():
    response = send_audio("quantity_words.ogg")
    data = response.json()
    assert response.status_code == 200
    check_key(data, "transcription")
    transcription = data["transcription"]
    assert isinstance(transcription, list)
    assert all(isinstance(line, str) for line in transcription)

# ✅ TC09 - Repetitive speech (allow merged or repeated lines)
def test_repetitive_speech():
    response = send_audio("repeat_item.ogg")
    data = response.json()
    print("Transcription:", data.get("transcription", ""))
    print("GPT Response:", data.get("gpt_response", ""))
    print("Unmatched Items:", data.get("unmatched", []))
    print("Parsed Order:", data.get("order", []))
    assert response.status_code == 200
    check_key(data, "order")
    quantities = [i["Quantity"] for i in data["order"] if isinstance(i, dict) and i.get("Item") != "Total Amount"]
    assert any(q > 1 for q in quantities) or quantities.count(1) >= 2

# ✅ TC10 - Multilingual input
def test_multilang_input():
    response = send_audio("multilang_order.wav")
    data = response.json()
    print("Transcription:", data.get("transcription", ""))
    print("GPT Response:", data.get("gpt_response", ""))
    print("Unmatched Items:", data.get("unmatched", []))
    print("Parsed Order:", data.get("order", []))
    assert response.status_code == 200
    check_key(data, "order")
    assert len(data["order"]) > 0
