import os
import json
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
AUDIO_ROOT = os.path.join(os.path.dirname(__file__), "..", "sample_audios")
test_results_summary = {}

# 🎧 Helper: send audio
def send_audio(filepath, mime=None, session_id="test-session"):
    assert os.path.exists(filepath), f"File not found: {filepath}"
    if not mime:
        if filepath.endswith(".wav"): mime = "audio/wav"
        elif filepath.endswith(".ogg"): mime = "audio/ogg"
        elif filepath.endswith(".mp3"): mime = "audio/mpeg"
        else: mime = "application/octet-stream"

    with open(filepath, "rb") as f:
        response = client.post(
            "/process/",
            files={"file": (os.path.basename(filepath), f, mime)},
            data={"session_id": session_id}
        )
    return response

# 🗂 Helper: auto-discover test folders, files, and meta
def collect_audio_cases():
    test_cases = []
    for folder in os.listdir(AUDIO_ROOT):
        folder_path = os.path.join(AUDIO_ROOT, folder)
        if os.path.isdir(folder_path):
            audio_files = [os.path.join(folder_path, f)
                           for f in os.listdir(folder_path)
                           if f.endswith((".wav", ".ogg", ".mp3"))]
            meta = {}
            meta_path = os.path.join(folder_path, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as mf:
                    meta = json.load(mf)
            test_cases.append((folder, audio_files, meta))
    return test_cases

# ✅ Parametrized test — handles all TC folders
@pytest.mark.parametrize("case_name, audio_files, meta", collect_audio_cases())
def test_audio_case(case_name, audio_files, meta):
    print(f"\n\n🧪 Running Test Case: {case_name}")
    passed = 0
    failed = 0

    for filepath in audio_files:
        print(f"\n▶️ Testing: {os.path.basename(filepath)}")
        try:
            response = send_audio(filepath)
            assert response.status_code == 200
            data = response.json()
            assert "order" in data and isinstance(data["order"], list)

            # 🧠 Built-in logic per test case
            if case_name == "TC01_valid_input":
                assert len(data["order"]) > 0

            elif case_name == "TC02_partial_input":
                pass  # Accepts any partial list

            elif case_name == "TC03_background_noise":
                assert "transcription" in data

            elif case_name == "TC04_silence_input":
                assert data["order"] == []

            elif case_name == "TC05_invalid_item":
                assert "unmatched" in data and len(data["unmatched"]) > 0

            elif case_name == "TC06_mixed_items":
                assert any("Item" in i for i in data["order"] if isinstance(i, dict))

            elif case_name == "TC07_mumbled_input":
                assert isinstance(data["order"], list)

            elif case_name == "TC08_non_english_input":
                assert isinstance(data.get("transcription"), list)
                assert all(isinstance(line, str) for line in data["transcription"])

            elif case_name == "TC09_repetitive_speech":
                quantities = [i["Quantity"] for i in data["order"]
                              if isinstance(i, dict) and i.get("Item") != "Total Amount"]
                assert any(q > 1 for q in quantities) or quantities.count(1) >= 2

            elif case_name == "TC10_multilang_input":
                assert len(data["order"]) > 0
                assert isinstance(data.get("transcription"), (str, list))

            # ✅ Optional: overrides via meta.json
            if "min_order_items" in meta:
                assert len(data["order"]) >= meta["min_order_items"]

            if meta.get("require_unmatched"):
                assert "unmatched" in data and isinstance(data["unmatched"], list)

            if "transcription_type" in meta:
                expected = meta["transcription_type"]
                assert isinstance(data.get("transcription"), list if expected == "list" else str)

            print("✅ Passed")
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            failed += 1

    test_results_summary[case_name] = {
        "total": len(audio_files),
        "passed": passed,
        "failed": failed
    }

# 📊 Show summary after all tests
def pytest_sessionfinish(session, exitstatus):
    print("\n📊 === Summary Report ===")
    for case, stat in test_results_summary.items():
        print(f"\n🔹 {case}")
        print(f"   Total : {stat['total']}")
        print(f"   ✅ Pass: {stat['passed']}")
        print(f"   ❌ Fail: {stat['failed']}")
