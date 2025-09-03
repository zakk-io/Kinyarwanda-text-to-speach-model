from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from transformers import VitsModel, AutoTokenizer
import torch, numpy as np, io
from scipy.io.wavfile import write as wav_write

app = Flask(__name__)
CORS(app)  # allow calls from your frontend (localhost, etc.)

# --- Load once at startup (keeps latency low) ---
MODEL_NAME = "facebook/mms-tts-kin"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = VitsModel.from_pretrained(MODEL_NAME).to(device).eval()
sampling_rate = model.config.sampling_rate

@torch.inference_mode()   # no gradients, faster
def synthesize_to_wav_bytes(text: str) -> bytes:
    # Tokenize and move to the same device as the model
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Generate waveform (float32 in [-1, 1])
    waveform = model(**inputs).waveform.squeeze(0).detach().cpu().numpy()

    # Gentle normalization to avoid clipping
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-9)

    # Convert to 16-bit PCM WAV in-memory (no files on disk)
    y_int16 = np.int16(np.clip(waveform, -1.0, 1.0) * 32767)
    buf = io.BytesIO()
    wav_write(buf, sampling_rate, y_int16)
    buf.seek(0)
    return buf.read()

@app.route("/kinyarwanda-tts", methods=["POST"])
def tts():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Missing 'text'"}), 400
    # keep things snappy + safe for demo
    if len(text) > 1200:
        return jsonify({"error": "Text too long; please keep it under 1200 characters."}), 413

    audio_bytes = synthesize_to_wav_bytes(text)
    # Return raw WAV bytes; the browser will play them
    return Response(
        audio_bytes,
        mimetype="audio/wav",
        headers={
            "Cache-Control": "no-store",
            "Content-Type": "audio/wav",
        },
    )

if __name__ == "__main__":
    # Run: python app.py  (serves on http://localhost:5000)
    app.run(host="0.0.0.0", port=5000)