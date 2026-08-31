import os
import shutil
import tempfile

from flask import Flask, jsonify, render_template, request
import yt_dlp
from faster_whisper import WhisperModel

app = Flask(__name__)

# tiny / base / small: a mayor modelo, mejor calidad pero más lento y más RAM.
# "base" es un buen equilibrio para una compu sin GPU.
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")

_model = None
_translator = None


def get_model():
    global _model
    if _model is None:
        _model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def download_audio(url, out_dir):
    out_template = os.path.join(out_dir, "audio.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    for fname in os.listdir(out_dir):
        if fname.startswith("audio.") and fname.endswith(".wav"):
            return os.path.join(out_dir, fname)

    raise RuntimeError(
        "No se pudo extraer el audio del video. "
        "Puede que el reel sea privado o que la URL no sea válida."
    )


def transcribe_audio(audio_path):
    model = get_model()
    segments, info = model.transcribe(audio_path, beam_size=5)
    text = " ".join(segment.text.strip() for segment in segments)
    return text.strip(), info.language


def get_translator():
    global _translator
    if _translator is not None:
        return _translator

    import argostranslate.package
    import argostranslate.translate

    installed = argostranslate.translate.get_installed_languages()
    en_lang = next((l for l in installed if l.code == "en"), None)
    es_lang = next((l for l in installed if l.code == "es"), None)

    if en_lang is None or es_lang is None:
        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()
        package = next(
            p for p in available if p.from_code == "en" and p.to_code == "es"
        )
        path = package.download()
        argostranslate.package.install_from_path(path)

        installed = argostranslate.translate.get_installed_languages()
        en_lang = next(l for l in installed if l.code == "en")
        es_lang = next(l for l in installed if l.code == "es")

    _translator = en_lang.get_translation(es_lang)
    return _translator


def translate_to_spanish(text):
    return get_translator().translate(text)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()

    if not url:
        return jsonify({"error": "Falta la URL del reel."}), 400
    if "instagram.com" not in url:
        return jsonify({"error": "La URL debe ser de Instagram."}), 400

    tmp_dir = tempfile.mkdtemp(prefix="reel_")
    try:
        audio_path = download_audio(url, tmp_dir)
        transcript, language = transcribe_audio(audio_path)

        result = {"language": language, "transcript": transcript, "translation": None}

        if language == "en" and transcript:
            try:
                result["translation"] = translate_to_spanish(transcript)
            except Exception as exc:
                result["translation_error"] = f"No se pudo traducir: {exc}"

        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
