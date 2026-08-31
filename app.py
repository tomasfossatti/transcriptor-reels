import os
import re
import shutil
import tempfile

import ctranslate2
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


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text):
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sentences or [text]


def get_translator_package():
    """Descarga (si hace falta) el paquete en->es de Argos y lo carga
    usando solo ctranslate2 + sentencepiece, sin pasar por
    argostranslate.translate (que importa stanza/torch y dispara OOM
    en hosts con poca RAM, como el free tier de Render)."""
    global _translator
    if _translator is not None:
        return _translator

    import argostranslate.package

    pkg = next(
        (
            p
            for p in argostranslate.package.get_installed_packages()
            if p.from_code == "en" and p.to_code == "es"
        ),
        None,
    )

    if pkg is None:
        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()
        available_pkg = next(
            p for p in available if p.from_code == "en" and p.to_code == "es"
        )
        download_path = available_pkg.download()
        argostranslate.package.install_from_path(download_path)
        pkg = next(
            p
            for p in argostranslate.package.get_installed_packages()
            if p.from_code == "en" and p.to_code == "es"
        )

    model_path = str(pkg.package_path / "model")
    translator = ctranslate2.Translator(model_path, device="cpu")
    _translator = (translator, pkg)
    return _translator


def translate_to_spanish(text):
    translator, pkg = get_translator_package()

    sentences = _split_sentences(text)
    tokenized = [pkg.tokenizer.encode(s) for s in sentences]

    target_prefix = None
    if pkg.target_prefix:
        target_prefix = [[pkg.target_prefix]] * len(tokenized)

    results = translator.translate_batch(
        tokenized,
        target_prefix=target_prefix,
        replace_unknowns=True,
        max_batch_size=32,
        beam_size=4,
    )

    translated_sentences = []
    for result in results:
        tokens = result.hypotheses[0]
        value = pkg.tokenizer.decode(tokens)
        if pkg.target_prefix and value.startswith(pkg.target_prefix):
            value = value[len(pkg.target_prefix):]
        translated_sentences.append(value.strip())

    return " ".join(translated_sentences)


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
