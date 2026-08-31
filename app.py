import json
import os
import shutil
import subprocess
import sys
import tempfile

from flask import Flask, jsonify, render_template, request
import yt_dlp

app = Flask(__name__)

WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")

# La transcripcion y la traduccion corren en subprocesos aparte (worker.py)
# en vez de en el proceso de Flask. Un proceso que termina le devuelve TODA
# su memoria al sistema operativo, algo que no se puede garantizar con
# del/gc.collect() en un proceso Python de larga duracion (el allocator de
# ctranslate2 puede retener memoria). En un host de 512MB (free tier de
# Render) es la unica forma confiable de que whisper y el traductor nunca
# queden residentes en memoria al mismo tiempo.


def run_worker(args, input_text=None):
    proc = subprocess.run(
        [sys.executable, WORKER_PATH] + args,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=280,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "Error en el worker")

    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("###RESULT###"):
            return json.loads(line[len("###RESULT###"):])

    raise RuntimeError("El worker no devolvio un resultado valido.")


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
    result = run_worker(["transcribe", audio_path])
    return result["transcript"], result["language"]


def translate_to_spanish(text):
    result = run_worker(["translate"], input_text=text)
    return result["translation"]


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
