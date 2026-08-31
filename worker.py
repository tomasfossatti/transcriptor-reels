"""
Worker de un solo uso: se invoca como subproceso separado para transcribir
o traducir, y termina apenas entrega el resultado. Así el sistema operativo
libera toda su memoria al salir, en vez de confiar en que Python/ctranslate2
la devuelvan dentro de un proceso de larga duración (necesario en hosts con
poca RAM, como el free tier de Render, con solo 512MB).
"""
import json
import os
import re
import sys

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text):
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sentences or [text]


def cmd_transcribe(audio_path):
    from faster_whisper import WhisperModel

    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device="cpu",
        compute_type="int8",
        cpu_threads=1,
        num_workers=1,
    )
    segments, info = model.transcribe(audio_path, beam_size=5)
    text = " ".join(segment.text.strip() for segment in segments)
    return {"transcript": text.strip(), "language": info.language}


def _get_installed_en_es_package():
    import argostranslate.package

    pkg = next(
        (
            p
            for p in argostranslate.package.get_installed_packages()
            if p.from_code == "en" and p.to_code == "es"
        ),
        None,
    )
    if pkg is not None:
        return pkg

    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()
    available_pkg = next(
        p for p in available if p.from_code == "en" and p.to_code == "es"
    )
    download_path = available_pkg.download()
    argostranslate.package.install_from_path(download_path)
    return next(
        p
        for p in argostranslate.package.get_installed_packages()
        if p.from_code == "en" and p.to_code == "es"
    )


def cmd_translate(text):
    import ctranslate2

    pkg = _get_installed_en_es_package()
    model_path = str(pkg.package_path / "model")
    translator = ctranslate2.Translator(
        model_path, device="cpu", inter_threads=1, intra_threads=1
    )

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

    return {"translation": " ".join(translated_sentences)}


def main():
    command = sys.argv[1]

    if command == "transcribe":
        result = cmd_transcribe(sys.argv[2])
    elif command == "translate":
        text = sys.stdin.read()
        result = cmd_translate(text)
    else:
        raise SystemExit(f"Comando desconocido: {command}")

    # Marcador unico + JSON como ultima linea de stdout, para poder
    # separarlo de cualquier log/warning que las librerias escriban antes.
    print("###RESULT###" + json.dumps(result))


if __name__ == "__main__":
    main()
