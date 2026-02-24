# MiniLLM – Gemma 3 1B Chat

Lokaler Chatbot auf Basis von **Google Gemma 3 1B** (GGUF, quantisiert Q4/Q8), betrieben mit `llama-cpp-python`.

## Voraussetzungen

- Python 3.10+
- macOS (Metal-GPU-Beschleunigung) / Linux / Windows

## Installation

```bash
# Virtuelle Umgebung erstellen
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Abhängigkeiten installieren
pip install -r requirements.txt
```

### GPU-Beschleunigung (optional)

Auf **macOS** (Apple Silicon) wird Metal automatisch unterstützt.

Für **NVIDIA CUDA**:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

## Modell herunterladen

```bash
# Q4 (kleiner, ~0.7 GB, schneller)
python download_model.py --quant q4

# Q8 (besser, ~1.2 GB)
python download_model.py --quant q8
```

Das Modell wird im Ordner `models/` gespeichert.

> **Hinweis:** Der Chat lädt das Modell auch automatisch herunter, falls es noch nicht vorhanden ist.

## Chat starten

```bash
# Standard (Q4, Streaming)
python chat.py

# Q8-Modell verwenden
python chat.py --quant q8

# Eigenes Modell angeben
python chat.py --model models/mein-modell.gguf

# Nur CPU (kein GPU-Offloading)
python chat.py --gpu-layers 0

# Größeres Kontextfenster
python chat.py --ctx-size 16384
```

## Chat-Befehle

| Befehl             | Beschreibung                         |
|---------------------|--------------------------------------|
| `/clear`           | Chatverlauf zurücksetzen             |
| `/system <prompt>` | System-Prompt ändern                 |
| `/quit`            | Chat beenden (oder Ctrl+C / Ctrl+D) |

## Konfiguration

| Parameter         | Default | Beschreibung                          |
|--------------------|---------|---------------------------------------|
| `--quant`         | `q4`    | Quantisierung: `q4` oder `q8`        |
| `--ctx-size`      | `8192`  | Kontextfenster-Größe                 |
| `--max-tokens`    | `1024`  | Max. Tokens pro Antwort               |
| `--temperature`   | `0.7`   | Sampling-Temperatur                   |
| `--gpu-layers`    | `-1`    | GPU-Layer (-1 = alle, 0 = nur CPU)    |
| `--no-stream`     | `false` | Streaming deaktivieren                |

## Projektstruktur

```
minillm/
├── chat.py              # Haupt-Chat-Anwendung
├── download_model.py    # Modell-Downloader
├── requirements.txt     # Python-Abhängigkeiten
├── README.md
└── models/              # GGUF-Modelle (wird automatisch erstellt)
```
