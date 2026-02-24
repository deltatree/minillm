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

## OpenAI-kompatible API

MiniLLM bietet einen HTTP-Server, der die **OpenAI Chat Completions API** nachbildet. Damit lässt sich das lokale Modell mit jedem Tool verwenden, das die OpenAI-API unterstützt (z.B. `openai`-Library, LangChain, Continue, etc.).

### Server starten

```bash
# Standard (Q4, Port 8000)
python server.py

# Q8-Modell, anderer Port
python server.py --quant q8 --port 9000

# Eigenes Modell
python server.py --model models/mein-modell.gguf

# Nur CPU
python server.py --gpu-layers 0
```

Der Server stellt folgende Endpoints bereit:

| Endpoint                    | Methode | Beschreibung                        |
|-----------------------------|---------|-------------------------------------|
| `/v1/chat/completions`     | POST    | Chat Completions (Streaming & Non-Streaming) |
| `/v1/models`               | GET     | Liste der verfügbaren Modelle       |
| `/health`                  | GET     | Health-Check                        |
| `/docs`                    | GET     | Interaktive API-Dokumentation (Swagger UI) |

### Server-Parameter

| Parameter         | Default     | Beschreibung                          |
|--------------------|-------------|---------------------------------------|
| `--quant`         | `q4`        | Quantisierung: `q4` oder `q8`        |
| `--model`         | –           | Pfad zu einer GGUF-Datei              |
| `--ctx-size`      | `8192`      | Kontextfenster-Größe                 |
| `--gpu-layers`    | `-1`        | GPU-Layer (-1 = alle, 0 = nur CPU)    |
| `--host`          | `0.0.0.0`  | Host-Adresse                          |
| `--port`          | `8000`      | Port                                  |

### Beispiel: curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-1b-it",
    "messages": [
      {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
      {"role": "user", "content": "Was ist die Hauptstadt von Österreich?"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### Beispiel: Python mit openai-Library

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # Lokaler Server braucht keinen Key
)

response = client.chat.completions.create(
    model="gemma-3-1b-it",
    messages=[
        {"role": "user", "content": "Erkläre Machine Learning in 2 Sätzen."}
    ],
    max_tokens=256,
)

print(response.choices[0].message.content)
```

### Beispiel: Python mit openai-Library (Streaming)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

stream = client.chat.completions.create(
    model="gemma-3-1b-it",
    messages=[{"role": "user", "content": "Hallo!"}],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Beispiel: Python mit requests

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "gemma-3-1b-it",
        "messages": [
            {"role": "user", "content": "Was ist 2 + 2?"}
        ],
    },
)

print(response.json()["choices"][0]["message"]["content"])
```

### Alle Beispiele auf einmal ausführen

```bash
# Server muss laufen (in einem separaten Terminal)
python example_api_call.py
```

## Projektstruktur

```
minillm/
├── chat.py              # Interaktiver Chat (Terminal)
├── server.py            # OpenAI-kompatibler HTTP-Server
├── example_api_call.py  # Beispielaufrufe gegen die API
├── download_model.py    # Modell-Downloader
├── requirements.txt     # Python-Abhängigkeiten
├── README.md
└── models/              # GGUF-Modelle (wird automatisch erstellt)
```
