#!/usr/bin/env python3
"""
Beispielaufrufe gegen die lokale OpenAI-kompatible API.

Voraussetzung: Der Server läuft bereits:
    python server.py

Dieses Skript zeigt drei Varianten:
  1. requests (einfacher HTTP-Call)
  2. openai Python-Library
  3. curl-Befehl (wird nur ausgegeben)
"""

import json
import sys

API_BASE = "http://localhost:8000/v1"


# ── Variante 1: Mit requests ───────────────────────────────────────────────

def example_with_requests():
    """Einfacher API-Aufruf mit der requests-Library."""
    import requests

    print("=" * 60)
    print("Variante 1: requests")
    print("=" * 60)

    response = requests.post(
        f"{API_BASE}/chat/completions",
        json={
            "model": "gemma-3-1b-it",
            "messages": [
                {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
                {"role": "user", "content": "Was ist die Hauptstadt von Österreich?"},
            ],
            "max_tokens": 256,
            "temperature": 0.7,
            "stream": False,
        },
    )

    if response.status_code != 200:
        print(f"Fehler: {response.status_code} – {response.text}")
        return

    data = response.json()
    print(f"\nModell: {data['model']}")
    print(f"Antwort: {data['choices'][0]['message']['content']}")
    print(f"Tokens: {data['usage']}")
    print()


# ── Variante 2: Mit openai-Library ─────────────────────────────────────────

def example_with_openai():
    """API-Aufruf mit der offiziellen openai Python-Library."""
    try:
        from openai import OpenAI
    except ImportError:
        print("(openai-Library nicht installiert – überspringe Variante 2)")
        print("  Installieren mit: pip install openai\n")
        return

    print("=" * 60)
    print("Variante 2: openai Python-Library")
    print("=" * 60)

    client = OpenAI(
        base_url=API_BASE,
        api_key="not-needed",  # Lokaler Server braucht keinen API-Key
    )

    response = client.chat.completions.create(
        model="gemma-3-1b-it",
        messages=[
            {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
            {"role": "user", "content": "Erkläre in 2 Sätzen, was Machine Learning ist."},
        ],
        max_tokens=256,
        temperature=0.7,
    )

    print(f"\nAntwort: {response.choices[0].message.content}")
    print(f"Tokens: prompt={response.usage.prompt_tokens}, "
          f"completion={response.usage.completion_tokens}\n")


# ── Variante 3: Mit openai-Library (Streaming) ─────────────────────────────

def example_with_openai_streaming():
    """Streaming-Aufruf mit der openai Python-Library."""
    try:
        from openai import OpenAI
    except ImportError:
        print("(openai-Library nicht installiert – überspringe Variante 3)\n")
        return

    print("=" * 60)
    print("Variante 3: openai Python-Library (Streaming)")
    print("=" * 60)

    client = OpenAI(
        base_url=API_BASE,
        api_key="not-needed",
    )

    print("\nAntwort: ", end="", flush=True)
    stream = client.chat.completions.create(
        model="gemma-3-1b-it",
        messages=[
            {"role": "user", "content": "Zähle von 1 bis 5 und sage zu jeder Zahl ein Wort."},
        ],
        max_tokens=256,
        temperature=0.7,
        stream=True,
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        print(token, end="", flush=True)
    print("\n")


# ── Variante 4: curl-Befehl anzeigen ───────────────────────────────────────

def show_curl_example():
    """Zeigt den entsprechenden curl-Befehl an."""
    print("=" * 60)
    print("Variante 4: curl")
    print("=" * 60)

    curl_cmd = f"""
curl {API_BASE}/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "gemma-3-1b-it",
    "messages": [
      {{"role": "system", "content": "Du bist ein hilfreicher Assistent."}},
      {{"role": "user", "content": "Was ist 2 + 2?"}}
    ],
    "max_tokens": 128,
    "temperature": 0.7
  }}'
"""
    print(curl_cmd)


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🔗 Beispielaufrufe gegen die lokale OpenAI-API")
    print(f"   Base-URL: {API_BASE}\n")

    # Prüfe, ob der Server erreichbar ist
    try:
        import requests
        health = requests.get("http://localhost:8000/health", timeout=3)
        if health.status_code == 200:
            info = health.json()
            print(f"   Server-Status: {info['status']} | Modell: {info['model']}\n")
        else:
            print("   ⚠️  Server antwortet, aber mit unerwartetem Status.\n")
    except Exception:
        print("   ⚠️  Server nicht erreichbar! Bitte zuerst starten:")
        print("      python server.py\n")
        sys.exit(1)

    example_with_requests()
    example_with_openai()
    example_with_openai_streaming()
    show_curl_example()
