#!/usr/bin/env python3
"""One-request-per-speaker resumable Gemini 3.8 TTS using the Interactions API."""
from __future__ import annotations
import argparse, base64, hashlib, json, os, re, time, wave
from pathlib import Path
from typing import Any
import requests

MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts")
API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
VOICE_TAGS = {"laugh", "sigh", "gasp", "cough", "breath", "short pause", "pause"}
TAG_RE = re.compile(r"<([^<>]+)>")
SEPARATOR_RE = re.compile(r"\s*\.\.\.\s*")

def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def make_turns(speaker: dict[str, Any]) -> list[dict[str, str]]:
    turns = []
    for raw in SEPARATOR_RE.split(speaker["text"]):
        raw = re.sub(r"^\s*\[[^\]]+\]\s*", "", raw.strip())
        if not raw:
            continue
        tags = [m.group(1).strip().lower().replace("_", " ") for m in TAG_RE.finditer(raw)]
        moods = [t for t in tags if t not in VOICE_TAGS and t not in {"chuckle", "laughing"}]
        def preserve(match):
            tag = match.group(1).strip().lower().replace("_", " ")
            if tag in VOICE_TAGS:
                return f"<{tag}>"
            if tag in {"chuckle", "laughing"}:
                return "<laugh>"
            return ""
        text = re.sub(r"\s+", " ", TAG_RE.sub(preserve, raw)).strip()
        if not text:
            continue
        style = speaker["style"] + (
            " Speak the supplied Arabic words verbatim in natural Modern Standard Arabic. "
            "Do not translate, paraphrase, omit, or add words."
        )
        if moods:
            style += " Acting direction for this utterance: " + ", ".join(dict.fromkeys(moods)) + "."
        turns.append({"text": text, "style": style})
    return turns

def extract_audio(payload: dict[str, Any]) -> bytes:
    audio = payload.get("output_audio") or {}
    if audio.get("data"):
        return base64.b64decode(audio["data"])
    for step in payload.get("steps", []):
        if step.get("type") == "model_output":
            for part in step.get("content", []):
                if part.get("type") == "audio" and part.get("data"):
                    return base64.b64decode(part["data"])
    raise RuntimeError("Interactions API response contained no model-output audio data: " + json.dumps(payload, ensure_ascii=False)[:1500])

def request_audio(session, api_key, speaker, turns):
    content = [{
        "type": "text",
        "text": turn["text"],
        "annotations": [{"type": "speech_metadata", "style": turn["style"]}],
    } for turn in turns]
    body = {
        "model": MODEL,
        "input": [{"type": "user_input", "content": content}],
        "response_format": {"type": "audio"},
        "generation_config": {"speech_config": [{"voice": speaker["voice"]}]},
    }
    last_error = None
    for attempt in range(1, 5):
        try:
            response = session.post(
                API_URL,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=body,
                timeout=(30, 600),
            )
            if response.status_code == 429:
                last_error = response.text[:2500]
                lower = last_error.lower()
                if "requests per day" in lower or "free_tier_requests" in lower or "retry in" in lower and ("h" in lower or "m" in lower):
                    raise RuntimeError("Google daily TTS quota is exhausted; stop now and retry after reset. " + last_error[:1600])
                if attempt == 4:
                    break
                delay = min(60, 5 * (2 ** (attempt - 1)))
                print(f"Temporary rate limit; retrying in {delay}s", flush=True)
                time.sleep(delay)
                continue
            if response.status_code >= 500:
                last_error = f"HTTP {response.status_code}: {response.text[:1600]}"
                if attempt == 4:
                    break
                time.sleep(min(60, 5 * (2 ** (attempt - 1))))
                continue
            if not response.ok:
                raise RuntimeError(f"Gemini Interactions API HTTP {response.status_code}: {response.text[:2500]}")
            return extract_audio(response.json())
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = str(exc)
            if attempt == 4:
                break
            time.sleep(min(60, 5 * (2 ** (attempt - 1))))
    raise RuntimeError("Gemini TTS retries exhausted: " + str(last_error))

def validate_wav(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        params = wf.getparams()
        if (params.nchannels, params.sampwidth, params.framerate, params.comptype) != (1, 2, 24000, "NONE"):
            raise RuntimeError(f"Unexpected WAV format in {path.name}: {params}")
        return wf.getnframes() / wf.getframerate()

def drive_upload(path: Path, folder_id: str, service_json: str) -> str | None:
    if not service_json:
        return None
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request
    info = json.loads(service_json)
    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/drive"]
    )
    credentials.refresh(Request())
    headers = {"Authorization": f"Bearer {credentials.token}"}
    escaped_name = path.name.replace("\\", "\\\\").replace("'", "\\'")
    query = f"'{folder_id}' in parents and name = '{escaped_name}' and trashed = false"
    found = requests.get(
        "https://www.googleapis.com/drive/v3/files",
        headers=headers,
        params={"q": query, "fields": "files(id,name,webViewLink)", "pageSize": 10},
        timeout=60,
    )
    found.raise_for_status()
    matches = found.json().get("files", [])
    content = path.read_bytes()
    if matches:
        result = requests.patch(
            f"https://www.googleapis.com/upload/drive/v3/files/{matches[0]['id']}",
            headers={**headers, "Content-Type": "audio/wav"},
            params={"uploadType": "media", "fields": "id,name,webViewLink"},
            data=content, timeout=300,
        )
    else:
        boundary = "mpc-dubbing-boundary"
        metadata = json.dumps({"name": path.name, "mimeType": "audio/wav", "parents": [folder_id]})
        body = (
            f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n{metadata}\r\n"
            f"--{boundary}\r\nContent-Type: audio/wav\r\n\r\n"
        ).encode("utf-8") + content + f"\r\n--{boundary}--".encode("utf-8")
        result = requests.post(
            "https://www.googleapis.com/upload/drive/v3/files",
            headers={**headers, "Content-Type": f"multipart/related; boundary={boundary}"},
            params={"uploadType": "multipart", "fields": "id,name,webViewLink"},
            data=body, timeout=300,
        )
    result.raise_for_status()
    data = result.json()
    return data.get("webViewLink") or data.get("id")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-speaker", type=int, default=int(os.getenv("START_SPEAKER", "6")))
    parser.add_argument("--end-speaker", type=int, default=int(os.getenv("END_SPEAKER", "10")))
    parser.add_argument("--source", default="data/speakers.json")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is missing.")
    source = json.loads(Path(args.source).read_text(encoding="utf-8"))
    speakers = source["speakers"]
    output = Path(args.output)
    final_root = output / "final"
    final_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "model": MODEL, "language": "Arabic", "sample_rate": 24000, "tracks": {}
    }
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "").strip()
    drive_json = os.getenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON", "").strip()
    session = requests.Session()

    for number, speaker in enumerate(speakers, start=1):
        if number < args.start_speaker or number > args.end_speaker:
            continue
        track_key = speaker["id"]
        filename = f"{number:02d}_{speaker['name'].replace(' ', '_')}_{speaker['voice']}.wav"
        final_path = final_root / filename
        state = manifest["tracks"].setdefault(track_key, {
            "speaker": speaker["name"], "voice": speaker["voice"], "status": "pending", "final_file": None
        })

        if final_path.exists() and final_path.stat().st_size > 44:
            duration = validate_wav(final_path)
            state.update({
                "status": "complete", "final_file": str(final_path),
                "duration_seconds": round(duration, 2), "bytes": final_path.stat().st_size,
                "sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
            })
            if drive_json and folder_id and not state.get("drive_url"):
                state["drive_url"] = drive_upload(final_path, folder_id, drive_json)
            atomic_json(manifest_path, manifest)
            print(f"TRACK_ALREADY_COMPLETE {number}: {speaker['name']}", flush=True)
            continue

        turns = make_turns(speaker)
        if not turns:
            raise RuntimeError(f"No speech turns found for {speaker['name']}")
        total_chars = sum(len(t["text"]) for t in turns)
        total_bytes = sum(len(t["text"].encode("utf-8")) for t in turns)
        print(
            f"TRACK {number}/10 {speaker['name']} voice={speaker['voice']} "
            f"turns={len(turns)} chars={total_chars} text_bytes={total_bytes}",
            flush=True,
        )

        audio = request_audio(session, api_key, speaker, turns)
        tmp = final_path.with_suffix(".wav.tmp")
        tmp.write_bytes(audio)
        tmp.replace(final_path)
        duration = validate_wav(final_path)
        state.update({
            "status": "complete", "final_file": str(final_path),
            "duration_seconds": round(duration, 2), "bytes": final_path.stat().st_size,
            "sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(), "turns": len(turns),
        })
        if drive_json and folder_id:
            state["drive_url"] = drive_upload(final_path, folder_id, drive_json)
            print(f"DRIVE_UPLOADED {final_path.name} {state['drive_url']}", flush=True)
        else:
            print(
                f"DRIVE_UPLOAD_SKIPPED for {final_path.name}: GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON not configured.",
                flush=True,
            )
        atomic_json(manifest_path, manifest)
        print(
            f"TRACK_COMPLETE {final_path.name} duration={duration:.2f}s bytes={final_path.stat().st_size}",
            flush=True,
        )

    requested = [speakers[i - 1]["id"] for i in range(args.start_speaker, args.end_speaker + 1) if 1 <= i <= len(speakers)]
    complete = all(manifest["tracks"].get(k, {}).get("status") == "complete" for k in requested)
    manifest["requested_speakers_complete"] = complete
    atomic_json(manifest_path, manifest)
    if complete:
        print("ALL_REQUESTED_SPEAKERS_COMPLETE", flush=True)
    else:
        raise RuntimeError("Some requested speaker tracks are not complete.")

if __name__ == "__main__":
    main()
