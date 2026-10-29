#!/usr/bin/env python3
"""Resumable Arabic multi-speaker dubbing with Gemini 3.8 Flash TTS."""
from __future__ import annotations
import argparse, base64, io, json, os, re, sys, time
from pathlib import Path
from typing import Any
import requests
import wave

MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts")
API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
VOICE_TAGS = {"laugh", "sigh", "gasp", "cough", "breath", "short pause"}
TAG_RE = re.compile(r"<([^<>]+)>")
SEPARATOR_RE = re.compile(r"\s*\.\.\.\s*")

def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def split_long(text: str, limit: int = 360) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return [text] if text else []
    sentences = re.split(r"(?<=[،؛؟.!])\s+", text)
    out, current = [], ""
    for sentence in sentences:
        if len(sentence) > limit:
            words = sentence.split()
            for word in words:
                if current and len(current) + len(word) + 1 > limit:
                    out.append(current.strip()); current = ""
                current += (" " if current else "") + word
            continue
        if current and len(current) + len(sentence) + 1 > limit:
            out.append(current.strip()); current = sentence
        else:
            current += (" " if current else "") + sentence
    if current.strip():
        out.append(current.strip())
    return out

def make_jobs(speaker: dict[str, Any]) -> list[dict[str, str]]:
    jobs = []
    for part_index, raw in enumerate(SEPARATOR_RE.split(speaker["text"]), start=1):
        if not raw.strip():
            continue
        raw = re.sub(r"^\s*\[[^\]]+\]\s*", "", raw.strip())
        tags = [m.group(1).strip().lower() for m in TAG_RE.finditer(raw)]
        vocal = []
        moods = []
        for tag in tags:
            normalized = tag.replace("_", " ").strip()
            if normalized in VOICE_TAGS:
                vocal.append(f"<{normalized}>")
            elif normalized in {"chuckle", "laughing"}:
                vocal.append("<laugh>")
            else:
                moods.append(normalized)
        # Keep only supported vocal-event tags in the spoken text. Acting cues
        # become speech_metadata.style instead of being read aloud.
        clean = TAG_RE.sub("", raw)
        clean = re.sub(r"^\s*\[[^\]]+\]\s*", "", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        if not clean:
            continue
        style = speaker["style"]
        if moods:
            style += " Emotional acting direction for this utterance: " + ", ".join(dict.fromkeys(moods)) + "."
        for sub_index, chunk in enumerate(split_long(clean), start=1):
            text = " ".join(vocal) + (" " if vocal else "") + chunk
            jobs.append({
                "text": text,
                "style": style,
                "job_id": f"{part_index:04d}_{sub_index:02d}",
            })
    return jobs

def extract_audio(payload: dict[str, Any]) -> bytes:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates: " + json.dumps(payload.get("promptFeedback", {}), ensure_ascii=False))
    parts = ((candidates[0].get("content") or {}).get("parts") or [])
    for part in parts:
        inline = part.get("inlineData") or part.get("inline_data") or {}
        data = inline.get("data")
        if data:
            audio = base64.b64decode(data)
            if audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
                raise RuntimeError("Gemini audio payload was not a complete WAV file")
            return audio
    raise RuntimeError("Gemini response contained no inline audio data")

def tts_request(session: requests.Session, api_key: str, voice: str, text: str, style: str) -> bytes:
    url = f"{API_ROOT}/{MODEL}:generateContent"
    body = {
        "contents": [{
            "role": "user",
            "parts": [{"text": text, "speech_metadata": {"style": style}}],
        }],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"voice": voice}},
        },
    }
    last_error = None
    for attempt in range(1, 7):
        try:
            response = session.post(
                url, headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=body, timeout=(20, 180)
            )
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}: {response.text[:900]}"
                delay = min(90, 5 * (2 ** (attempt - 1)))
                print(f"RETRY {attempt}/6 after {response.status_code}; wait={delay}s", flush=True)
                time.sleep(delay)
                continue
            if not response.ok:
                raise RuntimeError(f"Gemini API HTTP {response.status_code}: {response.text[:1600]}")
            return extract_audio(response.json())
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = str(exc)
            delay = min(90, 5 * (2 ** (attempt - 1)))
            print(f"NETWORK RETRY {attempt}/6; wait={delay}s", flush=True)
            time.sleep(delay)
    raise RuntimeError("Gemini request retries exhausted: " + str(last_error))

def read_wav(data: bytes):
    with wave.open(io.BytesIO(data), "rb") as wf:
        params = wf.getparams()
        frames = wf.readframes(wf.getnframes())
    if (params.nchannels, params.sampwidth, params.framerate, params.comptype) != (1, 2, 24000, "NONE"):
        raise RuntimeError(f"Unexpected WAV format: {params}")
    return params, frames

def assemble(chunks: list[Path], output: Path, gap_ms: int = 220) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    params = None
    with wave.open(str(output), "wb") as out:
        for index, path in enumerate(chunks):
            p, frames = read_wav(path.read_bytes())
            if params is None:
                params = p
                out.setnchannels(p.nchannels); out.setsampwidth(p.sampwidth); out.setframerate(p.framerate)
            elif (p.nchannels, p.sampwidth, p.framerate) != (params.nchannels, params.sampwidth, params.framerate):
                raise RuntimeError(f"Audio format mismatch in {path.name}")
            out.writeframes(frames)
            if index != len(chunks) - 1 and gap_ms:
                out.writeframes(b"\x00\x00" * int(24000 * gap_ms / 1000))

def upload_drive(path: Path, folder_id: str, service_json: str) -> str | None:
    if not service_json:
        return None
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request
    info = json.loads(service_json)
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/drive"]
    )
    creds.refresh(Request())
    headers = {"Authorization": f"Bearer {creds.token}"}
    safe_name = path.name.replace("'", "\\'")
    q = f"'{folder_id}' in parents and name = '{safe_name}' and trashed = false"
    found = requests.get(
        "https://www.googleapis.com/drive/v3/files",
        headers=headers, params={"q": q, "fields": "files(id,name)", "pageSize": 10}, timeout=60
    )
    found.raise_for_status()
    matches = found.json().get("files", [])
    content = path.read_bytes()
    if matches:
        file_id = matches[0]["id"]
        result = requests.patch(
            f"https://www.googleapis.com/upload/drive/v3/files/{file_id}",
            headers={**headers, "Content-Type": "audio/wav"},
            params={"uploadType": "media", "fields": "id,name,webViewLink"},
            data=content, timeout=300
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
            data=body, timeout=300
        )
    result.raise_for_status()
    return result.json().get("webViewLink") or result.json().get("id")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-speaker", type=int, default=int(os.getenv("START_SPEAKER", "6")))
    parser.add_argument("--end-speaker", type=int, default=int(os.getenv("END_SPEAKER", "10")))
    parser.add_argument("--source", default="data/speakers.json")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("GEMINI_API_KEY missing. Add the Gemini/Google AI Studio API key to GitHub Actions secrets.")
    source = json.loads(Path(args.source).read_text(encoding="utf-8"))
    speakers = source["speakers"]
    output = Path(args.output)
    chunks_root = output / "chunks"
    final_root = output / "final"
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "model": MODEL, "language": "Arabic", "sample_rate": 24000, "tracks": {}
    }
    session = requests.Session()
    for number, speaker in enumerate(speakers, start=1):
        if number < args.start_speaker or number > args.end_speaker:
            continue
        speaker_key = speaker["id"]
        speaker_dir = chunks_root / speaker_key
        speaker_dir.mkdir(parents=True, exist_ok=True)
        jobs = make_jobs(speaker)
        track_state = manifest["tracks"].setdefault(speaker_key, {
            "speaker": speaker["name"], "voice": speaker["voice"], "total_chunks": len(jobs),
            "completed_chunks": [], "final_file": None
        })
        track_state["total_chunks"] = len(jobs)
        print(f"TRACK {number}/10 {speaker['name']} voice={speaker['voice']} chunks={len(jobs)}", flush=True)
        for job_number, job in enumerate(jobs, start=1):
            chunk_path = speaker_dir / f"{job['job_id']}.wav"
            if chunk_path.exists() and chunk_path.stat().st_size > 44:
                if job["job_id"] not in track_state["completed_chunks"]:
                    track_state["completed_chunks"].append(job["job_id"])
                continue
            print(f"GENERATE {speaker_key} {job_number}/{len(jobs)} id={job['job_id']}", flush=True)
            audio = tts_request(session, api_key, speaker["voice"], job["text"], job["style"])
            read_wav(audio)
            tmp = chunk_path.with_suffix(".wav.tmp")
            tmp.write_bytes(audio)
            tmp.replace(chunk_path)
            if job["job_id"] not in track_state["completed_chunks"]:
                track_state["completed_chunks"].append(job["job_id"])
            atomic_json(manifest_path, manifest)
        ordered = [speaker_dir / f"{job['job_id']}.wav" for job in jobs]
        if not all(path.exists() for path in ordered):
            raise RuntimeError(f"Some chunks remain missing for {speaker_key}")
        final_path = final_root / f"{number:02d}_{speaker_key}_{speaker['voice']}.wav"
        if not final_path.exists():
            assemble(ordered, final_path)
        # Final validation before marking this speaker complete.
        with wave.open(str(final_path), "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()
            if wf.getnchannels() != 1 or wf.getframerate() != 24000 or wf.getsampwidth() != 2:
                raise RuntimeError(f"Final WAV validation failed for {final_path.name}")
        track_state["final_file"] = str(final_path)
        track_state["duration_seconds"] = round(duration, 2)
        track_state["status"] = "complete"
        atomic_json(manifest_path, manifest)
        print(f"TRACK_COMPLETE {final_path} duration={duration:.2f}s bytes={final_path.stat().st_size}", flush=True)
    # Drive upload is optional; GitHub Actions artifact remains the fallback.
    drive_folder = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "").strip()
    drive_service_json = os.getenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON", "").strip()
    if drive_folder and drive_service_json:
        for path in sorted(final_root.glob("*.wav")):
            link = upload_drive(path, drive_folder, drive_service_json)
            print(f"DRIVE_UPLOADED {path.name} {link}", flush=True)
    else:
        print("DRIVE_UPLOAD_SKIPPED: no service-account secret; WAV masters will be preserved as a workflow artifact.", flush=True)
    atomic_json(manifest_path, manifest)
    print("ALL_REQUESTED_SPEAKERS_COMPLETE", flush=True)

if __name__ == "__main__":
    main()
