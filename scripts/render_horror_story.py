#!/usr/bin/env python3
"""Resumable Arabic horror narration using Gemini 3.8 Flash TTS."""
from __future__ import annotations
import argparse, base64, hashlib, json, math, os, random, re, shutil, subprocess, sys, time, wave
from pathlib import Path
from typing import Any
import requests
import numpy as np

MODEL = os.getenv("TTS_MODEL", "gemini-3.8-flash-tts")
API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
VOICE = os.getenv("TTS_VOICE", "Fenrir")
SOURCE = Path(os.getenv("STORY_SOURCE", "story/caa_tahawil/story_segments.json"))
OUT = Path(os.getenv("OUTPUT_DIR", "output/horror-story"))
SR = 24000
MAX_CHARS = 1250
STYLE = {
 "baseline": "Warm, grounded Saudi/Gulf conversational male baritone, intimate close-mic storytelling, clear Arabic articulation, natural human cadence around 122-128 words per minute. Friendly and believable, never a radio-announcer voice.",
 "curiosity": "Same consistent male narrator and vocal identity. Slightly brighter conversational curiosity, natural Saudi/Gulf colloquial Arabic, measured pace around 120-125 words per minute, understated interest rather than theatrical excitement.",
 "suspense": "Same consistent male narrator. Lower, restrained chest register, slow 95-105 words per minute, quiet breath control and deliberate 1.5-2.2 second anticipation pauses. Eerie intimacy, dry and realistic, no monster imitation.",
 "whisper": "Same consistent male narrator. Very quiet intimate near-whisper, close-mic, breathy but intelligible Arabic, slow deliberate delivery around 95-105 words per minute. Keep the same underlying timbre; do not become a different character.",
 "panic": "Same consistent male narrator, now breathless and physically alarmed. Accelerate to approximately 150-160 words per minute with uneven but intelligible clauses, slight strained pitch lift, controlled gasps. Do not scream or become cartoonish.",
 "shock": "Same consistent male narrator. Cold, flat, dry, restrained revelation, slow precise wording, no melodic flourish. Let the silence carry the fear; do not exaggerate.",
 "relief": "Same consistent male narrator. Exhale and soften the delivery, warm but tired, measured Saudi/Gulf conversational tone, about 118-122 words per minute.",
 "reflective": "Same consistent male narrator. Deep, compassionate, reflective baritone around 115-120 words per minute, thoughtful pauses, natural moral gravity without sermonizing.",
}

def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def split_text(text: str, limit: int = MAX_CHARS) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return [text] if text else []
    sentences = re.split(r"(?<=[،؛؟.!])\s+", text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(sentence) > limit:
            words = sentence.split()
            for word in words:
                if current and len(current) + len(word) + 1 > limit:
                    chunks.append(current.strip())
                    current = ""
                current += (" " if current else "") + word
            continue
        if current and len(current) + len(sentence) + 1 > limit:
            chunks.append(current.strip())
            current = sentence
        else:
            current += (" " if current else "") + sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks

def extract_audio(payload: dict[str, Any]) -> bytes:
    audio = payload.get("output_audio") or {}
    if audio.get("data"):
        return base64.b64decode(audio["data"])
    for step in payload.get("steps", []):
        if step.get("type") == "model_output":
            for part in step.get("content", []):
                if part.get("type") == "audio" and part.get("data"):
                    return base64.b64decode(part["data"])
    raise RuntimeError("Gemini response contained no output_audio.data")

def validate_wav_bytes(data: bytes) -> float:
    with wave.open(__import__("io").BytesIO(data), "rb") as wf:
        params = wf.getparams()
        if (params.nchannels, params.sampwidth, params.framerate, params.comptype) != (1, 2, SR, "NONE"):
            raise RuntimeError(f"Unexpected Gemini WAV format: {params}")
        return wf.getnframes() / wf.getframerate()

def validate_wav_file(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        p = wf.getparams()
        if (p.nchannels, p.sampwidth, p.framerate, p.comptype) != (1, 2, SR, "NONE"):
            raise RuntimeError(f"Unexpected WAV format in {path.name}: {p}")
        if wf.getnframes() <= 0:
            raise RuntimeError(f"Empty WAV: {path}")
        return wf.getnframes() / wf.getframerate()

def request_audio(session: requests.Session, api_key: str, voice: str, style_key: str, text: str) -> bytes:
    style = STYLE.get(style_key, STYLE["baseline"])
    body = {
        "model": MODEL,
        "input": [{
            "type": "user_input",
            "content": [{
                "type": "text",
                "text": text,
                "annotations": [{"type": "speech_metadata", "style": style}],
            }],
        }],
        "response_format": {"type": "audio"},
        "generation_config": {"speech_config": [{"voice": voice}]},
    }
    last_error = None
    for attempt in range(1, 5):
        try:
            response = session.post(
                API_URL,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=body, timeout=(30, 600)
            )
            if response.status_code == 429:
                last_error = response.text[:2200]
                low = last_error.lower()
                if any(x in low for x in ("requests per day", "daily quota", "free_tier_requests", "quota exceeded")):
                    raise RuntimeError("Gemini daily quota exhausted: " + last_error)
                if attempt == 4:
                    break
                delay = min(75, 5 * (2 ** (attempt - 1)))
                print(f"RATE_LIMIT attempt={attempt}/4 wait={delay}s", flush=True)
                time.sleep(delay)
                continue
            if response.status_code in (400, 401, 403):
                raise RuntimeError(f"Gemini API rejected request ({response.status_code}): {response.text[:1800]}")
            if response.status_code >= 500:
                last_error = f"HTTP {response.status_code}: {response.text[:1800]}"
                if attempt == 4:
                    break
                time.sleep(min(75, 5 * (2 ** (attempt - 1))))
                continue
            response.raise_for_status()
            return extract_audio(response.json())
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = str(exc)
            if attempt == 4:
                break
            delay = min(75, 5 * (2 ** (attempt - 1)))
            print(f"NETWORK_RETRY attempt={attempt}/4 wait={delay}s", flush=True)
            time.sleep(delay)
    raise RuntimeError("TTS retries exhausted: " + str(last_error))

def build_chunk_plan(story: dict[str, Any]) -> list[dict[str, Any]]:
    plan = []
    for seg in story["segments"]:
        parts = split_text(seg["text"])
        for i, part in enumerate(parts, start=1):
            plan.append({
                "key": f"a{seg['act']:02d}_{seg['id']}_{i:02d}",
                "act": seg["act"], "segment_id": seg["id"], "style": seg["style"],
                "text": part, "pause_after_ms": seg.get("pause_after_ms", 0) if i == len(parts) else 180,
            })
    return plan

def concat_narration(plan: list[dict[str, Any]], chunk_paths: list[Path], output: Path) -> tuple[float, dict[int, tuple[float, float]]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    act_bounds: dict[int, list[float]] = {}
    elapsed = 0.0
    with wave.open(str(output), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(SR)
        for item, path in zip(plan, chunk_paths):
            duration = validate_wav_file(path)
            if item["act"] not in act_bounds:
                act_bounds[item["act"]] = [elapsed, elapsed]
            act_bounds[item["act"]][1] = elapsed + duration
            with wave.open(str(path), "rb") as wf:
                frames = wf.readframes(wf.getnframes())
            out.writeframes(frames)
            elapsed += duration
            gap_ms = int(item.get("pause_after_ms", 0))
            if gap_ms > 0:
                out.writeframes(b"\x00\x00" * int(SR * gap_ms / 1000))
                elapsed += gap_ms / 1000
    return elapsed, {k: (v[0], v[1]) for k, v in act_bounds.items()}

def make_ambience(path: Path, duration: int = 60) -> None:
    sr = 48000
    n = sr * duration
    rng = np.random.default_rng(4821)
    t = np.arange(n, dtype=np.float32) / sr
    slow_n = rng.normal(0, 1, int(duration * 500) + 2).astype(np.float32)
    slow_t = np.arange(len(slow_n), dtype=np.float32) / 500.0
    wind = np.interp(t, slow_t, slow_n).astype(np.float32)
    grit = rng.normal(0, 1, n).astype(np.float32)
    # Low desert air bed with two barely audible resonant components.
    ambient = (0.008 * wind + 0.00055 * grit +
               0.0015 * np.sin(2 * np.pi * 39.5 * t) +
               0.0007 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)
    # Two-second cosine crossfade to make the loop boundary unobtrusive.
    fade = int(2 * sr)
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    first = ambient[:fade].copy()
    last = ambient[-fade:].copy()
    ambient[:fade] = last * (1 - ramp) + first * ramp
    ambient[-fade:] = ambient[:fade]
    pcm = np.asarray(np.clip(ambient * 32767, -32768, 32767), dtype=np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

def effect_samples(kind: str, sr: int = 48000) -> np.ndarray:
    rng = np.random.default_rng({"knock": 18, "hoof": 29, "impact": 33}[kind])
    if kind == "knock":
        duration = 2.2
        y = np.zeros((int(sr * duration), 2), dtype=np.float32)
        for i, start in enumerate((0.08, 0.72, 1.38)):
            n = int(0.34 * sr); t = np.arange(n, dtype=np.float32) / sr
            env = np.exp(-t * 13.0)
            sig = 0.40 * np.sin(2*np.pi*(105+16*np.exp(-t*14))*t) * env
            sig += 0.10 * rng.normal(0,1,n).astype(np.float32) * np.exp(-t*45)
            at = int(start*sr); pan = (-0.45, 0.25, 0.5)[i]
            y[at:at+n,0] += sig * math.sqrt((1-pan)/2)
            y[at:at+n,1] += sig * math.sqrt((1+pan)/2)
    elif kind == "hoof":
        duration = 4.4
        y = np.zeros((int(sr * duration), 2), dtype=np.float32)
        for i, start in enumerate(np.arange(0.12, 3.95, 0.46)):
            n=int(0.22*sr); t=np.arange(n,dtype=np.float32)/sr
            env=np.exp(-t*20)
            sig=0.32*np.sin(2*np.pi*(58+10*np.exp(-t*12))*t)*env
            sig += 0.045*rng.normal(0,1,n).astype(np.float32)*np.exp(-t*35)
            at=int(start*sr); pan=math.sin(i*1.13)*0.78
            y[at:at+n,0]+=sig*math.sqrt((1-pan)/2)
            y[at:at+n,1]+=sig*math.sqrt((1+pan)/2)
    else:
        duration=1.0
        n=int(sr*duration); t=np.arange(n,dtype=np.float32)/sr
        env=np.exp(-t*8)
        low=0.55*np.sin(2*np.pi*(28+24*np.exp(-t*7))*t)*env
        metal=0.025*np.sin(2*np.pi*8000*t)*np.exp(-t*35)
        noise=0.012*rng.normal(0,1,n).astype(np.float32)*np.exp(-t*50)
        sig=(low+metal+noise).astype(np.float32)
        y=np.column_stack((sig,sig)).astype(np.float32)
    peak=max(1e-6,float(np.max(np.abs(y))))
    return y / peak

def make_foley(path: Path, total_duration: float, bounds: dict[int, tuple[float,float]]) -> None:
    sr=48000
    events=[]
    def add(kind, when, gain):
        if 0 <= when < total_duration:
            events.append((kind,float(when),effect_samples(kind),float(gain)))
    if 4 in bounds:
        s,e=bounds[4]; add("knock",s+18,0.24); add("hoof",s+(e-s)*0.53,0.16)
    if 5 in bounds:
        s,e=bounds[5]
        add("impact",s+66,0.28); add("hoof",s+165,0.19); add("knock",s+330,0.17)
    if 6 in bounds:
        s,e=bounds[6]; add("hoof",s+8,0.08)
    path.parent.mkdir(parents=True,exist_ok=True)
    seconds_per_block=15
    block_n=sr*seconds_per_block
    total_n=int(math.ceil(total_duration*sr))
    with wave.open(str(path),"wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sr)
        for base in range(0,total_n,block_n):
            n=min(block_n,total_n-base)
            block=np.zeros((n,2),dtype=np.float32)
            block_start=base/sr; block_end=(base+n)/sr
            for kind,when,samples,gain in events:
                ev_end=when+len(samples)/sr
                if ev_end<=block_start or when>=block_end: continue
                left=max(0,int((when-block_start)*sr))
                right=min(n,int((ev_end-block_start)*sr))
                src_left=max(0,int((block_start-when)*sr))
                src_right=src_left+(right-left)
                if right>left:
                    block[left:right]+=samples[src_left:src_right]*gain
            pcm=np.asarray(np.clip(block*32767,-32768,32767),dtype=np.int16)
            wf.writeframes(pcm.tobytes())

def run_ffmpeg(narration: Path, ambience: Path, foley: Path, flac: Path, mp3: Path) -> dict[str, Any]:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not installed")
    filter_complex = (
        "[0:a]aresample=48000,pan=stereo|c0=c0|c1=c0,volume=1.0[voice];"
        "[1:a]volume=0.23[amb0];"
        "[amb0][0:a]sidechaincompress=threshold=0.02:ratio=8:attack=25:release=450[amb];"
        "[2:a]volume=0.35[fx];"
        "[voice][amb][fx]amix=inputs=3:duration=first:normalize=0,"
        "loudnorm=I=-16:TP=-1.0:LRA=20:print_format=json[master]"
    )
    cmd=["ffmpeg","-y","-i",str(narration),"-stream_loop","-1","-i",str(ambience),"-i",str(foley),
         "-filter_complex",filter_complex,"-map","[master]","-ar","48000","-ac","2",
         "-c:a","flac","-sample_fmt","s32","-compression_level","5",str(flac)]
    p=subprocess.run(cmd,capture_output=True,text=True)
    if p.returncode:
        raise RuntimeError("FFmpeg FLAC mix failed: "+p.stderr[-3000:])
    # Save a distribution copy at 320 kb/s.
    p2=subprocess.run(["ffmpeg","-y","-i",str(flac),"-c:a","libmp3lame","-b:a","320k","-ar","48000",str(mp3)],
                      capture_output=True,text=True)
    if p2.returncode:
        raise RuntimeError("FFmpeg MP3 encode failed: "+p2.stderr[-2000:])
    # Retain the loudnorm analysis block for QA.
    match=re.findall(r"\{\s*\"input_i\"[\s\S]*?\}",p.stderr)
    return {"flac_bytes":flac.stat().st_size,"mp3_bytes":mp3.stat().st_size,"loudnorm":match[-1] if match else None}

def upload_drive(path: Path, folder_id: str, service_json: str) -> str | None:
    if not folder_id or not service_json:
        return None
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request
    info=json.loads(service_json)
    credentials=service_account.Credentials.from_service_account_info(
        info,scopes=["https://www.googleapis.com/auth/drive"])
    credentials.refresh(Request())
    headers={"Authorization":f"Bearer {credentials.token}"}
    q=f"'{folder_id}' in parents and name = '{path.name.replace(chr(39), chr(92)+chr(39))}' and trashed = false"
    files=requests.get("https://www.googleapis.com/drive/v3/files",headers=headers,
        params={"q":q,"fields":"files(id,name,webViewLink)","pageSize":10},timeout=60)
    files.raise_for_status(); found=files.json().get("files",[])
    if found:
        file_id=found[0]["id"]
        r=requests.patch(f"https://www.googleapis.com/upload/drive/v3/files/{file_id}",
            headers={**headers,"Content-Type":"application/octet-stream"},
            params={"uploadType":"media","fields":"id,name,webViewLink"},data=path.read_bytes(),timeout=900)
    else:
        boundary="horror-story-boundary"
        meta=json.dumps({"name":path.name,"mimeType":"application/octet-stream","parents":[folder_id]})
        body=(f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n{meta}\r\n"
              f"--{boundary}\r\nContent-Type: application/octet-stream\r\n\r\n").encode()+path.read_bytes()+f"\r\n--{boundary}--".encode()
        r=requests.post("https://www.googleapis.com/upload/drive/v3/files",headers={**headers,"Content-Type":f"multipart/related; boundary={boundary}"},
            params={"uploadType":"multipart","fields":"id,name,webViewLink"},data=body,timeout=900)
    r.raise_for_status()
    result=r.json()
    return result.get("webViewLink") or result.get("id")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",default=str(SOURCE))
    parser.add_argument("--output",default=str(OUT))
    args=parser.parse_args()
    api_key=os.getenv("GEMINI_API_KEY","").strip()
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is empty. Add GOOGLE_AI to MPC repository Actions secrets.")
    source=Path(args.source); out=Path(args.output)
    story=json.loads(source.read_text(encoding="utf-8"))
    if story.get("model") != MODEL:
        print(f"Using configured TTS model {MODEL}; story metadata says {story.get('model')}",flush=True)
    plan=build_chunk_plan(story)
    chunks_dir=out/"chunks"; chunks_dir.mkdir(parents=True,exist_ok=True)
    manifest_path=out/"manifest.json"
    story_hash=sha256(source)
    manifest=json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if manifest.get("story_sha256") not in (None,story_hash):
        raise RuntimeError("Story changed after checkpoint; use a fresh output directory.")
    manifest.update({"title":story["title"],"model":MODEL,"voice":VOICE,"story_sha256":story_hash,
                     "planned_chunks":len(plan),"completed_chunks":manifest.get("completed_chunks",{}),
                     "status":"in_progress","requested_words":sum(s.get("word_count",len(s["text"].split())) for s in story["segments"])})
    session=requests.Session()
    paths=[]
    for index,item in enumerate(plan,1):
        path=chunks_dir/(item["key"]+".wav")
        if path.exists() and path.stat().st_size>44:
            duration=validate_wav_file(path)
            manifest["completed_chunks"][item["key"]]={"path":str(path),"duration_seconds":round(duration,3),"sha256":sha256(path)}
            paths.append(path)
            continue
        print(f"TTS {index}/{len(plan)} act={item['act']} style={item['style']} chars={len(item['text'])}",flush=True)
        audio=request_audio(session,api_key,VOICE,item["style"],item["text"])
        duration=validate_wav_bytes(audio)
        tmp=path.with_suffix(".wav.tmp"); tmp.write_bytes(audio); tmp.replace(path)
        manifest["completed_chunks"][item["key"]]={"path":str(path),"duration_seconds":round(duration,3),"sha256":sha256(path)}
        paths.append(path)
        atomic_json(manifest_path,manifest)
    if len(paths)!=len(plan):
        raise RuntimeError("Not all chunks were synthesized")
    narration=out/"narration_clean_24k_mono.wav"
    duration,bounds=concat_narration(plan,paths,narration)
    atomic_json(out/"act_time_bounds.json",{"duration_seconds":duration,"acts":bounds})
    ambience=out/"tmp"/"desert_ambience_loop.wav"
    foley=out/"tmp"/"procedural_foley.wav"
    make_ambience(ambience)
    make_foley(foley,duration,bounds)
    flac=out/"final_master_48k_stereo_24bit.flac"
    mp3=out/"final_master_48k_stereo_320k.mp3"
    mix_report=run_ffmpeg(narration,ambience,foley,flac,mp3)
    # Verify duration and actual codec properties after render.
    probe=subprocess.run(["ffprobe","-v","error","-show_entries","format=duration,size:stream=codec_name,sample_rate,channels,bits_per_raw_sample","-of","json",str(flac)],
                         capture_output=True,text=True,check=True)
    flac_info=json.loads(probe.stdout)
    mp3_probe=subprocess.run(["ffprobe","-v","error","-show_entries","format=duration,size:stream=codec_name,sample_rate,channels,bit_rate","-of","json",str(mp3)],
                         capture_output=True,text=True,check=True)
    mp3_info=json.loads(mp3_probe.stdout)
    manifest.update({"status":"complete","completed_chunks":manifest["completed_chunks"],"duration_seconds":round(duration,2),
                     "actual_audio_chunks":len(plan),"narration_wav":{"path":str(narration),"bytes":narration.stat().st_size,"sha256":sha256(narration)},
                     "master_flac":{"path":str(flac),"bytes":flac.stat().st_size,"sha256":sha256(flac),"probe":flac_info},
                     "master_mp3":{"path":str(mp3),"bytes":mp3.stat().st_size,"sha256":sha256(mp3),"probe":mp3_info},
                     "mix_report":mix_report})
    atomic_json(manifest_path,manifest)
    folder=os.getenv("GOOGLE_DRIVE_FOLDER_ID","").strip()
    service=os.getenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON","").strip()
    links={}
    if folder and service:
        for p in (flac,mp3,manifest_path):
            try: links[p.name]=upload_drive(p,folder,service)
            except Exception as exc: print(f"DRIVE_UPLOAD_FAILED {p.name}: {exc}",flush=True)
    atomic_json(out/"delivery.json",{"status":"complete","drive_links":links,"manifest":"manifest.json"})
    print(json.dumps({"status":"complete","words":manifest["requested_words"],"chunks":len(plan),
                      "duration_seconds":duration,"flac_bytes":flac.stat().st_size,"mp3_bytes":mp3.stat().st_size,
                      "drive_uploaded":list(links.keys()),"flac_probe":flac_info,"mp3_probe":mp3_info},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
