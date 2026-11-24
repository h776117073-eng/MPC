#!/usr/bin/env python3
from __future__ import annotations
import argparse, base64, hashlib, json, os, re, sys, time, wave
from pathlib import Path
import requests

API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts")
RATE = 24000
VOICE_MAP = {
 "speaker_01":"Leda","speaker_02":"Alnilam","speaker_03":"Gacrux","speaker_04":"Charon",
 "speaker_05":"Sulafat","speaker_06":"Fenrir","speaker_07":"Zephyr","speaker_08":"Sadachbia",
 "speaker_09":"Rasalgethi","speaker_10":"Kore","speaker_11":"Algenib","speaker_12":"Orus",
 "speaker_13":"Despina","speaker_14":"Sadaltager","speaker_15":"Schedar","speaker_16":"Laomedeia",
 "speaker_17":"Iapetus"
}
class QuotaExhausted(Exception): pass

def load(p):
 return json.loads(Path(p).read_text(encoding="utf-8"))

def atomic_json(p, obj):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+".tmp")
 t.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8"); t.replace(p)

def sha256(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
 return h.hexdigest()

def find_parts(root):
 folder=Path(root)/"tts_ready"
 rows=[]
 for p in folder.glob("speaker_*/part_*.json"):
  m=load(p); a=re.search(r"speaker_(\d+)",p.as_posix()); b=re.search(r"part_(\d+)",p.stem)
  if not a or not b or not isinstance(m.get("TTS_ITEMS"),list): raise ValueError("Invalid TTS part: "+p.name)
  rows.append((int(a.group(1)),int(b.group(1)),p,m))
 rows.sort(key=lambda x:(x[0],x[1]))
 if not rows: raise ValueError("No TTS parts found")
 return [(x[2],x[3]) for x in rows]

def validate(root, parts):
 root=Path(root); q=load(root/"final_quality_report.json"); source=load(root/"source_analysis.json")
 if q.get("status")!="PASSED": raise ValueError("Dataset QA status is not PASSED")
 checks=q.get("checks",{})
 expected={"extracted_equals_distributed":True,"empty_arabic_segments_count":0,
 "missing_speaker_id_count":0,"inverted_timestamp_count":0,
 "duplicate_segments_across_speakers_count":0,"full_timeline_covered_to_final_dialogue":True}
 for k,v in expected.items():
  if checks.get(k)!=v: raise ValueError("Dataset QA check failed: "+k)
 ids=set()
 for path,p in parts:
  est=float(p.get("estimated_tts_duration_sec",0)); ceiling=float(p.get("operational_ceiling_sec",450)); hard=float(p.get("hard_cap_sec",480))
  items=p["TTS_ITEMS"]
  if p.get("within_limit") is not True or not 0<est<=ceiling or est>hard: raise ValueError("TTS duration limit failed: "+path.name)
  if not items or len(items)!=int(p.get("segment_count",-1)): raise ValueError("Segment count mismatch: "+path.name)
  for it in items:
   sid=str(it.get("segment_id","")).strip()
   if not sid or sid in ids: raise ValueError("Missing/duplicate segment ID: "+sid)
   ids.add(sid)
   if not str(it.get("spoken_text","")).strip(): raise ValueError("Empty spoken text: "+sid)
   if float(it.get("end",0))<=float(it.get("start",0)): raise ValueError("Bad source time range: "+sid)
 return source,q,len(ids)

def style_for(part,item,first):
 m=item.get("speech_metadata") or {}; cue=[]
 if m.get("emotion"):
  s="Emotion: "+str(m["emotion"])
  if m.get("emotion_intensity") is not None: s+=" (intensity "+str(round(float(m["emotion_intensity"])*100))+" percent)"
  cue.append(s)
 for k,label in [("speaking_style","Delivery"),("speaking_rate","Rate"),("pitch_direction","Pitch"),("volume","Volume")]:
  if m.get(k): cue.append(label+": "+str(m[k]))
 if m.get("vocal_events"): cue.append("Vocal cues: "+", ".join(map(str,m["vocal_events"])))
 if m.get("pause_before_ms"): cue.append("Brief pause before line, about "+str(m["pause_before_ms"])+" ms")
 if m.get("pause_after_ms"): cue.append("Brief pause after line, about "+str(m["pause_after_ms"])+" ms")
 text=("Speak the supplied Arabic words verbatim in natural Modern Standard Arabic. Do not translate, "
       "paraphrase, omit, add or explain words. Maintain a consistent character voice.")
 if first:
  text+=" Character design: "+str(part.get("VOICE_DESIGN_PROMPT",""))
  rules=part.get("VOICE_CONSISTENCY_RULES") or []
  if rules: text+=" Consistency rules: "+" ".join(map(str,rules))
 text+=" Current line direction: "+("; ".join(cue) if cue else "clear natural neutral delivery")+"."
 return text

def request_body(part,speaker_id):
 if speaker_id not in VOICE_MAP: raise ValueError("No voice map for "+speaker_id)
 content=[]
 for i,it in enumerate(part["TTS_ITEMS"]):
  content.append({"type":"text","text":str(it["spoken_text"]),
   "annotations":[{"type":"speech_metadata","style":style_for(part,it,i==0)}]})
 return {"model":MODEL,"input":[{"type":"user_input","content":content}],
         "response_format":{"type":"audio"},
         "generation_config":{"speech_config":[{"voice":VOICE_MAP[speaker_id]}]}}

def get_audio(data):
 a=data.get("output_audio") or {}
 if isinstance(a,dict) and a.get("data"): return base64.b64decode(a["data"]),a.get("mime_type","audio/wav"),data.get("id")
 for step in data.get("steps",[]):
  if step.get("type")=="model_output":
   for item in step.get("content",[]):
    if item.get("type") in ("audio","output_audio") and item.get("data"):
     return base64.b64decode(item["data"]),item.get("mime_type","audio/wav"),data.get("id")
 raise RuntimeError("Interactions API returned no audio payload")

def call_api(session,key,body,pid):
 for attempt in range(1,5):
  try:
   r=session.post(API_URL,headers={"x-goog-api-key":key,"Content-Type":"application/json"},json=body,timeout=(30,900))
  except (requests.Timeout,requests.ConnectionError) as e:
   if attempt==4: raise RuntimeError(pid+": network retries exhausted ("+type(e).__name__+")")
   delay=min(60,5*2**(attempt-1)); print("Temporary network issue; retrying in",delay,"s",flush=True); time.sleep(delay); continue
  if r.status_code==429:
   low=r.text.lower()
   if any(x in low for x in ("requests per day","per_day","daily quota","resource_exhausted","quota exceeded")):
    raise QuotaExhausted(pid+": Google AI daily quota exhausted (429)")
   if attempt==4: raise RuntimeError(pid+": rate limit persisted (429)")
   delay=min(60,5*2**(attempt-1)); print("Temporary rate limit; retrying in",delay,"s",flush=True); time.sleep(delay); continue
  if r.status_code in (401,403): raise RuntimeError(pid+": Google AI rejected the API key or permission ("+str(r.status_code)+")")
  if r.status_code==400: raise RuntimeError(pid+": Gemini rejected request/model (400)")
  if r.status_code>=500:
   if attempt==4: raise RuntimeError(pid+": Gemini server error persisted ("+str(r.status_code)+")")
   delay=min(60,5*2**(attempt-1)); time.sleep(delay); continue
  if not r.ok: raise RuntimeError(pid+": API error ("+str(r.status_code)+")")
  return get_audio(r.json())
 raise RuntimeError(pid+": API retry limit reached")

def write_wav(audio,dest):
 dest=Path(dest); dest.parent.mkdir(parents=True,exist_ok=True); tmp=dest.with_suffix(".wav.tmp")
 if audio[:4]==b"RIFF": tmp.write_bytes(audio)
 else:
  with wave.open(str(tmp),"wb") as w:
   w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(audio)
 tmp.replace(dest)
 with wave.open(str(dest),"rb") as w:
  p=w.getparams()
  if (p.nchannels,p.sampwidth,p.framerate)!=(1,2,RATE): raise ValueError("Unexpected WAV format: "+str(p))
  duration=w.getnframes()/w.getframerate()
 if duration<=0.1: raise ValueError("Generated WAV is too short")
 return {"duration_seconds":round(duration,3),"bytes":dest.stat().st_size,"sha256":sha256(dest),
         "sample_rate":RATE,"channels":1,"sample_width_bytes":2}

def make_masters(out,manifest):
 groups={}
 for pid,s in manifest.get("parts",{}).items():
  src=Path(out)/"parts"/(pid+".wav")
  if s.get("status")=="complete" and src.exists(): groups.setdefault(s["speaker_id"],[]).append((int(s["part_index"]),src))
 masters={}; folder=Path(out)/"master_by_speaker"; folder.mkdir(parents=True,exist_ok=True)
 for sid,rows in groups.items():
  rows.sort(key=lambda x:x[0]); dst=folder/(sid+"_"+VOICE_MAP[sid]+".wav"); tmp=dst.with_suffix(".wav.tmp")
  with wave.open(str(rows[0][1]),"rb") as first:
   params=first.getparams()
   with wave.open(str(tmp),"wb") as outwav:
    outwav.setparams(params)
    for _,src in rows:
     with wave.open(str(src),"rb") as r:
      if r.getparams()!=params: raise ValueError("WAV parameter mismatch for "+sid)
      outwav.writeframes(r.readframes(r.getnframes()))
  tmp.replace(dst)
  with wave.open(str(dst),"rb") as r: dur=r.getnframes()/r.getframerate()
  masters[sid]={"file":str(dst.relative_to(out)),"part_count":len(rows),"duration_seconds":round(dur,3),
   "bytes":dst.stat().st_size,"sha256":sha256(dst),"note":"Concatenated speaker stem; not video-synchronized."}
 manifest["speaker_masters"]=masters

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--dataset-root",default="dataset/ARABIC_DUBBING_DATASET")
 ap.add_argument("--output",default="output/dataset_dub")
 ap.add_argument("--start-part",type=int,default=1); ap.add_argument("--end-part",type=int,default=12)
 ap.add_argument("--dry-run",action="store_true"); args=ap.parse_args()
 key=os.getenv("GEMINI_API_KEY","").strip()
 if not args.dry_run and not key: raise SystemExit("GEMINI_API_KEY is not configured")
 parts=find_parts(args.dataset_root); source,quality,nseg=validate(args.dataset_root,parts)
 if args.start_part<1 or args.end_part<args.start_part or args.end_part>len(parts):
  raise SystemExit("Invalid part range; dataset contains "+str(len(parts))+" parts")
 selected=parts[args.start_part-1:args.end_part]
 if args.dry_run:
  print(json.dumps({"status":"DRY_RUN_OK","model":MODEL,"parts":len(parts),"selected":len(selected),
   "segments":nseg,"speakers":source.get("coverage_summary",{}).get("unique_speakers"),
   "voice_map":VOICE_MAP},ensure_ascii=False,indent=2)); return 0
 out=Path(args.output); out.mkdir(parents=True,exist_ok=True); mp=out/"manifest.json"
 if mp.exists(): manifest=load(mp)
 else:
  manifest={"model":MODEL,"source_video_id":source.get("video_metadata",{}).get("video_id"),
   "source_duration_seconds":source.get("video_metadata",{}).get("duration_sec"),
   "dataset_qa":{"status":quality.get("status"),"segments":nseg,"parts":len(parts)},
   "voice_map":VOICE_MAP,"mixed_role_speakers":["speaker_15","speaker_16","speaker_17"],
   "parts":{},"speaker_masters":{},"notes":[
    "Gemini prebuilt voices; no cloning of source actors.",
    "speaker_15, speaker_16 and speaker_17 each combine several real roles.",
    "WAV speaker masters concatenate parts and are not synchronized to video timecodes.",
    "Voice and emotion profiles are contextual estimates; source audio/video were not in the dataset archive."]}
 session=requests.Session(); quota=False
 for index,(path,part) in enumerate(selected,start=args.start_part):
  pid=str(part.get("part_id") or path.stem); sid=str(part.get("speaker_id") or "")
  dest=out/"parts"/(pid+".wav")
  if manifest.get("parts",{}).get(pid,{}).get("status")=="complete" and dest.exists():
   print("SKIP_ALREADY_COMPLETE",pid,flush=True); continue
  manifest.setdefault("parts",{})[pid]={"status":"in_progress","speaker_id":sid,
   "part_index":int(part.get("part_index",0)),"global_part_index":index,
   "segment_count":len(part["TTS_ITEMS"]),"video_start_sec":part.get("video_start_sec"),
   "video_end_sec":part.get("video_end_sec"),"estimated_tts_duration_sec":part.get("estimated_tts_duration_sec"),
   "voice":VOICE_MAP.get(sid)}
  atomic_json(mp,manifest)
  print("START_PART",str(index)+"/"+str(len(parts)),pid,"speaker="+sid,"voice="+str(VOICE_MAP.get(sid)),
   "segments="+str(len(part["TTS_ITEMS"])),flush=True)
  try:
   audio,mime,iid=call_api(session,key,request_body(part,sid),pid)
   meta=write_wav(audio,dest)
   manifest["parts"][pid].update({"status":"complete","mime_type":mime,"interaction_id":iid,**meta})
   print("COMPLETE_PART",pid,"duration="+str(meta["duration_seconds"])+"s",flush=True)
   atomic_json(mp,manifest)
  except QuotaExhausted as e:
   manifest["parts"][pid].update({"status":"quota_exhausted","error":str(e)})
   atomic_json(mp,manifest); quota=True
   print("QUOTA_EXHAUSTED_STOPPED; completed parts preserved.",flush=True); break
  except Exception as e:
   manifest["parts"][pid].update({"status":"failed","error":type(e).__name__+": "+str(e)[:400]})
   atomic_json(mp,manifest); raise
 make_masters(out,manifest)
 complete=sum(1 for x in manifest["parts"].values() if x.get("status")=="complete")
 summary={"completed_parts":complete,"total_parts":len(parts),"remaining_parts":len(parts)-complete,
  "failed_parts":sum(1 for x in manifest["parts"].values() if x.get("status")=="failed"),
  "quota_exhausted_stop":quota,"speaker_masters_ready":len(manifest.get("speaker_masters",{})),
  "warning":"Speaker WAV stems are not yet synchronized/mixed with source video."}
 manifest["run_summary"]=summary; atomic_json(mp,manifest); atomic_json(out/"summary.json",summary)
 print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True); return 0

if __name__=="__main__":
 try: raise SystemExit(main())
 except Exception as exc:
  print("FATAL:",type(exc).__name__,str(exc),file=sys.stderr,flush=True); raise
