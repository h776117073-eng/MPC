# MPC — Arabic multi-speaker dubbing

This repository contains the supplied Arabic dialogue scripts and a resumable Gemini TTS runner for the 01:42:35 Chinese drama.

## Ten isolated voice tracks

1. لي يانغ — Leda
2. لو مينغ — Achird
3. تشوانغ شيان يوي — Kore
4. فو شاوتشينغ — Puck
5. شيا مو يان — Zephyr
6. لي هواي شين — Fenrir
7. والدة لي يانغ — Aoede
8. لو تشن هونغ — Charon
9. الأدوار النسائية الثانوية — Callirrhoe
10. الأدوار الرجالية الثانوية — Iapetus

The ten tracks represent fourteen speaking roles. The first five WAV masters were uploaded earlier to the Drive destination folder, so the workflow defaults to speakers 6–10.

## Required GitHub Actions secret

Add the Google AI Studio API key as one of these repository Actions secrets (checked in order):
- `GEMINI_API_KEY` (recommended)
- `GOOGLE_AI_STUDIO_API_KEY`
- `GOOGLE_API_KEY`
- `GOOGLE_STUDIO_API_KEY`

The workflow never prints the secret value.

## Optional automatic Google Drive upload

For direct upload from GitHub Actions, add:
- `GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON` (JSON service-account key)
- Optionally `GOOGLE_DRIVE_FOLDER_ID`; defaults to the existing folder ID.

The service-account email must have Editor access to that Drive folder. Without this optional credential, complete WAV masters and the checkpoint manifest are still saved as a GitHub Actions artifact for transfer to Drive.

## Resume and output

- Pushes affecting the scripts or runner start speakers 6–10.
- Use Actions → Arabic dubbing - Gemini 3.8 Flash TTS → Run workflow to choose a speaker range.
- Completed per-utterance WAV chunks are cached; retries skip valid chunks and synthesize only missing pieces.
- Each speaker is exported as a separate mono 24 kHz / 16-bit WAV master.
- Emotion cues are sent as TTS style metadata; sigh, gasp, and laugh cues remain inline.
- The scripts contain no per-scene timecodes. These are character-separated voice tracks, not a synchronized video mix. Timing alignment against the 01:42:35 picture is a separate stage.
