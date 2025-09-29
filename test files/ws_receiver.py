import asyncio, json, base64, datetime, os, websockets
from concurrent.futures import ThreadPoolExecutor
import time
from collections import defaultdict, deque
import subprocess
import numpy as np

# ---------------- Config ----------------
TRANSCRIPT_FOLDER = "transcripts"
AUDIO_FOLDER = "audio_raw"
CLIPS_FOLDER = "clips"

AUDIO_SAMPLE_RATE = 16000   # Hz (Recall.ai format)
AUDIO_CHANNELS = 1
AUDIO_CODEC = "pcm_s16le"
SAMPLE_WIDTH = 2            # bytes (16-bit)
VIDEO_FPS = 2               # Recall.ai sends 2 FPS
CLIP_DURATION = 5.0         # seconds
# ----------------------------------------

for folder in [TRANSCRIPT_FOLDER, AUDIO_FOLDER, CLIPS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

executor = ThreadPoolExecutor(max_workers=4)

participant_data = defaultdict(lambda: {
    'frames': deque(),
    'audio_buffer': bytearray(),
    'last_audio_ts': None,
    'last_clip_time': None,
    'start_time': None
})


def save_transcript_sync(tfile, line):
    with open(tfile, "a", buffering=1) as f:   # line-buffered
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def flush_audio_to_wav(speaker, audio_bytes, start_time, clip_name):
    """Write aligned PCM buffer to WAV via ffmpeg."""
    raw_path = os.path.join(AUDIO_FOLDER, f"{clip_name}.raw")
    wav_path = os.path.join(CLIPS_FOLDER, f"{clip_name}_audio.wav")
    with open(raw_path, "wb") as f:
        f.write(audio_bytes)

    subprocess.run([
        'ffmpeg', '-y',
        '-f', 's16le',
        '-ar', str(AUDIO_SAMPLE_RATE),
        '-ac', str(AUDIO_CHANNELS),
        '-i', raw_path,
        '-t', str(CLIP_DURATION),
        '-c:a', AUDIO_CODEC,
        wav_path
    ], check=True, capture_output=True, text=True)

    os.remove(raw_path)
    print(f"🎵 Created audio clip: {wav_path}")
    return wav_path


def create_clips_sync(speaker, frames, audio_bytes, start_time, end_time):
    """Make 5-second audio + video clips per participant."""
    timestamp_str = datetime.datetime.fromtimestamp(start_time).strftime("%Y%m%d-%H%M%S")
    clip_name = f"{speaker}_{timestamp_str}".replace(" ", "_")

    temp_frames_dir = os.path.join(CLIPS_FOLDER, f"temp_{clip_name}_frames")
    os.makedirs(temp_frames_dir, exist_ok=True)

    try:
        # --- save frames ---
        frame_count = 0
        for frame_data, frame_time in frames:
            if start_time <= frame_time <= end_time:
                frame_path = os.path.join(temp_frames_dir, f"frame_{frame_count:04d}.png")
                with open(frame_path, "wb") as f:
                    f.write(base64.b64decode(frame_data))
                frame_count += 1

        if frame_count == 0:
            print(f"📹 No frames available for {speaker}")
            return

        # --- write raw audio ---
        raw_path = os.path.join(CLIPS_FOLDER, f"{clip_name}.raw")
        with open(raw_path, "wb") as f:
            f.write(audio_bytes)

        # convert raw → wav
        wav_path = os.path.join(CLIPS_FOLDER, f"{clip_name}.wav")
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "s16le",
            "-ar", str(AUDIO_SAMPLE_RATE),
            "-ac", str(AUDIO_CHANNELS),
            "-i", raw_path,
            "-t", str(CLIP_DURATION),
            wav_path
        ], check=True, capture_output=True, text=True)

        # --- make video only from frames ---
        frame_pattern = os.path.join(temp_frames_dir, "frame_%04d.png")
        video_temp = os.path.join(CLIPS_FOLDER, f"{clip_name}_temp.mp4")
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(VIDEO_FPS),
            "-i", frame_pattern,
            "-t", str(CLIP_DURATION),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            video_temp
        ], check=True, capture_output=True, text=True)

        # --- final mux (audio + video) ---
        final_clip = os.path.join(CLIPS_FOLDER, f"{clip_name}.mp4")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_temp,
            "-i", wav_path,
            "-c:v", "copy",     # no re-encode video
            "-c:a", "aac",      # encode audio
            "-shortest",        # stop at shortest stream
            final_clip
        ], check=True, capture_output=True, text=True)

        print(f"✅ Final clip saved: {final_clip}")

    finally:
        # cleanup temp files
        subprocess.run(["rm", "-rf", temp_frames_dir])
        if os.path.exists(raw_path): os.remove(raw_path)
        if os.path.exists(video_temp): os.remove(video_temp)
        if os.path.exists(wav_path): os.remove(wav_path)


def check_and_create_clips():
    now = time.time()
    for speaker, data in participant_data.items():
        if data['start_time'] is None:
            data['start_time'] = now
            data['last_clip_time'] = now
            continue
        if now - data['last_clip_time'] >= CLIP_DURATION:
            start, end = data['last_clip_time'], now
            relevant_frames = [f for f in data['frames'] if start <= f[1] <= end]

            if relevant_frames or len(data['audio_buffer']) > 0:
                loop = asyncio.get_running_loop()
                audio_bytes = bytes(data['audio_buffer'])
                loop.run_in_executor(
                    executor, create_clips_sync,
                    speaker, relevant_frames, audio_bytes, start, end
                )

            # reset
            data['last_clip_time'] = now
            data['frames'].clear()
            data['audio_buffer'].clear()
            data['last_audio_ts'] = None


async def handler(websocket):
    print("✅ Recall connected:", websocket.remote_address)

    async def clip_timer():
        while True:
            await asyncio.sleep(1.0)
            check_and_create_clips()

    clip_task = asyncio.create_task(clip_timer())

    try:
        async for message in websocket:
            try:
                event = json.loads(message)
            except Exception:
                continue

            evt_type = event.get("event")
            payload = (event.get("data") or {}).get("data") or {}
            participant = payload.get("participant", {})
            speaker = participant.get("name") or f"ID-{participant.get('id')}"
            ts_now = time.time()

            if evt_type == "video_separate_png.data":
                buf_b64 = payload.get("buffer", "")
                if buf_b64:
                    participant_data[speaker]['frames'].append((buf_b64, ts_now))

            elif evt_type == "audio_separate_raw.data":
                buf_b64 = payload.get("buffer", "")
                rel_ts = payload.get("timestamp", {}).get("relative")
                if buf_b64 and rel_ts is not None:
                    pcm_bytes = base64.b64decode(buf_b64)

                    data = participant_data[speaker]
                    if data['last_audio_ts'] is not None:
                        expected_samples = int((rel_ts - data['last_audio_ts']) * AUDIO_SAMPLE_RATE)
                        gap = expected_samples - (len(pcm_bytes) // SAMPLE_WIDTH)
                        if gap > 0:
                            # insert silence for missing samples
                            data['audio_buffer'].extend(b"\x00" * gap * SAMPLE_WIDTH)

                    data['audio_buffer'].extend(pcm_bytes)
                    data['last_audio_ts'] = rel_ts

            elif evt_type in ["transcript.data", "transcript.partial_data"]:
                words = [w.get("text", "") for w in payload.get("words", [])]
                partial = "(partial)" if evt_type == "transcript.partial_data" else ""
                line = f"{datetime.datetime.utcnow().isoformat()} {speaker}{partial}: {' '.join(words)}\n"

                # always show in terminal immediately
                print(line.strip(), flush=True)

                # only save finalized transcript lines to file
                if evt_type == "transcript.data":
                    tfile = os.path.join(TRANSCRIPT_FOLDER, f"{speaker}.txt")
                    loop = asyncio.get_running_loop()
                    loop.run_in_executor(executor, save_transcript_sync, tfile, line)
    finally:
        clip_task.cancel()


async def main():
    port = 8765
    async with websockets.serve(handler, "0.0.0.0", port):
        print(f"🚀 WS receiver listening on ws://0.0.0.0:{port}")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        executor.shutdown(wait=True)
