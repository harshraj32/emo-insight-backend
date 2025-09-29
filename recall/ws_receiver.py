import asyncio, json, base64, datetime, os, subprocess, time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import websockets
from pathlib import Path

from config import settings
from hume import hume_client
from affina import affina_feedback

executor = ThreadPoolExecutor(max_workers=4)

AUDIO_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
FPS = 2
CLIP_LEN = 5.0

participant_data = defaultdict(lambda: {
    'frames': deque(),
    'audio_buffer': bytearray(),
    'last_clip_time': None,
    'start_time': None
})


def create_clip(speaker, frames, audio_bytes, start, end):
    ts_str = datetime.datetime.fromtimestamp(start).strftime("%Y%m%d-%H%M%S")
    clip_name = f"{speaker}_{ts_str}"
    temp_dir = os.path.join(settings.CLIPS_DIR, f"tmp_{clip_name}")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # save frames
        for i, (frame_data, frame_time) in enumerate(frames):
            if start <= frame_time <= end:
                with open(os.path.join(temp_dir, f"frame_{i:04d}.png"), "wb") as f:
                    f.write(base64.b64decode(frame_data))

        # save audio.raw
        raw_path = os.path.join(temp_dir, "audio.raw")
        with open(raw_path, "wb") as f:
            f.write(audio_bytes)

        wav_path = os.path.join(temp_dir, "audio.wav")
        subprocess.run([
            "ffmpeg", "-y", "-f", "s16le", "-ar", str(AUDIO_RATE), "-ac", str(CHANNELS),
            "-i", raw_path, wav_path
        ], check=True, capture_output=True)

        # combine → mp4
        mp4_out = os.path.join(settings.CLIPS_DIR, f"{clip_name}.mp4")
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(FPS), "-i", os.path.join(temp_dir, "frame_%04d.png"),
            "-i", wav_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", mp4_out
        ], check=True, capture_output=True)

        print(f"🎬 Saved clip {mp4_out}")

        # ---- send to Hume ----
        results = hume_client.process_clip(Path(mp4_out), models={"prosody": {}, "face": {}})

        # ---- live feedback with Affina ----
        feedback = affina_feedback.generate_feedback(
            sales_rep=settings.SALES_REP_NAME,
            speaker=speaker,
            results=results
        )
        print(f"💡 Affina: {feedback}")

        # TODO: emit feedback to frontend (WebSocket/REST)
        os.remove(mp4_out)

    finally:
        subprocess.run(["rm", "-rf", temp_dir])


def check_clips():
    now = time.time()
    for speaker, data in participant_data.items():
        if data['start_time'] is None:
            data['start_time'] = now
            data['last_clip_time'] = now
            continue
        if now - data['last_clip_time'] >= CLIP_LEN:
            start, end = data['last_clip_time'], now
            frames = [f for f in data['frames'] if start <= f[1] <= end]
            audio_bytes = bytes(data['audio_buffer'])
            if frames or audio_bytes:
                loop = asyncio.get_running_loop()
                loop.run_in_executor(executor, create_clip, speaker, frames, audio_bytes, start, end)
            data['frames'].clear()
            data['audio_buffer'].clear()
            data['last_clip_time'] = now


async def handler(ws):
    print("✅ Recall connected:", ws.remote_address)
    async def clip_timer():
        while True:
            await asyncio.sleep(1)
            check_clips()
    clip_task = asyncio.create_task(clip_timer())
    try:
        async for msg in ws:
            event = json.loads(msg)
            evt_type = event.get("event")
            payload = (event.get("data") or {}).get("data") or {}
            speaker = payload.get("participant", {}).get("name") or "unknown"
            ts_now = time.time()
            if evt_type == "video_separate_png.data":
                buf = payload.get("buffer")
                if buf:
                    participant_data[speaker]['frames'].append((buf, ts_now))
            elif evt_type == "audio_separate_raw.data":
                buf = payload.get("buffer")
                if buf:
                    pcm = base64.b64decode(buf)
                    participant_data[speaker]['audio_buffer'].extend(pcm)
            elif evt_type.startswith("transcript"):
                print(f"📝 Transcript: {speaker}: {payload}")
    finally:
        clip_task.cancel()


async def main():
    async with websockets.serve(handler, "0.0.0.0", settings.WS_RECEIVER_PORT):
        print(f"🚀 WS receiver on ws://0.0.0.0:{settings.WS_RECEIVER_PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
