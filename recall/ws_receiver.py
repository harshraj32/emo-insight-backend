import asyncio
import base64
import datetime
import json
import os
import subprocess
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import WebSocket
from config import settings
from hume import hume_client
from hume.hume_summarize import summarize
from affina.coach import coach_feedback
import event_bus

executor = ThreadPoolExecutor(max_workers=4)

AUDIO_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
FPS = 2
CLIP_LEN = 5.0

# Buffer per participant (keyed by session_id:speaker)
participant_data = defaultdict(
    lambda: {
        "frames": deque(),
        "audio_buffer": bytearray(),
        "last_audio_ts": None,
        "last_clip_time": None,
        "start_time": None,
    }
)

def safe_summary(data):
    """Ensure data is JSON-serializable"""
    try:
        return json.loads(json.dumps(data))
    except Exception as e:
        return {"error": f"serialization failed: {e}"}

def create_clips_for_all_sync(session_id, participants_data, start, end):
    """
    Create and process clips for all participants in this time window.
    Returns summaries for each participant.
    """
    summaries = {}
    ts_str = datetime.datetime.fromtimestamp(start).strftime("%Y%m%d-%H%M%S")
    session_dir = os.path.join(settings.CLIPS_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    for speaker_name, data in participants_data.items():
        # Clean speaker name (remove any session prefix if accidentally included)
        clean_speaker = speaker_name.replace(f"{session_id}_", "") if speaker_name.startswith(f"{session_id}_") else speaker_name
        
        # Create unique clip names
        audio_clip_path = os.path.join(session_dir, f"{clean_speaker}_{ts_str}_audio.wav")
        video_clip_path = os.path.join(session_dir, f"{clean_speaker}_{ts_str}_video.mp4")
        temp_dir = os.path.join(session_dir, f"tmp_{clean_speaker}_{ts_str}")
        
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Initialize summaries for this speaker
            audio_summary = {"status": "no_data"}
            video_summary = {"status": "no_data"}

            # Process audio if available
            if len(data["audio_buffer"]) > 0:
                raw_path = os.path.join(temp_dir, "audio.raw")
                with open(raw_path, "wb") as f:
                    f.write(data["audio_buffer"])

                # Convert raw PCM to WAV
                try:
                    subprocess.run(
                        [
                            "ffmpeg", "-y",
                            "-f", "s16le",
                            "-ar", str(AUDIO_RATE),
                            "-ac", str(CHANNELS),
                            "-i", raw_path,
                            "-t", str(CLIP_LEN),
                            audio_clip_path,
                        ],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    
                    # Process through Hume
                    if os.path.exists(audio_clip_path) and os.path.getsize(audio_clip_path) > 0:
                        audio_results = hume_client.process_clip(
                            Path(audio_clip_path), 
                            models={"prosody": {"granularity": "utterance"}}
                        )
                        audio_summary = safe_summary(summarize(audio_results))
                        audio_summary["status"] = "processed"
                        print(f"✅ Audio processed for {clean_speaker}: {os.path.getsize(audio_clip_path)} bytes")
                    else:
                        audio_summary = {"status": "error", "error": "Empty audio file"}
                        
                except subprocess.CalledProcessError as e:
                    audio_summary = {"status": "error", "error": f"FFmpeg failed: {e.stderr}"}
                except Exception as e:
                    audio_summary = {"status": "error", "error": str(e)}

            # Process video if frames available
            frame_count = 0
            for frame_data, frame_time in data["frames"]:
                if start <= frame_time <= end:
                    frame_path = os.path.join(temp_dir, f"frame_{frame_count:04d}.png")
                    with open(frame_path, "wb") as f:
                        f.write(base64.b64decode(frame_data))
                    frame_count += 1

            if frame_count > 0:
                frame_pattern = os.path.join(temp_dir, "frame_%04d.png")
                
                try:
                    # Create video from frames
                    subprocess.run(
                        [
                            "ffmpeg", "-y",
                            "-framerate", str(FPS),
                            "-i", frame_pattern,
                            "-t", str(CLIP_LEN),
                            "-c:v", "libx264",
                            "-pix_fmt", "yuv420p",
                            video_clip_path,
                        ],
                        check=True,
                        capture_output=True,
                        text=True
                    )

                    # Process through Hume
                    if os.path.exists(video_clip_path) and os.path.getsize(video_clip_path) > 0:
                        video_results = hume_client.process_clip(
                            Path(video_clip_path), 
                            models={"face": {"fps_pred": 3}}
                        )
                        video_summary = safe_summary(summarize(video_results))
                        video_summary["status"] = "processed"
                        print(f"✅ Video processed for {clean_speaker}: {frame_count} frames")
                    else:
                        video_summary = {"status": "error", "error": "Empty video file"}
                        
                except subprocess.CalledProcessError as e:
                    video_summary = {"status": "error", "error": f"FFmpeg failed: {e.stderr}"}
                except Exception as e:
                    video_summary = {"status": "error", "error": str(e)}

            # Store summaries
            summaries[clean_speaker] = {
                "audio": audio_summary,
                "video": video_summary,
                "timestamp": ts_str
            }

        except Exception as e:
            summaries[clean_speaker] = {
                "audio": {"status": "error", "error": str(e)},
                "video": {"status": "error", "error": str(e)},
                "timestamp": ts_str
            }
            print(f"❌ Error processing {clean_speaker}: {e}")

        finally:
            # Cleanup temp directory
            if os.path.exists(temp_dir):
                subprocess.run(["rm", "-rf", temp_dir], capture_output=True)

    return summaries, ts_str

def check_and_create_clips(session_id):
    """
    Check if it's time to create clips for participants in this session.
    """
    now = time.time()
    participants_to_process = {}
    
    # Find all participants for this session
    session_prefix = f"{session_id}_"
    
    for key, data in list(participant_data.items()):
        if not key.startswith(session_prefix):
            continue
            
        # Extract speaker name from key
        speaker = key[len(session_prefix):]
        
        # Initialize timing if needed
        if data["start_time"] is None:
            data["start_time"] = now
            data["last_clip_time"] = now
            continue
        
        # Check if it's time to create a clip
        time_since_last = now - data["last_clip_time"]
        if time_since_last >= CLIP_LEN:
            clip_start = data["last_clip_time"]
            clip_end = now
            
            # Collect data for this time window
            relevant_frames = [(f, t) for f, t in data["frames"] if clip_start <= t <= clip_end]
            
            # Only process if we have data
            if relevant_frames or len(data["audio_buffer"]) > 0:
                participants_to_process[speaker] = {
                    "frames": list(relevant_frames),
                    "audio_buffer": bytes(data["audio_buffer"]),
                }
                print(f"📊 Queuing {speaker}: {len(relevant_frames)} frames, {len(data['audio_buffer'])} audio bytes")
            
            # Reset buffers for next clip
            data["frames"].clear()
            data["audio_buffer"].clear()
            data["last_clip_time"] = now
            data["last_audio_ts"] = None

    # Process clips if we have any
    if participants_to_process:
        print(f"🎬 Processing clips for {len(participants_to_process)} participants")
        
        # Run clip creation in executor
        loop = asyncio.get_running_loop()
        clip_start_time = now - CLIP_LEN
        
        # Create future for processing
        future = loop.run_in_executor(
            executor,
            create_clips_for_all_sync,
            session_id,
            participants_to_process,
            clip_start_time,
            now
        )
        
        # Handle results asynchronously
        async def process_results():
            try:
                summaries, ts_str = await future
                if summaries:
                    print(f"🎯 Got summaries for {len(summaries)} participants at {ts_str}")
                    await process_affina_feedback(session_id, summaries, ts_str)
            except Exception as e:
                print(f"❌ Error processing clips: {e}")
                import traceback
                traceback.print_exc()
        
        asyncio.create_task(process_results())
    
    print(f"⏱️ Clip timer ran for {session_id}")


async def process_affina_feedback(session_id, summaries, ts_str):
    """
    Send summaries to Affina coach and emit results.
    """
    try:
        sess = event_bus.sessions.get(session_id)
        if not sess:
            print(f"⚠️ Session {session_id} not found - creating placeholder session")
            # Create a minimal session to allow processing to continue
            event_bus.sessions[session_id] = {
                "user_name": "Unknown",
                "meeting_url": "Unknown",
                "objective": "No objective set",
                "emotions": ["concentration", "confusion", "boredom"],
                "bot_id": None,
                "created_at": time.time(),
                "phase": "pitch",
                "logs": ["Session recovered from orphaned bot"],
                "recent_events": [],
                "last_hume_summary": {},
            }
            sess = event_bus.sessions[session_id]
        
        # Now continue with normal processing
        context = {
            "phase": sess.get("phase", "pitch"),
            "objective": sess.get("objective", ""),
            "emotions": sess.get("emotions", []),
            "summaries": summaries,
        }
        
        # Emit emotion events
        for speaker, summary in summaries.items():
            if summary.get("audio", {}).get("status") == "processed" or \
               summary.get("video", {}).get("status") == "processed":
                emotion_data = {
                    "session_id": session_id,
                    "speaker": speaker,
                    "timestamp": ts_str,
                    "audio": summary.get("audio", {}),
                    "video": summary.get("video", {})
                }
                await event_bus.emit_emotion(emotion_data)
                print(f"📊 Emitted emotion for {speaker}: {summary.get('audio', {}).get('top_emotions', [])[:1]}")
        
        # Get recent transcript for context
        recent_transcript = ""
        for log in sess.get("logs", [])[-5:]:
            if "Transcript" in log and "partial" not in log:
                recent_transcript = log.split("Transcript", 1)[1].strip()
                break
        
        # Get coach feedback
        feedback = coach_feedback(context, recent_transcript or f"[No transcript at {ts_str}]")
        
        # Emit advice
        if isinstance(feedback, dict):
            advice = feedback.get("recommendation", json.dumps(feedback))
        else:
            advice = str(feedback)
        
        await event_bus.emit_advice(session_id, advice)
        print(f"💡 Affina advice: {advice[:100]}...")
        
        # Update logs
        sess["logs"].append(f"[{ts_str}] Processed {len(summaries)} speakers")
        sess["last_hume_summary"] = summaries
        await event_bus.emit_log(session_id, sess["logs"][-10:])
        
    except Exception as e:
        print(f"❌ Error in Affina processing: {e}")
        import traceback
        traceback.print_exc()

async def fastapi_handler(websocket: WebSocket):
    """
    Handle WebSocket connection from Recall bot.
    """
    await websocket.accept()
    
    # Get session_id from query params
    session_id = websocket.query_params.get("session_id")
    
    if not session_id:
        print("⚠️ WebSocket connected without session_id")
        await websocket.close(code=1008, reason="Missing session_id")
        return
    
    # Check if session exists, if not create a placeholder
    sess = event_bus.sessions.get(session_id)
    if not sess:
        print(f"⚠️ Unknown session_id: {session_id}, creating placeholder")
        event_bus.sessions[session_id] = {
            "user_name": "Unknown",
            "meeting_url": "Unknown", 
            "objective": "Session recovered",
            "emotions": ["concentration", "confusion", "boredom"],
            "bot_id": None,
            "created_at": time.time(),
            "phase": "pitch",
            "logs": [f"Bot connected with orphaned session {session_id}"],
            "recent_events": [],
            "last_hume_summary": {},
        }
        sess = event_bus.sessions[session_id]
    
    print(f"✅ Recall bot connected for session {session_id}")
    sess["logs"].append("Recall bot connected")
    await event_bus.emit_log(session_id, sess["logs"][-10:])
 