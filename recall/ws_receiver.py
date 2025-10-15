import asyncio
import base64
import datetime
import json
import logging
import os
import subprocess
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)
import event_bus
from affina.coach import coach_feedback_with_context
from affina import context_manager
from config import settings
from config import storage_utils
from fastapi import WebSocket
from hume import hume_client
from hume.hume_summarize import summarize_hume_batch

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
        clean_speaker = (
            speaker_name.replace(f"{session_id}_", "")
            if speaker_name.startswith(f"{session_id}_")
            else speaker_name
        )

        # Create unique clip names
        audio_clip_path = os.path.join(
            session_dir, f"{clean_speaker}_{ts_str}_audio.wav"
        )
        video_clip_path = os.path.join(
            session_dir, f"{clean_speaker}_{ts_str}_video.mp4"
        )
        temp_dir = os.path.join(session_dir, f"tmp_{clean_speaker}_{ts_str}")

        os.makedirs(temp_dir, exist_ok=True)

        try:
            audio_results = None
            video_results = None

            # Process audio if available
            if len(data["audio_buffer"]) > 0:
                raw_path = os.path.join(temp_dir, "audio.raw")
                with open(raw_path, "wb") as f:
                    f.write(data["audio_buffer"])

                try:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-f",
                            "s16le",
                            "-ar",
                            str(AUDIO_RATE),
                            "-ac",
                            str(CHANNELS),
                            "-i",
                            raw_path,
                            "-t",
                            str(CLIP_LEN),
                            audio_clip_path,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )

                    if (
                        os.path.exists(audio_clip_path)
                        and os.path.getsize(audio_clip_path) > 0
                    ):
                        audio_results = hume_client.process_clip(
                            Path(audio_clip_path),
                            models={"prosody": {"granularity": "utterance"}},
                        )
                        logger.debug(
                            f"✅ Audio processed for {clean_speaker}: {os.path.getsize(audio_clip_path)} bytes"
                        )

                        if isinstance(audio_results, list):
                            audio_results = audio_results[0]
                    else:
                        logger.warning(f"⚠️ Empty audio file for {clean_speaker}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"❌ FFmpeg audio failed for {clean_speaker}: {e.stderr}")
                except Exception as e:
                    logger.error(f"❌ Audio processing error for {clean_speaker}: {e}")

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
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-framerate",
                            str(FPS),
                            "-i",
                            frame_pattern,
                            "-t",
                            str(CLIP_LEN),
                            "-c:v",
                            "libx264",
                            "-pix_fmt",
                            "yuv420p",
                            video_clip_path,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )

                    if (
                        os.path.exists(video_clip_path)
                        and os.path.getsize(video_clip_path) > 0
                    ):
                        video_results = hume_client.process_clip(
                            Path(video_clip_path),
                            models={"face": {"fps_pred": 3}},
                        )
                        logger.debug(
                            f"✅ Video processed for {clean_speaker}: {frame_count} frames"
                        )

                        if isinstance(video_results, list):
                            video_results = video_results[0]
                    else:
                        logger.warning(f"⚠️ Empty video file for {clean_speaker}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"❌ FFmpeg video failed for {clean_speaker}: {e.stderr}")
                except Exception as e:
                    logger.error(f"❌ Video processing error for {clean_speaker}: {e}")

            # Build unified summary
            summary = summarize_hume_batch(
                audio_obj=audio_results,
                video_obj=video_results,
                participant=clean_speaker,
                timestamp=ts_str,
            )

            summaries[clean_speaker] = summary[clean_speaker]
            logger.debug(f"🎉 Summary created for {clean_speaker}")

            # Save transcript to disk        
        except Exception as e:
            summaries[clean_speaker] = {
                "audio": {"status": "error", "error": str(e)},
                "video": {"status": "error", "error": str(e)},
                "timestamp": ts_str,
            }
            logger.error(f"❌ Error processing {clean_speaker}: {e}")

        finally:
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
        speaker = key[len(session_prefix) :]

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
            relevant_frames = [
                (f, t) for f, t in data["frames"] if clip_start <= t <= clip_end
            ]

            # Only process if we have data
            if relevant_frames or len(data["audio_buffer"]) > 0:
                participants_to_process[speaker] = {
                    "frames": list(relevant_frames),
                    "audio_buffer": bytes(data["audio_buffer"]),
                }
                logger.debug(
                    f"📊 Queuing {speaker}: {len(relevant_frames)} frames, {len(data['audio_buffer'])} audio bytes"
                )

            # Reset buffers for next clip
            data["frames"].clear()
            data["audio_buffer"].clear()
            data["last_clip_time"] = now
            data["last_audio_ts"] = None

    # Process clips if we have any
    if participants_to_process:
        logger.info(f"🎬 Processing clips for {len(participants_to_process)} participants")

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
            now,
        )

        # Handle results asynchronously
        async def process_results():
            try:
                summaries, ts_str = await future
                if summaries:
                    logger.info(
                        f"🎯 Got summaries for {len(summaries)} participants at {ts_str}"
                    )
                    await process_affina_feedback(session_id, summaries, ts_str)
            except Exception as e:
                logger.error(f"❌ Error processing clips: {e}")
                import traceback
                traceback.print_exc()

        asyncio.create_task(process_results())


async def process_affina_feedback(session_id, summaries, ts_str):
    """
    Process emotion/transcript data and feed to context manager.
    Context manager decides when to call coach.
    """
    try:
        sess = event_bus.sessions.get(session_id)
        if not sess:
            logger.warning(f"⚠️ Session {session_id} not found for Affina processing")
            return

        sales_rep_name = sess.get("user_name", "Rep")

        # Get or create context for this session
        ctx = context_manager.get_or_create_context(
            session_id,
            sales_rep_name,
            sess.get("objective", ""),
            sess.get("phase", "pitch")
        )
        logger.info(
                        f"objective of the sales guy is {sess.get("objective")} and the name is {sales_rep_name} "
                    )
        # Update context metadata if changed
        ctx.update_metadata(
            phase=sess.get("phase"),
            objective=sess.get("objective")
        )

        # Add new data to context rolling windows
        for speaker, summary in summaries.items():
            # Save emotion trail to disk (already done in storage_utils)
            storage_utils.save_emotion_trail(session_id, speaker, ts_str, summary)
            
            # Add to context manager's rolling window
            emotion_entry = {
                "timestamp": ts_str,
                "datetime": datetime.datetime.now().isoformat(),
                "audio_emotions": summary.get("audio", {}).get("top_emotions", []),
                "video_emotions": summary.get("video", {}).get("top_emotions", []),
            }
            ctx.add_emotion_entry(speaker, emotion_entry)
            
        
        # Collect all emotions for batch emission to frontend
        batch_emotions = []
        for speaker, summary in summaries.items():
            audio_data = summary.get("audio", {})
            video_data = summary.get("video", {})

            if audio_data.get("status") == "ok" or video_data.get("status") == "ok":
                audio_emotions = audio_data.get("top_emotions", [])
                video_emotions = video_data.get("top_emotions", [])

                blended_label = storage_utils.get_blended_emotion_label(
                    video_emotions if video_emotions else audio_emotions
                )

                batch_emotions.append({
                    "speaker": speaker,
                    "timestamp": ts_str,
                    "audio": audio_data,
                    "video": video_data,
                    "is_sales_rep": sales_rep_name.lower() in speaker.lower(),
                    "blended_label": blended_label,
                })

        # Emit emotions to frontend
        if batch_emotions:
            await event_bus.emit_emotions_batch(session_id, batch_emotions)
            logger.info(f"📊 Sent batch of {len(batch_emotions)} emotions to frontend")

        # Check if context manager wants to provide coaching
        coaching_context = await context_manager.process_context_updates(session_id)
        
        if coaching_context and coaching_context.get('coaching_ready'):
            logger.info(f"🎯 Context manager triggered coaching for {session_id}")
            
            # Get feedback from coach
            feedback = coach_feedback_with_context(coaching_context)
            logger.debug(f"Coach feedback received: {feedback}")

            # Extract advice message
            if isinstance(feedback, dict):
                advice_message = feedback.get("feedback", "Processing.")
            else:
                advice_message = str(feedback)

            # Emit advice to frontend
            await event_bus.emit_advice(session_id, advice_message)

            # Update session logs
            sess["logs"].append(f"[{ts_str}] 🎯 Provided coaching advice")
            sess["last_coach_feedback"] = feedback
            
            await event_bus.emit_log(session_id, sess["logs"][-10:])
        else:
            logger.debug(f"Context not ready for coaching: {session_id}")

        # Update session state
        sess["last_hume_summary"] = summaries

    except Exception as e:
        logger.error(f"❌ Error in Affina processing: {e}")
        import traceback
        traceback.print_exc()
        try:
            await event_bus.emit_advice(
                session_id, f"Coach temporarily unavailable: {str(e)}"
            )
        except:
            pass


async def fastapi_handler(websocket: WebSocket):
    """Handle WebSocket connection from Recall bot."""
    await websocket.accept()
    
    session_id = websocket.query_params.get("session_id")
    
    if not session_id:
        logger.warning("⚠️ WebSocket connected without session_id")
        await websocket.close(code=1008, reason="Missing session_id")
        return
    
    sess = event_bus.sessions.get(session_id)
    if not sess:
        logger.warning(f"⚠️ Unknown session_id: {session_id}")
        await websocket.close(code=1008, reason="Unknown session")
        return
    
    logger.info(f"✅ Recall bot WebSocket connected for session {session_id}")
    sess["logs"].append("Bot connected - waiting to join meeting...")
    await event_bus.emit_log(session_id, sess["logs"][-10:])
    
    # Initialize context manager
    sales_rep_name = sess.get("user_name", "Rep")
    ctx = context_manager.get_or_create_context(
        session_id,
        sales_rep_name,
        sess.get("objective", ""),
        sess.get("phase", "pitch")
    )
    
    # Start clip processing timer
    async def clip_timer():
        while True:
            await asyncio.sleep(1.0)
            check_and_create_clips(session_id)
    
    clip_task = asyncio.create_task(clip_timer())
    
    try:
        while True:
            message = await websocket.receive_text()
            
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                continue
            
            evt_type = msg.get("event")
            data_wrapper = msg.get("data", {})
            payload = data_wrapper.get("data", {})
            
            # Debug logging for transcript events
            if evt_type and "transcript" in evt_type:
                logger.info(f"🎙️ TRANSCRIPT EVENT RECEIVED: {evt_type}")
                logger.debug(f"🎙️ Full payload: {json.dumps(payload, indent=2)[:500]}")
            
            # ===== Handle Transcript Events =====
            if evt_type == "transcript.data":
                participant = payload.get("participant", {})
                speaker = participant.get("name") or f"ID-{participant.get('id')}"
                
                # Extract text from words array
                words = payload.get("words", [])
                
                if words:
                    # Combine all words into one text string
                    transcript_text = " ".join([word.get("text", "") for word in words]).strip()
                    
                    if transcript_text:
                        # Generate timestamp
                        ts_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                        
                        logger.info(f"📝 Transcript from {speaker}: {transcript_text}")
                        
                        # Save to disk immediately
                        storage_utils.save_transcript_line(
                            session_id,
                            speaker,
                            ts_str,
                            transcript_text
                        )
                        
                        # Add to context manager
                        transcript_entry = {
                            "timestamp": ts_str,
                            "datetime": datetime.datetime.now().isoformat(),
                            "speaker": speaker,
                            "text": transcript_text,
                        }
                        ctx.add_transcript_entry(transcript_entry)
                        
                        # Log to session (show in frontend)
                        sess["logs"].append(f"💬 {speaker}: {transcript_text}")
                        await event_bus.emit_log(session_id, sess["logs"][-10:])
            
            # ===== Handle Partial Transcripts (Optional - for live captions) =====
            elif evt_type == "transcript.partial_data":
                participant = payload.get("participant", {})
                speaker = participant.get("name") or f"ID-{participant.get('id')}"
                
                words = payload.get("words", [])
                
                if words:
                    partial_text = " ".join([word.get("text", "") for word in words]).strip()
                    
                    if partial_text:
                        logger.debug(f"🎤 {speaker} (speaking): {partial_text[:50]}...")
            
            # ===== Handle Participant Events =====
            elif evt_type == "participant_events.join":
                participant = payload.get("participant", {})
                name = participant.get("name", "Unknown")
                is_host = participant.get("is_host", False)
                
                if is_host:
                    sess["logs"].append(f"✅ Host {name} joined - bot admitted!")
                else:
                    sess["logs"].append(f"👤 {name} joined the meeting")
                await event_bus.emit_log(session_id, sess["logs"][-10:])
                
            elif evt_type == "participant_events.leave":
                participant = payload.get("participant", {})
                name = participant.get("name", "Unknown")
                sess["logs"].append(f"👋 {name} left the meeting")
                await event_bus.emit_log(session_id, sess["logs"][-10:])
            
            elif evt_type == "participant_events.speech_on":
                participant = payload.get("participant", {})
                name = participant.get("name", "Unknown")
                sess["logs"].append(f"🎤 {name} started speaking")
                await event_bus.emit_log(session_id, sess["logs"][-10:])
            
            elif evt_type == "participant_events.webcam_on":
                participant = payload.get("participant", {})
                name = participant.get("name", "Unknown")
                sess["logs"].append(f"📹 {name} turned on camera")
                await event_bus.emit_log(session_id, sess["logs"][-10:])
            
            # ===== Handle Media Data Events (Keep for emotions) =====
            participant = payload.get("participant", {})
            speaker = participant.get("name") or f"ID-{participant.get('id')}"
            ts_now = time.time()
            participant_key = f"{session_id}_{speaker}"
            
            if evt_type == "video_separate_png.data":
                buf_b64 = payload.get("buffer")
                if buf_b64:
                    participant_data[participant_key]["frames"].append((buf_b64, ts_now))
            
            elif evt_type == "audio_separate_raw.data":
                buf_b64 = payload.get("buffer", "")
                rel_ts = payload.get("timestamp", {}).get("relative")

                if buf_b64 and rel_ts is not None:
                    pcm_bytes = base64.b64decode(buf_b64)
                    data = participant_data[participant_key]

                    # Gap fill
                    if data["last_audio_ts"] is not None:
                        expected_samples = int((rel_ts - data["last_audio_ts"]) * AUDIO_RATE)
                        gap = expected_samples - (len(pcm_bytes) // SAMPLE_WIDTH)
                        if gap > 0:
                            data["audio_buffer"].extend(b"\x00" * gap * SAMPLE_WIDTH)

                    data["audio_buffer"].extend(pcm_bytes)
                    data["last_audio_ts"] = rel_ts
            
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        sess["logs"].append(f"WebSocket error: {str(e)}")
        await event_bus.emit_log(session_id, sess["logs"][-10:])
    
    finally:
        clip_task.cancel()
        
        # Cleanup participant data
        session_prefix = f"{session_id}_"
        keys_to_remove = [k for k in participant_data.keys() if k.startswith(session_prefix)]
        for key in keys_to_remove:
            del participant_data[key]
        
        # Cleanup context manager
        context_manager.remove_context(session_id)
        
        logger.info(f"🔌 WebSocket disconnected for session {session_id}")
        await websocket.close()