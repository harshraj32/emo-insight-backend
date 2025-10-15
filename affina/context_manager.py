"""
Context Manager for Affina coaching system.
Manages rolling windows, summaries, and context preparation.
"""

import json
import time
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from config import storage_utils
from affina.summarizer import summarize_window, create_cumulative_summary

logger = logging.getLogger(__name__)

# Rolling window configuration
WINDOW_SIZE_SECONDS = 120  # 2 minutes
SUMMARY_INTERVAL_SECONDS = 30  # Summarize every 30 seconds

# Storage for context state per session
session_contexts = {}


class SessionContext:
    """
    Manages context for a single session.
    Maintains rolling windows and summaries.
    """
    
    def __init__(self, session_id: str, sales_rep_name: str, objective: str, phase: str):
        self.session_id = session_id
        self.sales_rep_name = sales_rep_name
        self.objective = objective
        self.phase = phase
        
        # Rolling windows (store last 2 minutes)
        self.transcript_window = []  # List of transcript entries
        self.emotion_window = {}     # {speaker: [emotion entries]}
        
        # Historical summaries (everything before current window)
        self.summaries = []  # List of summary dicts
        
        # Timing
        self.last_summary_time = time.time()
        self.window_start_time = time.time()
        
        # Storage paths
        self.session_dir = storage_utils.ensure_session_dir(session_id)
        self.summary_file = self.session_dir / "summaries.jsonl"
        
        logger.info(f"📋 Context manager initialized for session {session_id}")
    
    def update_metadata(self, phase: Optional[str] = None, objective: Optional[str] = None):
        """Update session metadata."""
        if phase:
            self.phase = phase
        if objective:
            self.objective = objective
    
    def add_transcript_entry(self, entry: dict):
        """Add transcript entry to rolling window."""
        # Add timestamp for window management
        entry['_added_at'] = time.time()
        self.transcript_window.append(entry)
        self._trim_transcript_window()
    
    def add_emotion_entry(self, speaker: str, entry: dict):
        """Add emotion entry to rolling window."""
        if speaker not in self.emotion_window:
            self.emotion_window[speaker] = []
        
        entry['_added_at'] = time.time()
        self.emotion_window[speaker].append(entry)
        self._trim_emotion_window(speaker)
    
    def _trim_transcript_window(self):
        """Remove entries older than WINDOW_SIZE_SECONDS."""
        cutoff = time.time() - WINDOW_SIZE_SECONDS
        self.transcript_window = [
            e for e in self.transcript_window 
            if e.get('_added_at', 0) > cutoff
        ]
    
    def _trim_emotion_window(self, speaker: str):
        """Remove emotion entries older than WINDOW_SIZE_SECONDS."""
        cutoff = time.time() - WINDOW_SIZE_SECONDS
        self.emotion_window[speaker] = [
            e for e in self.emotion_window[speaker] 
            if e.get('_added_at', 0) > cutoff
        ]
    
    def should_summarize(self) -> bool:
        """Check if it's time to create a summary."""
        elapsed = time.time() - self.last_summary_time
        return elapsed >= SUMMARY_INTERVAL_SECONDS
    
    async def create_summary(self) -> Optional[dict]:
        """
        Create a summary of the current window.
        Returns None if window is empty.
        """
        
        # Check if we have any data
        if not self.transcript_window and not self.emotion_window:
            logger.debug(f"[{self.session_id}] No data in window, skipping summary")
            return None
        
        try:
            # Run summarization
            summary = await asyncio.get_event_loop().run_in_executor(
                None,
                summarize_window,
                self.transcript_window,
                self.emotion_window,
                self.sales_rep_name,
                self.objective,
                self.phase
            )
            
            # Add metadata
            summary['timestamp'] = datetime.now().isoformat()
            summary['window_start'] = self.window_start_time
            summary['window_end'] = time.time()
            
            # Save to disk
            with open(self.summary_file, 'a') as f:
                f.write(json.dumps(summary) + '\n')
            
            # Add to summaries list
            self.summaries.append(summary)
            
            # Update timing
            self.last_summary_time = time.time()
            self.window_start_time = time.time()
            
            logger.info(
                f"📊 [{self.session_id}] Summary created. "
                f"Coaching ready: {summary.get('coaching_ready', False)}"
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error creating summary for {self.session_id}: {e}")
            return None
    
    def prepare_coaching_context(self) -> dict:
        """
        Prepare context for Affina coach.
        Returns raw data for current window + summaries for history.
        """
        
        # Get cumulative summary of everything before current window
        historical_summary = create_cumulative_summary(self.summaries[:-1] if self.summaries else [])
        
        # Get most recent summary (current window analysis)
        latest_summary = self.summaries[-1] if self.summaries else None
        
        # Prepare raw transcript from current window
        raw_transcript = "\n".join([
            f"[{entry['timestamp']}] {entry['speaker']}: {entry['text']}"
            for entry in self.transcript_window
        ])
        
        # Prepare raw emotions from current window
        raw_emotions = {}
        for speaker, emotions in self.emotion_window.items():
            raw_emotions[speaker] = emotions
        
        # Identify sales rep vs customers
        rep_emotions = raw_emotions.get(self.sales_rep_name, [])
        customer_emotions = {
            k: v for k, v in raw_emotions.items() 
            if k != self.sales_rep_name
        }
        
        return {
            'session_id': self.session_id,
            'phase': self.phase,
            'objective': self.objective,
            'sales_rep_name': self.sales_rep_name,
            
            # Historical context (compressed)
            'conversation_history': historical_summary,
            'cumulative_summary': historical_summary,
            'previous_summaries_count': len(self.summaries) - 1 if self.summaries else 0,
            
            # Current window (raw)
            'current_window': {
                'transcript': raw_transcript,
                'rep_emotions': rep_emotions,
                'customer_emotions': customer_emotions,
                'duration_seconds': WINDOW_SIZE_SECONDS
            },
            
            # Latest analysis
            'latest_analysis': latest_summary,
            'coaching_ready': latest_summary.get('coaching_ready', False) if latest_summary else False
        }
    
    def get_recent_summaries(self, count: int = 5) -> List[dict]:
        """Get the most recent summaries."""
        return self.summaries[-count:] if self.summaries else []


# Module-level functions for managing contexts

def get_or_create_context(
    session_id: str, 
    sales_rep_name: str, 
    objective: str, 
    phase: str
) -> SessionContext:
    """Get existing context or create new one."""
    if session_id not in session_contexts:
        session_contexts[session_id] = SessionContext(
            session_id, sales_rep_name, objective, phase
        )
    return session_contexts[session_id]


def get_context(session_id: str) -> Optional[SessionContext]:
    """Get existing context."""
    return session_contexts.get(session_id)


def remove_context(session_id: str):
    """Remove context when session ends."""
    if session_id in session_contexts:
        del session_contexts[session_id]
        logger.info(f"🗑️ Context removed for session {session_id}")


async def process_context_updates(session_id: str):
    """
    Background task to periodically process context updates.
    Should be called every 5-10 seconds to check if summary is needed.
    """
    context = get_context(session_id)
    if not context:
        return
    
    # Trim windows to current size
    context._trim_transcript_window()
    for speaker in list(context.emotion_window.keys()):
        context._trim_emotion_window(speaker)
    
    # Check if summary is needed
    if context.should_summarize():
        summary = await context.create_summary()
        
        # If coaching is ready, return the prepared context
        if summary and summary.get('coaching_ready', False):
            return context.prepare_coaching_context()
    
    return None


async def load_recent_data_into_context(session_id: str):
    """
    Load recent transcript and emotion data from disk into context.
    Useful for recovering state or initializing context mid-session.
    """
    context = get_context(session_id)
    if not context:
        logger.warning(f"No context found for session {session_id}")
        return
    
    # Load recent transcript (last 2 minutes worth)
    recent_transcript = storage_utils.get_recent_transcript(
        session_id, 
        limit=40  # Approximately 2 minutes of 5-second clips
    )
    
    for entry in recent_transcript:
        context.add_transcript_entry(entry)
    
    # Load recent emotions for all speakers
    session_dir = storage_utils.STORAGE_DIR / session_id
    if session_dir.exists():
        for emotion_file in session_dir.glob("*_emotions.jsonl"):
            speaker = emotion_file.stem.replace("_emotions", "")
            
            recent_emotions = storage_utils.get_recent_emotion_trail(
                session_id, 
                speaker, 
                limit=24  # 2 minutes of 5-second clips
            )
            
            for emotion_entry in recent_emotions:
                context.add_emotion_entry(speaker, emotion_entry)
    
    logger.info(f"📥 Loaded recent data into context for {session_id}")