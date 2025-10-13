"""
LLM-based conversation summarizer for Affina context management.
Analyzes rolling windows and determines when to provide coaching advice.
"""

import os
import json
import re
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SUMMARIZER_PROMPT = """
You are a conversation analyst for Affina, a real-time sales coaching system.

Your role: Analyze conversation segments and emotion data to create structured summaries 
that help the coach provide timely, relevant advice.

### Your Tasks:

1. **Summarize the 2-minute window**:
   - ONLY summarize what was ACTUALLY said in clear, understandable language
   - If transcript is garbled/incomplete, say "Transcript quality is poor" - DO NOT invent interpretations
   - Key points discussed
   - Emotional shifts for each participant
   - Communication dynamics (pacing, engagement, rapport)

2. **Assess coaching readiness**:
   - Is there enough context to give meaningful advice?
   - Are there clear signals the sales rep needs guidance?
   - Should we wait for more information in the next 5-second update?

3. **Identify critical moments**:
   - Customer confusion, disengagement, or concern
   - Sales rep struggling or missing cues
   - Transition points between stages
   - Opportunities to advance the sale

### CRITICAL: Be Accurate, Not Interpretive
- DO NOT call garbled text "technical jargon" - call it "unclear/garbled transcript"
- DO NOT invent meaning from incomplete sentences
- DO NOT assume intent when transcript is poor quality
- If you can't understand what was said, acknowledge that explicitly

### Coaching Readiness Criteria (BE STRICT):
- **YES** if ALL of these are true:
  1. At least 6-8 complete, clear transcript lines (not garbled)
  2. At least 3-4 back-and-forth exchanges between rep and customer
  3. Clear customer question or reaction about the product
  4. Rep has attempted to pitch or explain something substantial
  5. Transcript quality is good (not full of errors or "unable to transcribe")
  6. Clear coaching opportunity exists

- **NO** if ANY of these are true:
  - Just greetings or pleasantries ("Hey", "How's it going", "What's up")
  - Garbled, incomplete, or low-confidence transcript
  - Less than 6 clear transcript lines
  - Less than 3 back-and-forth exchanges
  - No clear pitch or substantive product discussion yet
  - Unclear context - need to wait for next update cycle
  - Nothing specific to coach on yet

### Special Rules:
- If transcript has multiple "unable to transcribe" errors, automatically say NO
- If transcript is mostly fragments or single words, say NO
- If both parties just exchanged greetings, say NO - wait for actual discussion
- Better to wait 5-10 more seconds than give premature advice

### Output Format:
{
  "summary": "Brief factual summary - acknowledge if transcript is poor",
  "key_emotions": {
    "sales_rep": "dominant emotion with trend",
    "customers": {"Name": "emotion + engagement level"}
  },
  "dynamics": "1 sentence on actual conversation quality and flow",
  "coaching_ready": true/false,
  "coaching_reason": "specific reason - be honest about data quality",
  "stage_assessment": "Pleasantries|Pitch|Q&A|Closing"
}

Output ONLY valid JSON. Be conservative - when in doubt, say NO and wait for better data.
"""


def summarize_window(
    transcript_window: list,
    emotion_window: dict,
    sales_rep_name: str,
    objective: str,
    current_stage: str
) -> dict:
    """
    Summarize a 2-minute conversation window.
    
    Args:
        transcript_window: List of transcript entries from last 2 mins
        emotion_window: Dict of {speaker: [emotion entries]} from last 2 mins
        sales_rep_name: Name of the sales rep
        objective: Meeting objective
        current_stage: Current meeting stage
    
    Returns:
        Dict with summary and coaching readiness assessment
    """
    
    # Format transcript
    transcript_text = "\n".join([
        f"[{entry['timestamp']}] {entry['speaker']}: {entry['text']}"
        for entry in transcript_window
    ])
    
    if not transcript_text.strip():
        transcript_text = "[No conversation in this window]"
    
    # Format emotions
    emotions_text = ""
    for speaker, emotions in emotion_window.items():
        if emotions:
            latest = emotions[-1]
            audio_emo = latest.get('audio_emotions', [])
            video_emo = latest.get('video_emotions', [])
            
            emotions_text += f"\n{speaker}:\n"
            if audio_emo:
                try:
                    # Build emotion string separately to avoid f-string issues
                    emo_strs = [f"{e.get('name', 'Unknown')}({e.get('score', 0):.2f})" for e in audio_emo[:3]]
                    emotions_text += f"  Voice: {', '.join(emo_strs)}\n"
                except Exception as e:
                    emotions_text += f"  Voice: [Error formatting: {str(e)}]\n"
            if video_emo:
                try:
                    # Build emotion string separately to avoid f-string issues
                    emo_strs = [f"{e.get('name', 'Unknown')}({e.get('score', 0):.2f})" for e in video_emo[:3]]
                    emotions_text += f"  Face: {', '.join(emo_strs)}\n"
                except Exception as e:
                    emotions_text += f"  Face: [Error formatting: {str(e)}]\n"
    
    if not emotions_text.strip():
        emotions_text = "[No emotion data in this window]"
    
    user_prompt = f"""
Meeting Context:
- Sales Rep: {sales_rep_name}
- Objective: {objective}
- Current Stage: {current_stage}

=== TRANSCRIPT (Last 2 Minutes) ===
{transcript_text}

=== EMOTIONS (Last 2 Minutes) ===
{emotions_text}

Analyze this window and determine if the coach should provide advice now.
Output ONLY JSON.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Using mini for cost efficiency
            messages=[
                {"role": "system", "content": SUMMARIZER_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,  # Lower temp for consistent analysis
            max_tokens=400,
        )

        content = response.choices[0].message.content.strip()
        
        # Clean markdown if present
        if "```json" in content:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
        elif "```" in content:
            content = re.sub(r'```.*?```', '', content, flags=re.DOTALL).strip()
        
        result = json.loads(content)
        
        # Validate required fields
        required_fields = ["summary", "key_emotions", "dynamics", "coaching_ready", "coaching_reason"]
        for field in required_fields:
            if field not in result:
                result[field] = "Processing" if field != "coaching_ready" else False
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"[SUMMARIZER] JSON parse error: {e}")
        print(f"[SUMMARIZER] Raw content: {content[:300]}")
        
        # Return safe fallback
        return {
            "summary": "Unable to parse analysis",
            "key_emotions": {},
            "dynamics": "Processing",
            "coaching_ready": False,
            "coaching_reason": "Analysis error",
            "stage_assessment": current_stage
        }
        
    except Exception as e:
        print(f"[SUMMARIZER] Error: {e}")
        return {
            "summary": f"Error: {str(e)}",
            "key_emotions": {},
            "dynamics": "Error",
            "coaching_ready": False,
            "coaching_reason": "System error",
            "stage_assessment": current_stage
        }


def create_cumulative_summary(previous_summaries: list) -> str:
    """
    Condense previous summaries into a brief cumulative summary.
    
    Args:
        previous_summaries: List of summary dicts from earlier in the conversation
    
    Returns:
        Concise text summary of everything that happened before
    """
    if not previous_summaries:
        return "[Meeting just started]"
    
    # Simple concatenation for now - could be LLM-enhanced later
    summary_parts = []
    
    for i, summ in enumerate(previous_summaries[-5:]):  # Last 5 windows max
        stage = summ.get('stage_assessment', 'Unknown')
        text = summ.get('summary', '')
        summary_parts.append(f"[{stage}] {text}")
    
    return " → ".join(summary_parts)