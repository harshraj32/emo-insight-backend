import os
import json
import re
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Use the enhanced coaching prompt from document 4
AFFINA_PROMPT = """
You are **Affina**, a sharp, emotionally intelligent sales buddy and real-time conversation analyst.  
You've studied every leading book, framework, and insight on **sales psychology, human behavior, emotional intelligence, and persuasive communication**.  
You instinctively understand tone, pacing, trust, curiosity, and subtle shifts in engagement.  
You listen like a human, think like a strategist, and speak like a teammate who's been on hundreds of successful sales calls.  

Your mission:  
Help the **sales rep** understand what's really happening between them and the customer — emotionally and conversationally — and guide them toward achieving their **sales objective**.  
You never give robotic or generic advice. You observe, interpret, and respond like a real coach sitting beside the rep in the call — grounded, calm, insightful.

---

### 🧩 INPUTS
You will receive:
- The **stage** of the call (Pleasantries, Pitch, Q&A, or Closing)
- The **sales objective**
- **Historical summary** of conversation so far (compressed)
- The **current 2-minute window** (raw transcript + emotions)
- **Latest analysis** from the conversation analyzer
- The **customer's emotion summary**
- The **sales rep's emotion summary**

---

### 🎯 OUTPUT STYLE
When giving advice, always respond with **one cohesive, human-sounding paragraph** (1–3 sentences max).  
Your advice must naturally blend emotional understanding with practical direction.

Structure your thinking as follows:

1️⃣ **Emotion Insight (start with a short, vivid observation)**  
   Describe what the customer feels and why, using natural language — not analytical labels.  
   Examples:  
   - "He's processing quietly."  
   - "He's poking holes, not doubting you."  
   - "He's mentally scrolling emails."  
   - "He's holding back, waiting to see if you'll overpitch."  

2️⃣ **Contextual Coaching (follow up with deep, situation-aware guidance)**  
   Analyze how the sales rep's approach and the customer's reaction connect to the rep's **objective** and the **stage** of the meeting.  
   - If the rep's delivery is strong but rushed → suggest slowing to let interest mature.  
   - If the rep is doing fine but the customer is disengaged → show how to re-engage attention smoothly.  
   - If the customer is probing → help the rep acknowledge and pivot confidently.  
   - If the rep is unclear → suggest one crisp communication fix that restores clarity.  
   - If both are aligned → guide the rep to transition toward the next micro-goal or call stage.  

   For scenarios not explicitly mentioned, infer the most human and effective response possible — draw from your deep understanding of **conversation psychology, rapport dynamics, and sales communication theory**.  
   You can creatively adapt tone and strategy to fit the flow.  

---

### 🧭 HOW YOU THINK
1. Interpret emotion signals and transcript meaning holistically.  
   Don't just classify — understand *why* they feel that way and *how it affects progress*.  
2. Adjust your guidance based on the **meeting stage**:  
   - **Pleasantries** → comfort, warmth, rapport.  
   - **Pitch** → clarity, engagement, pacing, story.  
   - **Q&A** → listening, validation, confidence.  
   - **Closing** → conviction, commitment, ease.  
3. Always tie advice back to the **sales objective** — help the rep move one clear step closer to it.  
4. Sound human and emotionally precise — never preach, repeat, or fill space.  

---

### 💬 STYLE RULES
- Tone: modern, grounded, supportive — confident but never robotic.  — zero fluff.
- **MAXIMUM 2 sentences. Period.** More than that = failure.
- No filler words like "it seems", "perhaps", "you might want to consider"
- No clichés like "be positive" or "speak warmly." Say *what to do*, not *how to feel*.  
- Avoid overexplaining — concise, sharp insights that make the rep *instantly understand the moment.*  
- Handle any unfamiliar or unexpected conversational scenario gracefully — you've seen every kind of human behavior; improvise intelligently.  
- Keep it realistic, conversational, and emotionally attuned.  
- You are a trusted ally, not a critic.  

---

### 📦 OUTPUT FORMAT
Always return JSON:
{
  "feedback": "Your concise, actionable advice here (1-3 sentences)"
}
"""


def coach_feedback_with_context(coaching_context: dict) -> dict:
    """
    Provide coaching using structured context from context manager.
    
    Args:
        coaching_context: Dict from context_manager.prepare_coaching_context()
            Contains:
            - conversation_history: Compressed summary of prior conversation
            - current_window: Raw transcript and emotions from last 2 mins
            - latest_analysis: Analysis from summarizer
            - phase, objective, sales_rep_name
    
    Returns:
        Dict with feedback
    """
    
    ctx = coaching_context
    current = ctx.get('current_window', {})
    analysis = ctx.get('latest_analysis', {})
    
    # Check for valid data
    has_data = bool(current.get('transcript', '').strip())
    
    if not has_data:
        return {
            "feedback": "Waiting for conversation data. Ensure participants are speaking and cameras/mics are on."
        }
    
    # Format customer emotions for prompt
    customer_emotions_summary = []
    for name, emotions in current.get('customer_emotions', {}).items():
        if emotions:
            latest = emotions[-1]
            audio = latest.get('audio_emotions', [])
            video = latest.get('video_emotions', [])
            
            emo_str = f"{name}: "
            if audio:
                emo_str += f"Voice({audio[0]['name']} {audio[0]['score']:.2f}) "
            if video:
                emo_str += f"Face({video[0]['name']} {video[0]['score']:.2f})"
            customer_emotions_summary.append(emo_str)
    
    # Format rep emotions
    rep_emotions_summary = ""
    rep_emotions = current.get('rep_emotions', [])
    if rep_emotions:
        latest = rep_emotions[-1]
        audio = latest.get('audio_emotions', [])
        video = latest.get('video_emotions', [])
        
        if audio:
            rep_emotions_summary += f"Voice({audio[0]['name']} {audio[0]['score']:.2f}) "
        if video:
            rep_emotions_summary += f"Face({video[0]['name']} {video[0]['score']:.2f})"
    
    user_prompt = f"""
=== MEETING CONTEXT ===
Sales Rep: {ctx.get('sales_rep_name', 'Rep')}
Objective: {ctx.get('objective', 'Close the deal')}
Stage: {ctx.get('phase', 'Pitch')}

=== CONVERSATION HISTORY (Summary) ===
{ctx.get('conversation_history', '[Meeting just started]')}

=== CURRENT WINDOW (Last 2 Minutes - RAW DATA) ===
Transcript:
{current.get('transcript', '[No conversation]')}

Sales Rep Emotions:
{rep_emotions_summary or '[No data]'}

Customer Emotions:
{chr(10).join(customer_emotions_summary) if customer_emotions_summary else '[No data]'}

=== ANALYZER ASSESSMENT ===
Summary: {analysis.get('summary', 'Processing')}
Dynamics: {analysis.get('dynamics', 'Processing')}
Stage Assessment: {analysis.get('stage_assessment', ctx.get('phase', 'Unknown'))}
Coaching Reason: {analysis.get('coaching_reason', 'Real-time monitoring')}

---

Based on this context, provide actionable coaching for {ctx.get('sales_rep_name', 'the rep')}.
Focus on what they should do RIGHT NOW based on customer reactions.
Output ONLY JSON with "feedback" field.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": AFFINA_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=200,  # Keep advice concise
        )

        content = response.choices[0].message.content.strip()
        
        # Clean markdown
        if "```json" in content:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
        elif "```" in content:
            content = re.sub(r'```.*?```', '', content, flags=re.DOTALL).strip()
        
        try:
            result = json.loads(content)
            if "feedback" not in result:
                result["feedback"] = "Keep engaging naturally."
            return result
        except json.JSONDecodeError as e:
            logger.error(f"[COACH] JSON parse failed: {e}")
            logger.debug(f"[COACH] Raw: {content[:500]}")
            
            # Try to extract feedback from raw text
            if "feedback" in content:
                match = re.search(r'"feedback"\s*:\s*"([^"]*)"', content)
                if match:
                    return {"feedback": match.group(1)}
            
            return {"feedback": "Keep the conversation flowing naturally."}
            
    except Exception as e:
        logger.error(f"[COACH] Error: {e}")
        return {
            "feedback": "Focus on your meeting objective.",
            "error": str(e)
        }


# Legacy function for backward compatibility
def coach_feedback(context: dict, transcript: str) -> dict:
    """
    Legacy coaching function. 
    Converts old format to new context format.
    """
    logger.warning("[COACH] Using legacy coach_feedback. Consider migrating to coach_feedback_with_context.")
    
    # Convert to new format
    coaching_context = {
        'phase': context.get('phase', 'Pitch'),
        'objective': context.get('objective', 'Close the deal'),
        'sales_rep_name': context.get('sales_rep_name', 'Rep'),
        'conversation_history': '[Legacy mode - no history]',
        'current_window': {
            'transcript': transcript,
            'rep_emotions': [],
            'customer_emotions': {}
        },
        'latest_analysis': {
            'summary': 'Legacy analysis',
            'coaching_reason': 'Real-time monitoring'
        }
    }
    
    return coach_feedback_with_context(coaching_context)