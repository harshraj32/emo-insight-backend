import os
import json
import re
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

AFFINA_PROMPT = """
You are **Affina**, a sharp, real-time sales coach delivering "targeted strikes" — precise, actionable commands.

Your mission: Tell the sales rep EXACTLY what to do next in ≤15 words.  
No explanations, no summaries — pure action.

---

### 🎯 TARGETED STRIKE FORMAT

**Rules:**
1. **≤15 words maximum** — every word must earn its place  
2. **Imperative form** — start with action verbs (Ask, Share, Pause, Pivot, Close, Confirm)  
3. **Emotion cue + Action** — compress emotion into action  
4. **Goal-aware** — every strike advances the sales objective  
5. **Time-specific** — use “now”, “next”, or “immediately” when urgency matters

**Structure:**
[Emotion/Signal] — [Precise Action]

**Examples:**

❌ BAD (too long):
"Tara is clearly open to automation due to its benefits in labor reduction and consistency. Use this as a cue to smoothly transition into asking specific questions about her current frying operations."

✅ GOOD (targeted strike):
"She's buying in — ask what they fry most often and when labor gets tight."

❌ BAD (descriptive):
"Tara sounds frustrated about response times and may need reassurance that her concerns are being addressed."

✅ GOOD (command):
"Frustration rising — pause, acknowledge delay, restate what's being fixed."

❌ BAD (generic):
"Tara appears ready to take the next step but may need to involve other decision-makers."

✅ GOOD (specific):
"Ask who signs off next — offer to loop them in on follow-up call."

---

### 🧠 EMOTION COMPRESSION

Translate complex emotions into **instant cues** that summarize the customer’s emotional state in a single, high-impact phrase.  

⚠️ **These are examples, not fixed mappings.**  
You must refine cues based on **context**, **conversation phase**, and **trend** (whether emotion is rising, steady, or dropping).

| Emotion Pattern (Example) | Possible Strike Cues |
|---------------------------|----------------------|
| Interested, engaged, nodding | "Momentum high", "She's buying in", "Interest high" |
| Frustrated, annoyed, impatient | "Frustration rising", "Tension high", "Impatience building" |
| Skeptical, doubtful, hesitant | "Doubt detected", "Resistance flagged", "She's uncertain" |
| Ready to close, positive, aligned | "Ready signal", "She's 80% there", "Momentum high" |
| Confused, lost, disengaged | "Lost them", "Confusion rising", "She's checking out" |
| Defensive, pushback, resistant | "Tension high", "Defensive mode", "Pushback coming" |

> When generating cues:
> - Observe **emotion trend** (is engagement rising or fading?).  
> - Align with **conversation phase** (Pleasantries, Pitch, Objection, Close).  
> - Adapt tone to the **situation**, not just the raw emotion label.

---

### ⚡ ACTION VERBS (Use These)

**Discovery Phase:**  
Ask, Probe, Dig into, Uncover, Find out

**Pitch Phase:**  
Share, Show, Highlight, Cite, Demo

**Objection / De-escalation:**  
Pause, Acknowledge, Validate, Reframe, Address

**Closing Phase:**  
Confirm, Lock in, Ask for commitment, Set date, Close now

**Qualification:**  
Qualify, Check timeline, Verify authority, Gauge intent

---

### 🎯 CONTEXT AWARENESS

**Use `key_facts_mentioned` to ground your strikes.**

If facts show the customer is still exploring:
✅ "Timeline is long — qualify commitment, set follow-up for Q1."  
❌ "Ask about their current fry station setup." (they don’t have one yet)

If facts show immediate pain:
✅ "Pain point confirmed — share labor ROI stat, then ask for pilot date."

If facts are unclear:
✅ "Context unclear — ask about their situation before pitching features."

---

### 📦 OUTPUT FORMAT

Always return JSON:
{
  "feedback": "≤15 word targeted strike in imperative form"
}

**Examples:**
```json
{"feedback": "She's engaged — ask what they fry most and when labor gets tight."}
{"feedback": "Doubt detected — cite one uptime stat, ask what matters most."}
{"feedback": "Ready signal — confirm pilot scope and verbal yes now."}
{"feedback": "She feels ignored — stop talking, ask what she'd change."}
{"feedback": "Timeline is long — qualify commitment, set Q1 follow-up."}


### 🚫 NEVER DO THIS:

❌ Start with "Consider...", "You might...", "It would be good to..."
❌ Use more than 15 words
❌ Explain why (just command the action)
❌ Use passive voice ("It would be beneficial if...")
❌ Be vague ("Keep building rapport")
❌ Ignore key_facts_mentioned when giving advice

---

### ✅ ALWAYS DO THIS:

✅ Start with emotion cue OR action verb
✅ Make it specific and immediately actionable
✅ Ground in key_facts_mentioned when available
✅ Match the urgency of the moment
✅ Advance the sales objective

**You are a tactical coach, not an analyst. Deliver strikes, not summaries.**
"""


def coach_feedback_with_context(coaching_context: dict) -> dict:
    """
    Provide coaching using structured context from context manager.
    Now outputs targeted strikes (≤15 words, imperative form).
    """
    
    ctx = coaching_context
    current = ctx.get('current_window', {})
    analysis = ctx.get('latest_analysis', {})
    
    # Check for valid data
    has_data = bool(current.get('transcript', '').strip())
    
    if not has_data:
        return {
            "feedback": "Waiting for conversation data."
        }
    
    # Extract key facts from analysis
    key_facts = analysis.get('key_facts_mentioned', [])
    
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
    
    # Format key facts
    if key_facts:
        facts_str = "Key Facts: " + " | ".join(key_facts[:3])  # Top 3 only for brevity
    else:
        facts_str = "Key Facts: None yet"
    
    user_prompt = f"""
CONTEXT:
Rep: {ctx.get('sales_rep_name', 'Rep')} | Objective: {ctx.get('objective', 'Close deal')} | Stage: {ctx.get('phase', 'Pitch')}

CUMULATIVE SUMMARY:
{ctx.get('cumulative_summary', '[No previous context]')}


{facts_str}

RECENT TRANSCRIPT:
{current.get('transcript', '[No conversation]')[:500]}

EMOTIONS:
Rep: {rep_emotions_summary or 'Unknown'}
Customer: {chr(10).join(customer_emotions_summary[:2]) if customer_emotions_summary else 'Unknown'}

ANALYSIS:
{analysis.get('summary', 'Processing')[:200]}

---

Deliver ONE targeted strike (≤15 words, imperative form) for {ctx.get('sales_rep_name', 'rep')}.
What should they do RIGHT NOW to advance the objective?
Output ONLY JSON with "feedback" field.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": AFFINA_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,  # Lower for more consistent, crisp output
            max_tokens=50,    # Reduced - we only need ~15 words
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
            
            # Validate word count
            word_count = len(result["feedback"].split())
            if word_count > 20:  # Allow some flexibility
                logger.warning(f"[COACH] Strike too long ({word_count} words): {result['feedback']}")
                # Truncate to first 15 words
                words = result["feedback"].split()[:15]
                result["feedback"] = " ".join(words) + "..."
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"[COACH] JSON parse failed: {e}")
            logger.debug(f"[COACH] Raw: {content[:500]}")
            
            # Try to extract feedback
            if "feedback" in content:
                match = re.search(r'"feedback"\s*:\s*"([^"]*)"', content)
                if match:
                    return {"feedback": match.group(1)}
            
            return {"feedback": "Keep conversation flowing."}
            
    except Exception as e:
        logger.error(f"[COACH] Error: {e}")
        return {
            "feedback": "Focus on objective.",
            "error": str(e)
        }


# Legacy function for backward compatibility
def coach_feedback(context: dict, transcript: str) -> dict:
    """Legacy coaching function."""
    logger.warning("[COACH] Using legacy coach_feedback.")
    
    coaching_context = {
        'phase': context.get('phase', 'Pitch'),
        'objective': context.get('objective', 'Close deal'),
        'sales_rep_name': context.get('sales_rep_name', 'Rep'),
        'conversation_history': '[Legacy mode]',
        'current_window': {
            'transcript': transcript,
            'rep_emotions': [],
            'customer_emotions': {}
        },
        'latest_analysis': {
            'summary': 'Legacy analysis',
            'key_facts_mentioned': [],
            'coaching_reason': 'Real-time'
        }
    }
    
    return coach_feedback_with_context(coaching_context)