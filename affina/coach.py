import os
import json
import re
from openai import OpenAI

# Init client - Fixed env variable name
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- Affina System Prompt ----------------
AFFINA_PROMPT = """
You are Affina, a motivated, supportive Sales Buddy who coaches sales reps 
in real time during Zoom or Google Meet calls. You analyze:

1. The sales rep's transcript and emotions (voice + face).
2. Each customer/attendee's transcript and emotions (voice + face).
3. The flow of the call: Pleasantries → Pitch → Q&A → Closing.

---

### Stage-Specific Emotion Weighting

- **Pleasantries (Opening Small Talk)**  
  - Prioritize **voice tone** (warmth, excitement, hesitancy).  
  - Face emotions have lower weight, only use if strongly expressive.  

- **Pitch Phase (Rep Talking, Customer Listening)**  
  - Prioritize **customer face reactions** (confusion, boredom, trust).  
  - Use customer voice as supporting evidence.  
  - Rep delivery tone matters (energy, clarity).  

- **Q&A Phase (Customer Asking Questions)**  
  - Treat **voice and face equally** (tone + expression both matter).  
  - Focus on whether the rep's answers land or not.  

- **Closing Phase (End of Call)**  
  - Prioritize **voice** (confidence, warmth, clarity).  
  - Face is secondary (if they look disengaged, call it out).  

---

### Call Flow & Feedback Rules

1. Pleasantries  
   - Rep excited + customer receptive → "You started this off great…"  
   - Rep flat/confused → "Energy's low, bring more warmth so they open up."  

2. Pitch Phase  
   - Customer confused → "They looked lost – slow it down and clarify."  
   - Customer bored → "They're drifting – add a story or engage them."  
   - Customer concentrating → "They're locked in – guide step by step."  
   - Customer doubtful → "They're skeptical – back it with proof."  
   - Customer trusting → "They're vibing – lean into rapport."  

3. Q&A Phase  
   - Customer satisfied → "That answer landed – good job."  
   - Customer doubtful → "They didn't fully buy it – reinforce with proof."  
   - Customer confused → "They're still unclear – simplify your response."  

4. Closing Phase  
   - Customer positive → "They're interested – push for next steps."  
   - Customer mixed → "Not fully sold yet – recap value before closing."  
   - Customer disengaged → "They tuned out – tighten your close, ask perspective."  

---

### Output Format
Always output in JSON with:
- "stage" → Pleasantries | Pitch | Q&A | Closing  
- "speaker" → "Rep" or "Customer(s)"  
- "transcript_snippet" → latest text said (if available)  
- "dominant_channel" → "voice", "face", or "balanced"  
- "top_emotion" → strongest emotion detected (after weighting)  
- "recommendation" → 1–2 sentence actionable coaching advice  

IMPORTANT: Output ONLY valid JSON, no markdown code blocks, no extra text.
"""

# ---------------- Function ----------------
def coach_feedback(context: dict, transcript_line: str) -> dict:
    """
    Calls ChatGPT with the Hume multi-speaker summaries + transcript snippet
    and returns Affina's structured JSON coaching feedback.

    context keys:
      - phase: str
      - objective: str
      - emotions: list[str] (emotions to track)
      - summaries: dict { "rep": {...}, "customer1": {...}, ... }
    """

    # Check if we have any valid emotion data
    has_valid_data = False
    for speaker, summary in context.get("summaries", {}).items():
        audio_emotions = summary.get("audio", {}).get("top_emotions")
        video_emotions = summary.get("video", {}).get("top_emotions")
        if audio_emotions or video_emotions:
            has_valid_data = True
            break
    
    # If no valid data, return a default response
    if not has_valid_data:
        return {
            "stage": context.get("phase", "Pleasantries").title(),
            "speaker": "System",
            "transcript_snippet": transcript_line[:100] if transcript_line else "No transcript",
            "dominant_channel": "none",
            "top_emotion": "No data",
            "recommendation": "Waiting for emotion data. Make sure the bot is in the meeting and participants have their cameras/mics on."
        }

    user_prompt = f"""
    Here is the latest Hume analysis (multi-speaker) and transcript snippet.

    Meeting context:
    - Phase: {context.get("phase")}
    - Objective: {context.get("objective")}
    - Selected Emotions: {context.get("emotions")}

    Multi-speaker summaries:
    {json.dumps(context.get("summaries", {}), indent=2)}

    Transcript snippet:
    "{transcript_line}"

    Provide coaching feedback in JSON only, using the format defined above.
    Output ONLY the JSON object, no markdown code blocks, no extra text.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": AFFINA_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
            max_tokens=300,
        )

        content = response.choices[0].message.content.strip()
        
        # Clean up the response - remove markdown code blocks if present
        if "```json" in content:
            # Extract JSON from markdown code block
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
        elif "```" in content:
            # Remove any other code blocks
            content = re.sub(r'```.*?```', '', content, flags=re.DOTALL).strip()
        
        # Try to parse the JSON
        try:
            result = json.loads(content)
            # Ensure we have the required fields
            if "recommendation" not in result:
                result["recommendation"] = "Keep engaging with your audience."
            return result
        except json.JSONDecodeError as e:
            print(f"[DEBUG] Failed to parse coach JSON: {e}")
            print(f"[DEBUG] Raw content: {content[:500]}")
            
            # Try to extract just the recommendation if JSON parsing fails
            if "recommendation" in content:
                match = re.search(r'"recommendation"\s*:\s*"([^"]*)"', content)
                if match:
                    return {
                        "recommendation": match.group(1),
                        "stage": context.get("phase", "Unknown"),
                        "speaker": "Unknown",
                        "transcript_snippet": transcript_line[:100] if transcript_line else "",
                        "dominant_channel": "unknown",
                        "top_emotion": "Processing"
                    }
            
            # Fallback response
            return {
                "recommendation": "Keep the conversation flowing naturally.",
                "stage": context.get("phase", "Unknown"),
                "speaker": "Rep",
                "transcript_snippet": transcript_line[:100] if transcript_line else "",
                "dominant_channel": "unknown",
                "top_emotion": "Processing"
            }
            
    except Exception as e:
        print(f"[DEBUG] Coach error: {e}")
        return {
            "recommendation": "Stay focused on your meeting objective.",
            "error": str(e)
        }