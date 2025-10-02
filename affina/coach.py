import os
import json
import re
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

AFFINA_PROMPT = """
You are Affina, a sales coach providing real-time guidance to sales reps during live calls.

Analyze:
1. SALES REP's emotions and delivery
2. CUSTOMER's reactions and engagement  
3. Conversation flow and emotion trails over time

CRITICAL: Coach ONLY the sales rep on what to do based on customer reactions.

### Emotion Interpretation
When emotions are close (within 0.05-0.07):
- Don't force a single label
- Describe as blended: "slight boredom with calm focus"
- Acknowledge nuance

### Stage-Specific Coaching

**Pleasantries**: Voice warmth, energy, rapport building
**Pitch**: Focus on customer reactions (face + voice)
  - Confused → "Slow down and clarify"
  - Bored → "Add story or ask question"
  - Interested → "Keep this energy"
**Q&A**: Check if answers land well
**Closing**: Guide on when to push vs back off

### Output Format
{
  "stage": "Pleasantries|Pitch|Q&A|Closing",
  "sales_rep_state": "brief emotional state description",
  "customer_state": "brief reaction description",
  "emotion_trend": "stable|improving|declining",
  "recommendation": "1-2 actionable sentences"
}

Output ONLY valid JSON, no markdown.
"""

def coach_feedback(context: dict, transcript: str) -> dict:
    """
    Provide coaching with emotion trails and conversation history.
    """
    
    sales_rep_name = context.get("sales_rep_name", "Rep")
    rep_summary = context.get("rep_summary")
    customer_summaries = context.get("customer_summaries", {})
    emotion_trails = context.get("emotion_trails", {})

    # Check for valid data
    has_valid_data = False
    
    if rep_summary and rep_summary.get("data"):
        rep_data = rep_summary["data"]
        if rep_data.get("audio", {}).get("top_emotions") or rep_data.get("video", {}).get("top_emotions"):
            has_valid_data = True
    
    for speaker, summary in customer_summaries.items():
        if summary.get("audio", {}).get("top_emotions") or summary.get("video", {}).get("top_emotions"):
            has_valid_data = True
            break
    
    if not has_valid_data:
        return {
            "stage": context.get("phase", "Pleasantries").title(),
            "sales_rep_state": "No data",
            "customer_state": "No data",
            "emotion_trend": "unknown",
            "recommendation": "Waiting for emotion data. Ensure cameras/mics are on."
        }

    # Format emotion trails for context
    trails_summary = {}
    for speaker, trail in emotion_trails.items():
        if trail:
            trails_summary[speaker] = f"Last {len(trail)} states tracked"

    user_prompt = f"""
Meeting Context:
- Phase: {context.get("phase")}
- Objective: {context.get("objective")}
- Sales Rep: {sales_rep_name}
- Emotions to Monitor: {', '.join(context.get("emotions", []))}
SALES REP ({rep_summary['speaker'] if rep_summary else 'Unknown'}):
Current State: {json.dumps(rep_summary.get('data', {}) if rep_summary else {}, indent=2)}
Emotion Trail: {trails_summary.get(rep_summary['speaker'] if rep_summary else '', 'No history')}

CUSTOMERS:
{json.dumps({k: {"current": v, "trail": trails_summary.get(k, 'No history')} for k, v in customer_summaries.items()}, indent=2)}

Recent Conversation (last 20 exchanges):
{transcript}

Provide coaching for {sales_rep_name} based on customer reactions and emotion trends.
Output ONLY JSON, no markdown.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": AFFINA_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
            max_tokens=350,
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
            if "recommendation" not in result:
                result["recommendation"] = "Keep engaging naturally."
            return result
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON parse failed: {e}")
            print(f"[DEBUG] Raw: {content[:500]}")
            
            if "recommendation" in content:
                match = re.search(r'"recommendation"\s*:\s*"([^"]*)"', content)
                if match:
                    return {
                        "recommendation": match.group(1),
                        "stage": context.get("phase", "Unknown"),
                        "sales_rep_state": "Processing",
                        "customer_state": "Processing",
                        "emotion_trend": "unknown"
                    }
            
            return {
                "recommendation": "Keep the conversation flowing.",
                "stage": context.get("phase", "Unknown"),
                "sales_rep_state": "Processing",
                "customer_state": "Processing",
                "emotion_trend": "unknown"
            }
            
    except Exception as e:
        print(f"[DEBUG] Coach error: {e}")
        return {
            "recommendation": "Focus on your meeting objective.",
            "stage": context.get("phase", "Unknown"),
            "sales_rep_state": "Error",
            "customer_state": "Error",
            "emotion_trend": "unknown",
            "error": str(e)
        }