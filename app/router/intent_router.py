import json
import re
from openai import OpenAI
from app.config import config
from app.router.models import RouteDecision, RouteType

client = OpenAI(
    base_url=config.OPENROUTER_BASE_URL,
    api_key=config.OPENROUTER_API_KEY,
)

SYSTEM_ROUTER_PROMPT = """You are an Intelligent Operations Router for Nectar Facilities Platform.
Analyze the user request and select the exact recommended route.

Available Routes:
1. RAG: General definitions, HVAC procedures, troubleshooting docs. (e.g. "What is an AHU?")
2. MCP: Real-time sensor data, current asset status, live temp. (e.g. "What is Chiller-01's current temperature?")
3. RAG_MCP_REASONING: Diagnostics requiring live telemetry AND documentation. (e.g. "Why did Chiller-01 fail?")
4. MCP_ACTION: Executing operations/creations (e.g. "Create a maintenance request for AHU-02.")
5. DATA_AGENT: Analytics/summaries (e.g. "Summarize today's energy usage.")
6. GENERAL_LLM: Greetings or out-of-scope banter.

CRITICAL: You MUST respond with ONLY a raw JSON object. No markdown formatting, no code blocks (```json ... ```), no preamble text, no explanations. The JSON must start with { and end with } with no surrounding text. Do not include any reasoning outside the JSON.
"""

def _extract_json(text: str | None) -> dict | None:
    if not text:
        return None
    text = text.strip()

    # Remove markdown code fences ```json ... ``` or ``` ...
    text = re.sub(r"```json\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"```\s*", "", text, flags=re.DOTALL)

    # Find the first { and last } to extract JSON block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Try parsing the whole text as JSON (after stripping)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    return None

def route_request(user_query: str) -> RouteDecision:
    try:
        response = client.chat.completions.create(
            model=config.ROUTER_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_ROUTER_PROMPT},
                {"role": "user", "content": user_query}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        result_content = response.choices[0].message.content
        if not result_content:
            raise ValueError("LLM returned empty content")

        data = _extract_json(result_content)

        if data is None:
            raise ValueError("Failed to extract valid JSON from LLM response")

        return RouteDecision(
            user_query=user_query,
            route=RouteType(data.get("route", "GENERAL_LLM")),
            confidence=float(data.get("confidence", 0.9)),
            reasoning=data.get("reasoning", "Routed via LLM classification"),
            extracted_entities=data.get("extracted_entities", {})
        )
    except Exception as e:
        return RouteDecision(
            user_query=user_query,
            route=RouteType.GENERAL_LLM,
            confidence=0.5,
            reasoning=f"Fallback triggered due to error: {str(e)}",
            extracted_entities={}
        )