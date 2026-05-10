import os
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "Missing GEMINI_API_KEY or GOOGLE_API_KEY in .env file."
    )


SYSTEM_PROMPT = """
You are CareerCounsellorAI, an India-focused engineering career counselling agent.

Your target users:
- Indian Class 12 students
- Students preparing for JEE Main, JEE Advanced, BITSAT, VITEEE, SRMJEEE, COMEDK, MHT-CET, KCET, WBJEE, JAC Delhi, IPU, AKTU, and other engineering entrances
- Parents of engineering aspirants

Your job:
Give practical, honest, India-specific engineering career counselling.

You must help with:
1. JEE score, percentile, rank, and college possibilities
2. IIT, NIT, IIIT, GFTI, state colleges, and private college comparisons
3. Branch selection: CSE, AI/ML, ECE, EE, Mechanical, Civil, Chemical, Biotechnology, etc.
4. College vs branch tradeoffs
5. JoSAA, CSAB, state counselling, and private counselling strategy
6. Backup options if JEE result is not strong
7. Career outcomes, placements, internships, skills, and higher studies

Very important rules:
- Do not guarantee admission.
- Do not invent cutoffs, fees, placements, rankings, or counselling dates.
- For current facts, always use web_search or jee_counselling_search before answering.
- Use web search for:
  - college comparison
  - cutoff
  - closing rank
  - opening rank
  - placement
  - fee
  - counselling date
  - seat matrix
  - NIRF
  - branch comparison
  - JEE score/rank/percentile prediction
- Prefer official sources when available.
- If results are not enough, clearly say that the data is insufficient.
- Cutoffs change every year, so always mention that predictions are based on previous trends.
- Use categories like Ambitious, Moderate, Safe, and Backup.

When giving college prediction, ask for missing details if needed:
- JEE Main percentile/rank
- Category: General, EWS, OBC-NCL, SC, ST, PwD
- Home state
- Gender, if relevant
- Preferred branch
- Budget
- Location preference
- Other exams given

Answer style:
- Keep every final answer short and crisp.
- Maximum 50 words.
- Use 2-4 short bullets only when needed.
- Ask only one follow-up question if required.
- No long tables unless explicitly requested.
- No lengthy explanations.
"""


class ChatRequest(BaseModel):
    question: str = Field(..., description="Question asked by student or parent")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID to continue previous conversation",
    )


class ChatResponse(BaseModel):
    session_id: str
    answer: str


class ResetRequest(BaseModel):
    session_id: str


def format_search_results(query: str, results: List[dict]) -> str:
    if not results:
        return f"No search results found for query: {query}"

    output = [f"Search query: {query}\n"]

    for i, result in enumerate(results, start=1):
        title = result.get("title", "No title")
        href = result.get("href") or result.get("url") or result.get("link", "No URL")
        body = result.get("body") or result.get("snippet", "")

        output.append(
            f"{i}. {title}\n"
            f"URL: {href}\n"
            f"Snippet: {body}\n"
        )

    return "\n".join(output)


@tool("web_search")
def web_search(query: str, max_results: int = 3) -> str:
    """
    Search the full public web using DuckDuckGo.

    Use this for current information, latest college data, fees, placements,
    exam dates, counselling updates, branch comparisons, rankings, and news.
    This is not restricted to a fixed list of websites.
    """

    try:
        max_results = max(1, min(int(max_results), 3))

        with DDGS() as ddgs:
            results = list(
                ddgs.text(
                    query,
                    region="in-en",
                    safesearch="moderate",
                    max_results=max_results,
                )
            )

        return format_search_results(query, results)

    except Exception as e:
        return f"Web search failed for query '{query}'. Error: {str(e)}"


@tool("jee_counselling_search")
def jee_counselling_search(student_query: str, max_results_per_query: int = 3) -> str:
    """
    Search the full web specifically for Indian engineering counselling.

    Use this when the user asks about JEE score, percentile, rank, college prediction,
    JoSAA, CSAB, NIT, IIIT, IIT, GFTI, branch choice, previous year cutoffs,
    opening rank, closing rank, fees, placements, or counselling strategy.
    """

    queries = [
        f"{student_query} JoSAA CSAB cutoff closing rank latest",
        f"{student_query} JEE Main college prediction previous year cutoff",
        f"{student_query} engineering college fees placements official",
    ]

    all_outputs = []

    try:
        max_results_per_query = max(1, min(int(max_results_per_query), 3))

        with DDGS() as ddgs:
            for query in queries:
                results = list(
                    ddgs.text(
                        query,
                        region="in-en",
                        safesearch="moderate",
                        max_results=max_results_per_query,
                    )
                )

                all_outputs.append(format_search_results(query, results))

        return "\n\n" + "=" * 80 + "\n\n".join(all_outputs)

    except Exception as e:
        return f"JEE counselling search failed. Error: {str(e)}"


def build_agent():
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        api_key=GEMINI_API_KEY,
        temperature=0.2,
    )

    tools = [
        web_search,
        jee_counselling_search,
    ]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent

def extract_text_from_content(content: Any) -> str:
    """
    Converts Gemini/LangChain message content into a plain string.

    Handles:
    - normal string output
    - list of content blocks like [{"type": "text", "text": "..."}]
    - objects with .text or .content
    """

    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)

            elif isinstance(block, dict):
                if "text" in block:
                    text_parts.append(str(block["text"]))
                elif block.get("type") == "text" and "content" in block:
                    text_parts.append(str(block["content"]))
                elif isinstance(block.get("content"), str):
                    text_parts.append(block["content"])

            else:
                block_text = getattr(block, "text", None)
                if isinstance(block_text, str):
                    text_parts.append(block_text)

                block_content = getattr(block, "content", None)
                if isinstance(block_content, str):
                    text_parts.append(block_content)

        return "\n".join(part for part in text_parts if part).strip()

    text_attr = getattr(content, "text", None)
    if isinstance(text_attr, str):
        return text_attr.strip()

    content_attr = getattr(content, "content", None)
    if isinstance(content_attr, str):
        return content_attr.strip()

    return str(content).strip()



app = FastAPI(
    title="CareerCounsellorAI API",
    description="FastAPI backend for Indian Class 12 engineering career counselling agent using Gemini API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = build_agent()

# In-memory chat history.
# For production, replace this with Redis/Postgres/Supabase.
chat_sessions: Dict[str, List[Dict[str, str]]] = {}


@app.get("/")
def root():
    return {
        "message": "CareerCounsellorAI API is running with Gemini",
        "docs": "/docs",
        "chat_endpoint": "/chat",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": GEMINI_MODEL,
        "provider": "gemini",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = request.session_id or str(uuid.uuid4())

    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    chat_history = chat_sessions[session_id]

    messages = chat_history + [
        {
            "role": "user",
            "content": question,
        }
    ]

    try:
        result = await run_in_threadpool(
            agent.invoke,
            {
                "messages": messages
            },
        )

        final_message = result["messages"][-1]
        answer = extract_text_from_content(final_message.content)

        if not answer:
            answer = "I could not generate a text response. Please try again."

        # Hard safety check: never return more than RESPONSE_WORD_LIMIT words.
        

        chat_sessions[session_id].append(
            {
                "role": "user",
                "content": question,
            }
        )

        chat_sessions[session_id].append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        return ChatResponse(
            session_id=session_id,
            answer=answer,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent failed: {str(e)}",
        )


@app.post("/reset")
def reset_session(request: ResetRequest):
    if request.session_id in chat_sessions:
        del chat_sessions[request.session_id]

    return {
        "status": "reset",
        "session_id": request.session_id,
    }


@app.get("/sessions")
def list_sessions():
    return {
        "total_sessions": len(chat_sessions),
        "session_ids": list(chat_sessions.keys()),
    }