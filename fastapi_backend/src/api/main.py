import os
import logging

# --- load .env variables before anything else ---
try:
    from dotenv import load_dotenv
    load_dotenv()  # Loads variables from .env at project root, if found
except ImportError:
    pass  # dotenv is in requirements.txt, so this should succeed; if not, continue as normal

from fastapi import FastAPI, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import aiohttp
import asyncio

# --- Set up logging for debug/error tracing ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chat-api")

# --- Application Metadata with OpenAPI Tags ---
app = FastAPI(
    title="TestAssist AI Chat Backend",
    description=(
        "Backend API for chatbot answering application testing questions with Retrieval-Augmented Generation (RAG), "
        "leveraging answers.txt knowledge base and Google Gemini LLM."
    ),
    version="2.1.0",
    openapi_tags=[
        {"name": "health", "description": "Health check endpoint"},
        {"name": "chat", "description": "Chat endpoints (RAG/LLM and chat history)"},
        {"name": "llm", "description": "Gemini and context info"},
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANSWERS_FILE = os.path.join(os.path.dirname(__file__), "answers.txt")

# --- Models ---

class RoleContent(BaseModel):
    role: str = Field(..., description="Message role, either 'user' or 'assistant'.")
    content: str = Field(..., description="Message content text.")

# PUBLIC_INTERFACE
class ChatHistoryQuery(BaseModel):
    """
    Accepts a full chat history between user and assistant; used for RAG context.
    """
    history: list[RoleContent] = Field(..., description="The full chat history: array of {role, content} objects.")

# PUBLIC_INTERFACE
class RAGChatAnswer(BaseModel):
    """
    Returns the RAG/LLM-style answer to a user chat query.
    """
    answer: str = Field(..., description="Answer in markdown, contextually generated from chat history and retrieval.")
    from_gemini: bool = Field(default=True, description="True if answer is generated using Gemini or LLM, else False.")
    retrieval_refs: list[str] = Field(default=[], description="List of referenced/found matched answers for traceability.")

# --- Exception for 400 error ---
class FastAPIHTTP400(Exception):
    def __init__(self, detail):
        self.detail = detail

@app.exception_handler(FastAPIHTTP400)
async def fastapi_http_400_handler(request: Request, exc: FastAPIHTTP400):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=exc.detail,
    )

# --- Gemini config check ---
async def require_gemini_config():
    """
    Raises a 400 error if GEMINI_API_KEY or GEMINI_API_URL is missing.
    Returns (api_url, api_key) if both present.
    """
    missing = []
    api_url = os.getenv("GEMINI_API_URL")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_url:
        missing.append("GEMINI_API_URL")
    if not api_key:
        missing.append("GEMINI_API_KEY")
    if missing:
        detail = {
            "error": "Gemini API configuration error (RAG required live LLM)",
            "message": "Required environment variable(s) missing.",
            "missing": missing
        }
        raise FastAPIHTTP400(detail)
    return api_url, api_key

# -- Answers.txt utility: Load and parse all Q&A pairs into a list --

def get_all_qa_pairs():
    pairs = []
    try:
        with open(ANSWERS_FILE, "r", encoding="utf-8") as f:
            lines = [l.rstrip() for l in f if l.strip()]
        q, a = None, None
        for line in lines:
            if line.startswith("Q:"):
                if q and a:
                    pairs.append({"question": q, "answer": a})
                q = line[2:].strip()
                a = None
            elif line.startswith("A:"):
                a = line[2:].strip()
            elif a is not None:
                a += " " + line.strip()
        if q and a:
            pairs.append({"question": q, "answer": a})
    except Exception as ex:
        logger.error(f"Could not load answers.txt: {ex}")
        pairs = []
    return pairs

def load_answers_txt_raw():
    try:
        with open(ANSWERS_FILE, "r", encoding="utf-8") as f:
            qa_content = f.read().strip()
        if qa_content:
            return (
                "The following Q&A pairs from 'answers.txt' are authoritative for the current application. "
                "If you find a matching question, answer exactly as in answers.txt. Otherwise, use your best knowledge.\n"
                "----- BEGIN ANSWERS.TXT -----\n"
                f"{qa_content}\n"
                "----- END ANSWERS.TXT -----\n"
            )
    except Exception:
        pass
    return ""

# --- RAG: Retrieve best Q&A matches from answers.txt based on most recent user message and conversation context ---
def retrieve_relevant_qa(history, max_matches=3):
    """
    Simple retrieval: For the most recent user message, find the most similar/matching Qs in answers.txt.
    Uses lowercasing and substring/fuzzy containment.
    Returns up to max_matches pairs, most relevant first.
    """
    qa_pairs = get_all_qa_pairs()
    if not history or not qa_pairs:
        return []
    user_messages = [msg.content.strip() for msg in history if msg.role == "user" and msg.content.strip()]
    # Most recent user message is usually the query to answer
    last_user = user_messages[-1] if user_messages else ""
    if not last_user:
        return []

    import difflib
    lower_qs = [(qap["question"].lower(), qap["answer"], qap["question"]) for qap in qa_pairs]
    query = last_user.lower()
    matches = []

    # Substring, then difflib fuzzy match
    # 1. Strict substring
    for q_lower, a_text, orig_q in lower_qs:
        if query in q_lower or q_lower in query:
            matches.append({"question": orig_q, "answer": a_text, "score": 3})

    # 2. Fuzzy partial ratio (difflib)
    scored = []
    for q_lower, a_text, orig_q in lower_qs:
        score = difflib.SequenceMatcher(None, q_lower, query).ratio()
        scored.append((score, orig_q, a_text))
    scored.sort(reverse=True)
    seen = set(m["question"] for m in matches)
    for score, orig_q, a_text in scored:
        if orig_q not in seen and score > 0.55:
            matches.append({"question": orig_q, "answer": a_text, "score": score})
            seen.add(orig_q)

    # 3. Limit number of matches (prioritizing substring first)
    matches = matches[:max_matches]
    return matches

# --- Gemini LLM integration (with fallback mock if not available) ---
# PUBLIC_INTERFACE
async def query_gemini_or_mock(prompt: str, chat_context: str, use_real: bool, api_url=None, api_key=None):
    """
    Query Gemini with chat_context + prompt, or fall back to local mock if Gemini API is not available.
    Returns generated answer (markdown string).
    """
    full_prompt = (
        f"{chat_context}\n"
        "You are a QA/Test Assistant. Continue this chat, answering the next user message as helpfully as possible.\n"
        f"User and assistant chat so far:\n"
        "{CHAT_TRANSCRIPT}\n"
        f"Now answer as concisely and accurately as possible to the user's last message. Format as markdown.\n"
        f"----- END CONTEXT -----\n"
        f"User's last question: {prompt}\n"
    )

    # Replace placeholder with real chat in context
    chat_transcript = chat_context
    if "{CHAT_TRANSCRIPT}" in full_prompt:
        chat_transcript = "" # Not needed, supplied above
        full_prompt = full_prompt.replace("{CHAT_TRANSCRIPT}", "")

    # Call real Gemini if configured, else use mock
    if use_real and api_url and api_key:
        from yarl import URL
        json_payload = {
            "contents": [
                {
                    "parts": [{"text": full_prompt}]
                }
            ]
        }
        headers = {"Content-Type": "application/json"}
        parsed_url = URL(api_url).with_query(key=api_key)
        api_full_url = str(parsed_url)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_full_url, headers=headers, json=json_payload) as resp:
                    text_resp = await resp.text()
                    if resp.status == 200:
                        try:
                            data = await resp.json()
                            answer = (
                                data.get("candidates", [{}])[0]
                                    .get("content", {})
                                    .get("parts", [{}])[0]
                                    .get("text", "")
                            )
                            if not answer or not answer.strip():
                                logger.warning("Gemini LLM returned empty or None answer, fallback to raw text.")
                                return text_resp
                            return answer
                        except Exception as ex:
                            logger.error(f"Gemini parse error: {ex}, raw: {text_resp}")
                            return text_resp
                    else:
                        logger.error(f"Gemini API error {resp.status}: {text_resp}")
                        raise FastAPIHTTP400({
                            "error": "Gemini API error",
                            "message": f"Gemini responded with HTTP {resp.status}",
                            "body": text_resp
                        })
        except Exception as e:
            logger.error(f"Gemini call/network error: {e}")
            # Fallback to mock in a pinch, avoid total failure
            return f"[Backend error calling Gemini: {e}. Using mock answer.]\n"

    # Mock logic: blend available context as if LLM
    mock_start = ""
    if chat_context:
        mock_start = "Relevant facts from known Q&A:\n"
        lines = []
        for line in chat_context.strip().split("\n"):
            if line.startswith("Q:") or line.startswith("A:"):
                lines.append(line)
        mock_start += "\n".join(lines[-12:]) + "\n\n"
    return f"{mock_start}I'm sorry, I cannot provide a specific answer from the knowledge base for your question. (LLM mock)."

# --- API Endpoints ---

# PUBLIC_INTERFACE
@app.post(
    "/chat",
    response_model=RAGChatAnswer,
    tags=["chat"],
    summary="RAG: Conversational Chat endpoint (Retrieval + Gemini LLM, chat history aware)",
    description=(
        "RAG API: Accepts full conversation history and answers using a combination of retrieval from answers.txt "
        "and Gemini LLM context. Returns one markdown answer."
    ),
    responses={
        400: {"description": "Gemini API misconfiguration or live Gemini error."}
    }
)
async def rag_chat_endpoint(payload: ChatHistoryQuery):
    """
    Chatbot RAG endpoint for full conversational context.
    - Accepts POST payload: {history: [{role: ..., content: ...}, ...]}
    - Does retrieval on the latest user message and prior history.
    - Retrieves closest Q&A pairs from answers.txt, incorporates them into context.
    - Calls Gemini (real or mock) with combined context and returns the answer.
    - Returns { "answer": ..., "from_gemini": true, "retrieval_refs": [<matched Qs>] }
    """
    logger.info(f"Received RAG /chat call: history={len(payload.history)} messages")
    use_gemini = False
    api_url = api_key = None
    try:
        api_url, api_key = await require_gemini_config()
        use_gemini = True
    except Exception as e:
        logger.warning(f"Gemini not configured (ok for mock/local): {e}")

    # Step 1: Retrieve relevant Q&A from answers.txt
    relevant_pairs = retrieve_relevant_qa(payload.history, max_matches=3)
    retrieval_context = ""
    retrieval_refs = []
    if relevant_pairs:
        for pair in relevant_pairs:
            retrieval_context += f"Q: {pair['question']}\nA: {pair['answer']}\n"
            retrieval_refs.append(pair['question'])

    # Step 2: Build the chat+retrieved context for the LLM
    chat_history_markdown = ""
    for entry in payload.history:
        chat_history_markdown += f"{entry.role.capitalize()}: {entry.content}\n"
    if retrieval_context:
        context_block = f"Relevant Q&A:\n{retrieval_context}\nChat transcript:\n{chat_history_markdown}"
    else:
        context_block = f"Chat transcript:\n{chat_history_markdown}"

    # Step 3: The full prompt sent to LLM/Gemini
    last_user_msg = None
    for entry in reversed(payload.history):
        if entry.role == "user":
            last_user_msg = entry.content.strip()
            break
    if not last_user_msg:
        return RAGChatAnswer(answer="Sorry, no valid question found in chat history.", from_gemini=False, retrieval_refs=[])

    # Step 4: Call Gemini (real or mock)
    answer = await query_gemini_or_mock(
        prompt=last_user_msg,
        chat_context=context_block,
        use_real=use_gemini,
        api_url=api_url,
        api_key=api_key
    )
    # Ensure returned answer is markdown and not blank
    if not answer or not str(answer).strip():
        answer = "Sorry, could not generate an answer at this time. Please try again later."
    return RAGChatAnswer(
        answer=answer,
        from_gemini=use_gemini,
        retrieval_refs=retrieval_refs
    )

# --- Other endpoints (unchanged): health, llm_info, answers Q&A debug ---

@app.get("/", tags=["health"])
def health_check():
    """Simple always-OK health check endpoint."""
    return {"message": "Healthy"}

@app.get(
    "/answers",
    tags=["chat"],
    summary="List all Q&A pairs from answers.txt (debug/info only)",
    description="Returns all Q&A pairs found in the answers.txt file as reference."
)
def list_qa_pairs():
    """
    Returns the full contents of answers.txt file as Q&A pairs, for frontend display/reference
    (not for answer matching).
    """
    return get_all_qa_pairs()

@app.get(
    "/llm_info",
    tags=["llm"],
    summary="LLM (Gemini) integration/config info",
    description="Details about Gemini API usage and required backend configuration."
)
def llm_info():
    """
    Provides details on how Gemini is used, which env variables are required, and usage policy.
    """
    return {
        "llm_used": "Google Gemini + answers.txt retrieval (RAG)",
        "env_vars": ["GEMINI_API_URL", "GEMINI_API_KEY"],
        "policy": (
            "RAG (retrieval-augmented generation): each answer is generated using retrieved Q&A from answers.txt and current chat, "
            "with Gemini as LLM for conversational/fallback logic."
        )
    }

@app.get(
    "/docs/ws",
    tags=["llm"],
    summary="WebSocket usage info",
    description="WebSocket API documentation placeholder."
)
def ws_usage():
    """ Placeholder for potential future WebSocket API. Not implemented. """
    return {
        "info": "WebSocket endpoints are not supported at this time. Use /chat (POST) for all chatbot queries."
    }
