# FastAPI RAG Chatbot Backend

A FastAPI backend that provides a Retrieval-Augmented Generation (RAG) chatbot API. It answers application testing questions using:
- A local file-based knowledge base (answers.txt)
- Google Gemini LLM for generative responses

The service exposes REST endpoints for chat, health checks, and configuration info, with permissive CORS for easy client integration. It is designed to be easily extensible for future WebSocket-based real-time chat.

## Features

- Retrieval-Augmented Generation (RAG): Combines local Q&A retrieval with Gemini LLM
- Local knowledge base: Parses Q&A pairs from `src/api/answers.txt`
- Strict Gemini key enforcement: Requires environment variables for live Gemini calls
- Simple health and LLM info endpoints
- Permissive CORS by default for rapid frontend integration
- Clean architecture and Pydantic models with OpenAPI documentation
- Extensibility: Clear placeholder for future WebSocket support

## Endpoints

Base URL: http(s)://<host>:3001

- GET `/` (Health)
  - Returns a simple health check payload: `{"message": "Healthy"}`
  - Tags: health

- POST `/chat` (Main RAG Chat)
  - Accepts a full chat history and answers using local retrieval + Gemini LLM.
  - Request Body (JSON):
    {
      "history": [
        { "role": "user", "content": "..." },
        { "role": "assistant", "content": "..." }
      ]
    }
  - Response (JSON):
    {
      "answer": "<markdown answer>",
      "from_gemini": true,
      "retrieval_refs": ["<matched-questions>"]
    }
  - Errors:
    - 400: "Gemini API key is not configured." (if Gemini env not set or runtime error)
  - Tags: chat

- GET `/answers` (Debug/Info)
  - Lists all Q&A pairs parsed from `answers.txt` for reference.
  - Tags: chat

- GET `/llm_info` (LLM Config Info)
  - Shows LLM usage and required environment variables.
  - Tags: llm

- GET `/docs/ws` (WebSocket Usage Placeholder)
  - Placeholder endpoint documenting that WebSocket support may be added later.
  - Tags: llm

OpenAPI schema is available at `/openapi.json`. Interactive docs available at `/docs` (Swagger UI) and `/redoc`.

## Knowledge Base (answers.txt)

- Location: `src/api/answers.txt`
- Format: A simple list of Q&A pairs:
  Q: <question>
  A: <answer>

- Parsing behavior:
  - The backend reads and parses Q&A pairs, then performs simple retrieval based on the latest user message:
    - Substring checks
    - Fuzzy similarity (difflib) to find close matches
  - Up to 3 top matches are passed to Gemini as contextual guidance.

## Google Gemini Integration

- The backend constructs a context prompt including:
  - Matching Q&A snippets from `answers.txt`
  - The full chat transcript
  - The user’s most recent question
- The service then calls the Gemini API to produce a final response.

Strict configuration enforcement:
- If Gemini configuration is missing, the backend returns HTTP 400 with:
  { "detail": "Gemini API key is not configured." }

Required environment variables:
- GEMINI_API_URL
  - Example: https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent
- GEMINI_API_KEY
  - Your Gemini API key

Do not hardcode these values. Use environment variables or a .env file. Never commit secrets.

## CORS

CORS is configured to:
- allow_origins = ["*"]
- allow_methods = ["*"]
- allow_headers = ["*"]
- allow_credentials = True

This enables quick integration from any frontend during development. Tighten these settings for production.

## Setup

1) Install dependencies
- Ensure Python 3.10+ (recommended)
- From the `fastapi_backend` folder:
  pip install -r requirements.txt
  OR
  ./install.sh

2) Configure environment variables
- Export variables in your environment, or create a .env file (not committed to VCS):
  GEMINI_API_URL=https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent
  GEMINI_API_KEY=YOUR_ACTUAL_GEMINI_API_KEY_HERE

3) Run the server
- From the `fastapi_backend` folder:
  uvicorn src.api.main:app --host 0.0.0.0 --port 3001
  OR
  ./run.sh

The API will be available at:
- http://localhost:3001
- Docs: http://localhost:3001/docs
- OpenAPI: http://localhost:3001/openapi.json

## Example Request

POST /chat
Content-Type: application/json

{
  "history": [
    { "role": "user", "content": "What should I test in the shopping cart?" }
  ]
}

Response:
{
  "answer": "…",
  "from_gemini": true,
  "retrieval_refs": ["What test cases should be included when testing the shopping cart module?"]
}

Note:
- If GEMINI_API_URL and GEMINI_API_KEY are not provided, the endpoint returns HTTP 400 with the message:
  { "detail": "Gemini API key is not configured." }

## Extensibility (Future WebSocket Support)

- The app includes a placeholder `GET /docs/ws` describing potential WebSocket usage.
- To add real-time chat:
  - Implement a FastAPI WebSocket route (e.g., `/ws/chat`)
  - Maintain conversation state per connection
  - Reuse existing retrieval and Gemini call utilities
  - Update OpenAPI docs and docs endpoint to note real-time usage

## Project Structure

fastapi_backend/
- requirements.txt
- run.sh
- src/
  - api/
    - main.py        # FastAPI app and endpoints
    - answers.txt    # Local Q&A knowledge base
    - gemini.md      # Gemini integration guidance

## Notes

- Security: Do not log secrets. Restrict CORS in production. Rotate API keys regularly.
- Rate Limiting: Consider adding rate limiting and authentication for production.
- Observability: Add structured logging and monitoring for Gemini errors and latency.

## License

This project is provided as-is for demonstration and integration purposes. Add your preferred license before distributing.
