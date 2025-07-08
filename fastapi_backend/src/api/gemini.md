# Google Gemini API Integration (Backend)

This backend supports integration with Google Gemini LLM API to generate answers if not found in `answers.txt`.

## How it works

- The backend loads sample Q&A from `answers.txt` for fast, offline answers.
- If a user question cannot be reasonably matched to a known answer, the system sends the question to Google Gemini (LLM) and returns its response as fallback.

## Required Environment Variables

To use a real Gemini API endpoint:

- `GEMINI_API_URL`: Base URL of your Gemini API endpoint (e.g. `https://api.gemini.com/v1/chat`)
- `GEMINI_API_KEY`: API key for Gemini authentication (`Bearer` token)

If these are not configured, the backend will return a simulated Gemini response for demonstration and development.

## Example Usage

**REST endpoint:**
- `POST /answer` or `POST /chat` with `{"question": "<your-question>"}` body.
- Returns: `{"answer": "<matched or generated answer>", "from_gemini": true|false}`

## Fallback Logic

- The backend first searches `answers.txt` with substring and fuzzy matching.
- If none found, it queries Gemini and returns the LLM-generated answer.

Please ensure your keys are protected and not included in your source code or repositories.
