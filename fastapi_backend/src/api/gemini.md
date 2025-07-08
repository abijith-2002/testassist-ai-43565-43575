# Google Gemini API Integration (Backend)

This backend supports integration with Google Gemini LLM API to generate answers if not found in `answers.txt`.

## How it works

- The backend loads sample Q&A from `answers.txt` for fast, offline answers.
- If a user question cannot be reasonably matched to a known answer, the system sends the question to Google Gemini (LLM) using the actual Gemini API endpoint and returns its response as fallback.

## Required Environment Variables

To use the real Gemini API endpoint, set the following variables in your `.env` file at the project root or in the deployment environment:

- `GEMINI_API_URL`: Base URL of your Gemini API endpoint (example: `https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent`)
- `GEMINI_API_KEY`: API key for Gemini authentication (`Bearer` token, e.g., `AIza...`)

**DO NOT** hardcode these in code; always set as environment variables or in a properly secured `.env` file.

If these are not configured, the backend will return a simulated Gemini response for demonstration and development.

### `.env` Example

```
GEMINI_API_URL=https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent
GEMINI_API_KEY=YOUR_ACTUAL_GEMINI_API_KEY_HERE
```

## Example Usage

**REST endpoint:**
- `POST /answer` or `POST /chat` with `{"question": "<your-question>"}` body.
- Returns: `{"answer": "<matched or generated answer>", "from_gemini": true|false}`

## Fallback Logic

- The backend first searches `answers.txt` with substring and fuzzy matching.
- If none found, it queries Gemini and returns the LLM-generated answer.

Please ensure your GEMINI_API_KEY and GEMINI_API_URL are protected and never committed to source code or public repositories.
