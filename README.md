# Project Repository

This is the initial README file for the project.

## Chat API Integration Requirements

To query the FastAPI backend's chat endpoint from the frontend (or any client), use the following:

### Endpoint

- **POST** `/chat` (or `/answer`)

### Payload Format

Send a JSON request body with this exact structure:

```json
{
  "question": "Your query text here"
}
```

- The field is named `question` (not `query` or `message`). It must be a string.

#### Example CURL:

```sh
curl -X POST https://your-backend-url/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a test case?"}'
```

### Response

Returns a JSON object:

```json
{
  "answer": "A test case is ...",
  "from_gemini": true
}
```

- If the request uses the wrong field name (such as `query` instead of `question`), the backend will return a 422 error due to validation.

### Error Handling

- HTTP 422 indicates that the request body did not match the expected schema.
- HTTP 400 is returned for Gemini API misconfiguration or live error.

_NOTE: The frontend code must ensure it uses `question` as the payload field when making requests to /chat._
