# Building the API with FastAPI

Your goal is to create `serve.py` — a web service that wraps the full RAG pipeline and exposes it over HTTP so a website can send questions and receive answers.

## The big picture first

Right now, your RAG pipeline lives entirely in `rag.py` and runs from the command line. To connect it to a website, you need an **API** — a process that listens on a network port, accepts HTTP requests with a question in them, runs the pipeline, and sends back a JSON response.

Here is what a single query will do end-to-end once this is built:

```
Browser
  │
  │  POST /query  {"question": "Who are Rose's grandchildren?"}
  ▼
serve.py  (FastAPI)
  │
  │  calls rag.answer(question, ...)
  ▼
rag.py
  ├── hybrid_search()  →  top-20 candidates from index.json
  ├── rerank()         →  reranked to top-8 via cross-encoder
  └── Groq API call   →  LLM generates answer from those 8 chunks
  │
  │  returns {"answer": "...", "sources": [...]}
  ▼
serve.py  serializes to JSON
  │
  │  HTTP 200  {"answer": "Sean, Zac, Julia, Jennifer", "sources": [...]}
  ▼
Browser
```

`serve.py` doesn't contain any RAG logic — it's purely the HTTP layer sitting in front of `rag.py`.

---

## Background: what is FastAPI?

FastAPI is a Python web framework. You define functions and decorate them with routes (`@app.get(...)`, `@app.post(...)`), and FastAPI handles turning incoming HTTP requests into Python function calls and turning return values back into JSON responses.

It's built on two libraries you won't interact with directly:
- **Starlette** — the underlying async web toolkit that handles the actual TCP/HTTP plumbing
- **Pydantic** — validates that incoming and outgoing JSON matches the shape you declared, and raises descriptive errors if not

One useful side effect: FastAPI reads your type annotations and auto-generates interactive API docs at `/docs`. You can test your endpoints in the browser without writing any frontend code.

---

## Step 1 — Install

```bash
.venv/bin/pip install "fastapi[standard]"
```

The `[standard]` extra includes **uvicorn**, which is the actual server process. FastAPI defines your routes and logic, but uvicorn is what listens on port 8000 and speaks HTTP. Think of FastAPI as the application and uvicorn as the server running it — similar to how Django/Flask apps are often run behind gunicorn.

---

## Step 2 — Create serve.py and make a minimal app

Create `serve.py`. At the top, import FastAPI and create an app instance:

```python
from fastapi import FastAPI

app = FastAPI(title="Rose's Memories")
```

This `app` object is the entry point uvicorn needs. It holds all your routes and middleware.

Now add a single route — a health check so you can verify the server is running before touching any RAG logic:

```python
@app.get("/health")
def health():
    return {"status": "ok"}
```

The decorator `@app.get("/health")` registers this function as the handler for `GET /health` requests. FastAPI automatically serializes the returned dict to JSON.

Run it:

```bash
.venv/bin/uvicorn serve:app --reload
```

`serve:app` means "the object named `app` in `serve.py`". `--reload` watches your files and restarts the server on save — only use this during development.

Open `http://localhost:8000/health` in your browser. You should see `{"status":"ok"}`. Also try `http://localhost:8000/docs` — that's the auto-generated UI.

---

## Step 3 — Understand the startup problem

Your `rag.py` does three expensive things when it runs:
1. Reads `index.json` from disk (4.7 MB, 429 chunks)
2. Loads the sentence-transformers embedding model into memory (~80 MB)
3. Loads the BGE cross-encoder reranker model into memory (~120 MB)

Together this takes 2–5 seconds. If you called `rag.answer()` directly inside a route handler, every single HTTP request would pay that cost. A user asking a question would wait 5 seconds before the pipeline even starts — that's unacceptable.

The solution is to load all of that **once when the server starts** and hold it in memory for the lifetime of the process. Every incoming request then reuses the already-loaded resources. FastAPI provides a mechanism for this called a **lifespan event**.

---

## Step 4 — Add a lifespan handler

A lifespan function is an async context manager — a function that uses `yield` to split into two phases. Everything before the `yield` runs on startup; everything after runs when the server shuts down.

Add this to `serve.py`, above the `app = FastAPI(...)` line:

```python
from contextlib import asynccontextmanager

state = {}

@asynccontextmanager
async def lifespan(app):
    # startup — runs once before the server accepts any requests
    print("Loading index and model...")
    # you'll fill this in next
    yield
    # shutdown — runs when the server is killed (Ctrl+C)
    state.clear()
```

`state` is a plain module-level dict. It acts as in-memory storage shared across all route handlers for the life of the server process. This is the simplest approach for a single-process server — no need for a cache layer or global variables.

Then pass `lifespan` to FastAPI so it knows to call it:

```python
app = FastAPI(title="Rose's Memories", lifespan=lifespan)
```

Restart the server and confirm it still starts without errors.

---

## Step 5 — Load the index and model at startup

Import the tools you need from `rag.py` and your other dependencies:

```python
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from rag import EMBED_MODEL, GROQ_API_KEY, RERANKER_MODEL, load_index, answer
```

You're importing the constants (`EMBED_MODEL`, `GROQ_API_KEY`, `RERANKER_MODEL`) from `rag.py` so that `serve.py` doesn't need its own copies — a single source of truth.

Now fill in the startup section of your lifespan function:

```python
state["records"], state["index_embs"] = load_index()
state["model"] = SentenceTransformer(EMBED_MODEL)
state["reranker"] = CrossEncoder(RERANKER_MODEL)
state["groq"] = Groq(api_key=GROQ_API_KEY)
print(f"  {len(state['records'])} chunks ready")
```

`load_index()` returns two things: the list of chunk records (text, source, context) and a numpy array of their embeddings. Both go into `state` so the route handler can pass them to `rag.answer()` without reloading anything.

Restart the server. You should see the loading output in your terminal before it starts accepting requests.

---

## Step 6 — Update the health endpoint

Now that you know how many chunks are loaded, surface that in the health check. This is useful for confirming the index loaded correctly and for monitoring in production:

```python
@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(state["records"])}
```

Hit `/health` again and verify the chunk count matches the 429 that `index.py` reported.

---

## Step 7 — Define your request and response shapes

Before writing the query endpoint, define what JSON goes in and what comes out. FastAPI uses **Pydantic models** for this.

```python
from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
```

Why bother? Three reasons:

1. **Automatic validation.** If a client sends `{"question": 123}` (a number instead of a string) or omits the field entirely, FastAPI returns a descriptive 422 error automatically. You write zero validation code.

2. **Documentation.** FastAPI uses these models to generate the `/docs` UI. Anyone using your API can see exactly what to send and what they'll get back.

3. **Contract enforcement.** `response_model=QueryResponse` on the route tells FastAPI to validate your *output* too — if `rag.answer()` ever returns a dict missing the `sources` field, FastAPI will catch it before it reaches the client.

`QueryRequest` maps to the JSON body the client sends. `QueryResponse` maps to what you return. The `sources` field will contain the filenames of the bio files that contributed chunks to the answer — useful for the frontend to show attribution.

---

## Step 8 — Add the query endpoint

This is the core of the service. It receives a question, passes it through the full RAG pipeline, and returns the answer.

```python
from fastapi import HTTPException

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = answer(
        req.question,
        state["records"],
        state["index_embs"],
        state["model"],
        state["groq"],
        state["reranker"],
    )
    return QueryResponse(**result)
```

Notice what this function is *not* doing: no embedding, no search, no LLM call — all of that is inside `rag.answer()`. This route is purely the HTTP adapter that translates an HTTP request into a `rag.answer()` call and translates the result back into an HTTP response.

`HTTPException` short-circuits the handler and returns an error response immediately. Status 400 means "bad request" — the client sent something invalid. The `detail` string appears in the response body.

`QueryResponse(**result)` unpacks the dict that `rag.answer()` returns (`{"answer": "...", "sources": [...]}`) into the Pydantic model, which FastAPI then serializes to JSON.

Restart the server and test the full pipeline:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What were Roses grandparents like?"}'
```

You should see an answer drawn from the bio files.

---

## Step 9 — Add CORS

CORS (Cross-Origin Resource Sharing) is a browser security policy. When a web page at `https://yoursite.com` tries to call `http://localhost:8000/query`, the browser first sends a **preflight request** asking the server "do you allow requests from this origin?". If the server doesn't respond correctly, the browser blocks the actual request — even if the server is running fine.

This only affects browser-based clients. `curl` and server-to-server calls are not subject to CORS.

Add this middleware **before** your route definitions:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

`allow_origins=["*"]` permits requests from any origin. That's fine for local development and a private personal project. If this were public-facing, you'd replace `"*"` with your actual domain (e.g. `["https://yoursite.com"]`) to prevent other sites from using your API.

The middleware must be registered before routes because FastAPI processes middleware in registration order — a request hits middleware first, then routes.

---

## Step 10 — Add a main block

Right now you have to invoke uvicorn manually from the terminal. Add a main block so you can also run the server with `python serve.py`:

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)
```

`host="0.0.0.0"` tells uvicorn to accept connections on all network interfaces, not just loopback (`127.0.0.1`). This matters when deploying — if your server is on a remote machine or inside a container, `localhost` would only be reachable from inside that machine itself.

Note that you pass `"serve:app"` as a string rather than the `app` object directly. This is because uvicorn needs to be able to import and reload the module by name — passing the object directly disables `--reload`.

---

## What your finished serve.py should contain, in order

1. Imports (FastAPI, middleware, Pydantic, Groq, SentenceTransformer, CrossEncoder, rag.py)
2. `state = {}`
3. `lifespan()` — loads index, embedding model, reranker, Groq client
4. `app = FastAPI(lifespan=lifespan)`
5. CORS middleware
6. `QueryRequest` and `QueryResponse` Pydantic models
7. `GET /health` — returns status and chunk count
8. `POST /query` — runs the full RAG pipeline
9. `if __name__ == "__main__"` block

---

## Common mistakes

**`KeyError` on `state`** — a request arrived before startup finished, so `state` is still empty. Rare in practice. Means you hit the endpoint immediately after starting the server before lifespan completed.

**`response_model` missing** — FastAPI won't validate the response shape. The route will still work but you lose the guarantee that the output matches `QueryResponse`, and the `/docs` UI won't show the response schema.

**`answer()` signature mismatch** — `rag.answer()` now takes a `reranker` argument. Make sure you're passing `state["reranker"]` to it — check the current signature in `rag.py` before writing the call.

**`load_dotenv()` not called before `GROQ_API_KEY` is read** — `rag.py` already handles this at module import time. Since `serve.py` imports from `rag.py`, `load_dotenv()` will have run by the time `GROQ_API_KEY` is used.

**CORS not working** — make sure `add_middleware` comes before route definitions, and that you actually restarted the server after adding it. `--reload` doesn't always catch middleware changes cleanly.
