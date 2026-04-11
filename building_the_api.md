# Building the API with FastAPI

Your goal is to create `serve.py` — a web service that exposes `rag.py` over HTTP so a website can send questions and receive answers.

---

## Background: what is FastAPI?

FastAPI is a Python web framework. You define functions and decorate them with routes (`@app.get(...)`, `@app.post(...)`), and FastAPI handles turning HTTP requests into function calls and function return values back into JSON responses.

It's built on two libraries you won't interact with directly:
- **Starlette** — the underlying async web toolkit
- **Pydantic** — validates that incoming/outgoing JSON matches the shape you declared

One nice side effect: FastAPI reads your type annotations and auto-generates interactive docs at `/docs`.

---

## Step 1 — Install

```bash
.venv/bin/pip install "fastapi[standard]"
```

The `[standard]` extra includes `uvicorn`, the server that actually listens for HTTP connections and hands them to FastAPI.

---

## Step 2 — Create serve.py and make a minimal app

Create `serve.py`. At the top, import FastAPI and create an app instance:

```python
from fastapi import FastAPI

app = FastAPI(title="Rose's Memories")
```

Now add a single route — a health check so you can verify the server is running:

```python
@app.get("/health")
def health():
    return {"status": "ok"}
```

Run it:

```bash
.venv/bin/uvicorn serve:app --reload
```

`serve:app` means "the object named `app` in `serve.py`". `--reload` restarts the server when you save the file.

Open `http://localhost:8000/health` in your browser. You should see `{"status":"ok"}`. Also try `http://localhost:8000/docs` — that's the auto-generated UI.

---

## Step 3 — Understand the startup problem

Your `rag.py` loads the sentence-transformers model and `index.json` every time `answer()` is called. That takes 2–3 seconds. You cannot do that on every HTTP request.

The solution is to load those resources **once when the server starts** and keep them in memory. FastAPI calls this a **lifespan event**.

---

## Step 4 — Add a lifespan handler

A lifespan function is an async context manager. Everything before the `yield` runs on startup; everything after runs on shutdown.

Add this to `serve.py`, above the `app = FastAPI(...)` line:

```python
from contextlib import asynccontextmanager

state = {}

@asynccontextmanager
async def lifespan(app):
    # startup
    print("Loading index and model...")
    # you'll fill this in next
    yield
    # shutdown
    state.clear()
```

Then pass it to FastAPI:

```python
app = FastAPI(title="Rose's Memories", lifespan=lifespan)
```

Restart the server and confirm it still starts without errors.

---

## Step 5 — Load the index and model at startup

Import the tools you need from `rag.py` and your other dependencies:

```python
from groq import Groq
from sentence_transformers import SentenceTransformer
from rag import EMBED_MODEL, GROQ_API_KEY, load_index, answer
```

Now fill in the startup section of your lifespan function:

```python
state["records"], state["index_embs"] = load_index()
state["model"] = SentenceTransformer(EMBED_MODEL)
state["groq"] = Groq(api_key=GROQ_API_KEY)
print(f"  {len(state['records'])} chunks ready")
```

`state` is just a plain dict that lives in memory for the life of the server process. Every route handler can read from it.

Restart the server. You should see the loading output in your terminal.

---

## Step 6 — Update the health endpoint

Now that you know how many chunks are loaded, return that too:

```python
@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(state["records"])}
```

Hit `/health` again and verify the chunk count matches what `index.py` reported.

---

## Step 7 — Define your request and response shapes

Before writing the query endpoint, define what JSON goes in and what comes out. FastAPI uses **Pydantic models** for this. It will automatically validate incoming requests and return a 422 error if the shape is wrong.

```python
from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
```

`QueryRequest` describes the JSON body the client sends. `QueryResponse` describes what you'll return.

---

## Step 8 — Add the query endpoint

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
    )
    return QueryResponse(**result)
```

A few things to notice:
- `response_model=QueryResponse` tells FastAPI to validate your return value — it will error if your dict is missing a field.
- `HTTPException` sends an error response with a specific status code. 400 means "bad request" (the client sent invalid input).
- `QueryResponse(**result)` unpacks the dict that `rag.answer()` returns into the Pydantic model.

Restart the server and test it:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What were Roses grandparents like?"}'
```

---

## Step 9 — Add CORS

CORS (Cross-Origin Resource Sharing) is a browser security rule that blocks a web page from calling an API on a different domain. Since your website will be on a different origin than this server, you need to allow it.

Add this **before** your route definitions:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

`allow_origins=["*"]` permits any origin. That's fine for local development — when you deploy, tighten it to your actual domain.

---

## Step 10 — Add a main block

Right now you have to run `uvicorn serve:app ...` from the terminal. Add a main block so you can also run it directly with `python serve.py`:

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)
```

`host="0.0.0.0"` means accept connections from any network interface, not just localhost. You'll need that when deploying.

---

## What your finished serve.py should contain, in order

1. Imports
2. `state = {}`
3. `lifespan()` — loads index, model, Groq client
4. `app = FastAPI(lifespan=lifespan)`
5. CORS middleware
6. `QueryRequest` and `QueryResponse` Pydantic models
7. `GET /health`
8. `POST /query`
9. `if __name__ == "__main__"` block

---

## Common mistakes

**`KeyError` on `state`** — a request arrived before startup finished. Rare, but means you're sending requests immediately after starting the server.

**`response_model` missing** — FastAPI won't validate the response shape. Your route will still work but you lose the guarantee that the output matches `QueryResponse`.

**`load_dotenv()` not called before `GROQ_API_KEY` is read** — `rag.py` already handles this, but if you ever move constants around, keep `load_dotenv()` at the very top.

**CORS not working** — make sure `add_middleware` comes before route definitions and that you actually restarted the server after adding it.
