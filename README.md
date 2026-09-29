# Pixel Play

A small local studio that turns model pigment probabilities into paintings.
Type a prompt, and the app asks a model what color belongs in every cell of a
coarse grid, then paints those weighted pigments with contour-following strokes,
an underpainting, and impasto relief lighting. Each painting joins an in-memory
carousel and disappears when you reload.

## Quick start

Requires Python 3.9+ and a modern browser (Chrome, Edge, Firefox, or Safari).
No packages, no build step.

Copy `.env.example` to `.env` and point it at a model — the browser never sees
anything in that file. It works with **local OpenAI-compatible servers** like
Ollama, LM Studio, or vLLM (just a URL and a model name, no key needed):

```ini
# .env — Ollama example
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3.2
OPENAI_API_KEY=
```

or with Jev — TypeSafe's model or **any third-party Jev-compatible model** —
or any hosted OpenAI-compatible model:

```ini
# .env — Jev example                  # .env — hosted example
JEV_BASE_URL=https://api.typesafe.ai/v1/systemone
                                     OPENAI_BASE_URL=https://api.openai.com/v1
JEV_MODEL=jev-latest                 OPENAI_MODEL=gpt-4o-mini
                                     OPENAI_API_KEY=sk-...
JEV_API_KEY=...
```

Both backends take the same three knobs — **key, model name, and URL** — so a
third-party Jev/System One endpoint is just another base URL and model name:

```ini
# .env — third-party Jev-compatible model
PAINT_PROVIDER=jev
JEV_BASE_URL=https://jev.example.com/v1/systemone
JEV_MODEL=my-jev-fork
JEV_API_KEY=...                      # omit if the endpoint needs no key
```

Then run the helper and open **http://127.0.0.1:8791**:

```sh
python server.py                 # Windows: python server.py
```

Open Studio settings, pick a representation and grid size, and enter a prompt.
Use `python server.py --port 8795` if the default port is busy, or
`python server.py --env D:\configs\pixel.env` to load settings from elsewhere
(`--env none` skips the file). Stop the helper with Ctrl+C.

### Settings

Everything can come from `.env` or from real environment variables, which
override the file. Keys are read by the helper only and are never returned to
the browser.

| Variable | Meaning |
| --- | --- |
| `OPENAI_BASE_URL` / `AI_BASE_URL` | Any OpenAI-compatible endpoint, e.g. `http://localhost:11434/v1` |
| `OPENAI_MODEL` / `AI_MODEL` | Model name that endpoint serves |
| `OPENAI_API_KEY` / `AI_API_KEY` | Key for that endpoint; optional for local servers |
| `JEV_BASE_URL` / `JEV_ENDPOINT` | Any Jev/System One-compatible endpoint |
| `JEV_MODEL` | Model name that endpoint serves, e.g. `jev-latest` |
| `JEV_API_KEY` | Key for that endpoint; optional for endpoints without auth |
| `PAINT_PROVIDER` | `auto` (default), `jev`, or `chat` |
| `PORT` | Default port for the helper |

With `PAINT_PROVIDER=auto`, any Jev configuration (key, URL, or model) wins;
otherwise any OpenAI-compatible configuration selects that backend.

## How it works

- `providers.py` — backends. Each one turns a prompt into a **field**: a grid of
  cells, each holding up to four weighted RGB pigment candidates. The Jev
  backend asks typed Choice questions (one per cell channel) and reads the
  returned probability distributions; the chat backend asks any
  OpenAI-compatible model for a compact palette grid. Both normalize into the
  same field shape.
- `server.py` — a loopback-only helper. It serves the studio files and exposes
  `/api/paint`, attaching the key from the environment to outbound model calls.
- `web/paint.js` — the painter, pure functions over typed arrays. Mean pigment
  becomes the underpainting, per-cell distributions supply stroke pigments,
  entropy steers relief strength, and a height map carries the impasto lighting.
  Rendering runs in `web/paint.worker.js` at 560×560 and is deterministic for a
  given prompt and settings.
- `web/app.js`, `web/index.html`, `web/style.css` — the gallery and composer.

The four representations — 16-color palette, HSL, Binary RGB, and silhouette —
choose how pigments are asked for. Jev answers carry genuine probability
distributions; chat models return a palette grid whose neighbouring cells leak
into each pigment mix so edges stay painterly.

## Security and privacy

- Model keys live only in `.env` (gitignored) or the helper's environment. They
  are attached to outbound model requests inside the server process and are
  never stored, logged, or returned to the browser.
- The helper binds to `127.0.0.1`, checks the request origin, and answers with
  `no-store` and a strict content security policy. Access logging is disabled so
  prompts are never written to disk.
- Prompts are sent only to the model endpoint you configured.

## Checks

```sh
python -B -m unittest discover -s tests -p 'test_*.py'
node --test tests/painter.test.mjs
```

Everything runs offline against synthetic probability fields and recorded model
replies; no API calls are made and no tokens are spent.
