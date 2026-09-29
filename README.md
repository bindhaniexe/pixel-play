# Pixel Play

A small local studio that turns model pigment probabilities into paintings.
Type a prompt, and the app asks a model what color belongs in every cell of a
coarse grid, then paints those weighted pigments with contour-following strokes,
an underpainting, and impasto relief lighting. Each painting joins an in-memory
carousel and disappears when you reload.

## Quick start

Requires Python 3.9+ and a modern browser (Chrome, Edge, Firefox, or Safari).
No packages, no build step.

Give the helper a model key through the environment — the browser never sees it:

```sh
# Jev (TypeSafe System One) — typed questions, real probability answers
export JEV_API_KEY=...            # Windows: $env:JEV_API_KEY = "..."

# or any OpenAI-compatible chat model
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...        # optional, defaults to https://api.openai.com/v1
export OPENAI_MODEL=...           # optional, defaults to gpt-4o-mini

python server.py
```

Open **http://127.0.0.1:8791**, open Studio settings, pick a representation and
grid size, and enter a prompt. Use `python server.py --port 8795` if the default
port is busy. Stop the helper with Ctrl+C.

### Environment variables

| Variable | Meaning |
| --- | --- |
| `JEV_API_KEY` | TypeSafe API key; selects the Jev backend |
| `JEV_MODEL`, `JEV_ENDPOINT` | Override the Jev model or endpoint |
| `OPENAI_API_KEY` / `AI_API_KEY` | Key for any OpenAI-compatible backend |
| `OPENAI_BASE_URL` / `AI_BASE_URL` | Alternate OpenAI-compatible base URL |
| `OPENAI_MODEL` / `AI_MODEL` | Chat model name |
| `PAINT_PROVIDER` | `auto` (default), `jev`, or `chat` |
| `PORT` | Default port for the helper |

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

- Model keys live only in the helper's environment. They are attached to
  outbound model requests inside the server process and are never stored,
  logged, or returned to the browser.
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
