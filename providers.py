"""Model backends for Pixel Play.

Each provider turns a text prompt into a *field*: a square grid of cells where
every cell holds up to four weighted RGB pigment candidates. The browser
renderer only ever consumes fields, so backends stay interchangeable.

API keys are read from environment variables and live inside this process only.
Nothing in this module logs, stores, or returns key material.
"""
from __future__ import annotations

import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
CHAT_ENDPOINT = "https://api.openai.com/v1/chat/completions"
CHAT_MODEL = "gpt-4o-mini"

REPRESENTATIONS = ("palette", "hsl", "rgb", "silhouette")
GRID_SIZES = (8, 12, 16, 24, 32)

# Upper bound on typed questions per System One call. All questions inside one
# call are evaluated in parallel by the model, so batches are kept large but
# bounded; a cell's channels always stay inside a single batch.
MAX_QUESTIONS_PER_CALL = 144
PARALLEL_CALLS = 4
HTTP_TIMEOUT = 120

# A sixteen-swatch painter's palette: deep darks, earth mid-tones, cool and
# warm lights. Swatch order is stable so question criteria never shift.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (31, 34, 51),
    (57, 64, 92),
    (107, 112, 143),
    (164, 167, 189),
    (217, 214, 205),
    (242, 237, 226),
    (140, 59, 46),
    (194, 100, 63),
    (224, 149, 86),
    (232, 192, 122),
    (124, 139, 82),
    (76, 107, 74),
    (47, 75, 63),
    (34, 54, 79),
    (92, 125, 156),
    (176, 196, 207),
)
PAPER = (243, 238, 228)
INK = (24, 22, 32)

HUE_BINS = 12          # 30-degree sectors
LEVEL_BINS = 5         # saturation and lightness steps
BIT_PLANES = 4         # binary RGB: four bit planes per channel

Hue = float
_open: Callable[..., Any] = urlopen


class ProviderError(RuntimeError):
    """The model is unreachable, or answered with something unusable."""


# ---------------------------------------------------------------- field model


def make_field(
    width: int, height: int, cells: list[list[list[float]]], meta: dict[str, Any]
) -> dict[str, Any]:
    """Assemble and validate a paint field."""
    if len(cells) != width * height:
        raise ProviderError("cell count does not match the grid size")
    for candidates in cells:
        if not candidates:
            raise ProviderError("a cell arrived without pigments")
        total = 0.0
        for pigment in candidates:
            if len(pigment) != 4:
                raise ProviderError("a pigment needs three channels and a weight")
            r, g, b, weight = pigment
            for channel in (r, g, b):
                if not math.isfinite(channel) or not 0 <= channel <= 255:
                    raise ProviderError("a pigment channel is out of range")
            if not math.isfinite(weight) or weight <= 0:
                raise ProviderError("a pigment weight must be positive")
            total += weight
        if not math.isfinite(total) or total <= 0:
            raise ProviderError("pigment weights do not sum to a positive value")
    return {"width": width, "height": height, "cells": cells, "meta": meta}


def _normalise(candidates: Sequence[Sequence[float]], limit: int = 4) -> list[list[float]]:
    """Rescale weights to sum to one and keep only the heaviest pigments."""
    ordered = sorted(
        ([float(c[0]), float(c[1]), float(c[2]), float(c[3])] for c in candidates),
        key=lambda p: -p[3],
    )[:limit]
    total = sum(p[3] for p in ordered)
    for pigment in ordered:
        pigment[3] /= total
    return ordered


def _clamp_byte(value: float) -> float:
    return max(0.0, min(255.0, value))


def hsv_to_rgb(hue: float, saturation: float, lightness: float) -> tuple[float, float, float]:
    """Convert HSV (hue in degrees) to 0-255 RGB."""
    h = (hue % 360.0) / 60.0
    c = saturation * lightness
    x = c * (1 - abs(h % 2 - 1))
    m = lightness - c
    sector = int(h) % 6
    pairs = ((c, x, 0.0), (x, c, 0.0), (0.0, c, x), (0.0, x, c), (x, 0.0, c), (c, 0.0, x))
    r, g, b = pairs[sector]
    return ((r + m) * 255.0, (g + m) * 255.0, (b + m) * 255.0)


# ----------------------------------------------------------- question plumbing


@dataclass
class Question:
    """One typed question bound to one channel of one grid cell."""

    qid: str
    cell: int
    channel: str
    payload: dict[str, Any]


@dataclass
class QuestionPlan:
    questions: list[Question] = dc_field(default_factory=list)
    batches: list[list[Question]] = dc_field(default_factory=list)


def _grid_positions(size: int) -> list[tuple[int, int]]:
    return [(col, row) for row in range(size) for col in range(size)]


def _swatch_criteria() -> dict[str, str]:
    criteria = {}
    for index, (r, g, b) in enumerate(PALETTE):
        hexes = f"#{r:02x}{g:02x}{b:02x}"
        warmth = "warm" if r > b else "cool"
        value = "dark" if (r + g + b) / 3 < 96 else "light" if (r + g + b) / 3 > 180 else "mid"
        criteria[f"s{index}"] = f"{hexes}, a {value} {warmth} pigment"
    return criteria


def _level_criteria(labels: Sequence[str], prefix: str = "l") -> dict[str, str]:
    return {f"{prefix}{i}": label for i, label in enumerate(labels)}


def build_jev_plan(prompt: str, representation: str, size: int) -> QuestionPlan:
    """Build typed questions covering every grid cell exactly once.

    Channels of the same cell are always packed into the same batch so one
    call answers them against identical context.
    """
    if representation not in REPRESENTATIONS:
        raise ProviderError(f"unknown representation: {representation}")
    if size not in GRID_SIZES:
        raise ProviderError(f"unknown grid size: {size}")

    plan = QuestionPlan()
    counter = 0

    def add(cell: int, channel: str, instructions: str, criteria: dict[str, str]) -> None:
        nonlocal counter
        qid = f"q{counter}"
        counter += 1
        plan.questions.append(
            Question(
                qid=qid,
                cell=cell,
                channel=channel,
                payload={"type": "choice", "instructions": instructions, "criteria": criteria},
            )
        )

    positions = _grid_positions(size)
    swatches = _swatch_criteria() if representation == "palette" else {}
    hue_criteria = _level_criteria(
        [f"hue around {i * 30 + 15} degrees" for i in range(HUE_BINS)], prefix="h"
    )
    sat_criteria = _level_criteria(
        ["almost grey", "muted", "moderate", "rich", "fully saturated"]
    )
    light_criteria = _level_criteria(["near black", "shadow", "middle", "bright", "near white"])
    bit_criteria = {"b0": "the bit is off", "b1": "the bit is on"}
    figure_criteria = {"subject": "part of the main subject", "ground": "background or empty space"}

    for cell, (col, row) in enumerate(positions):
        where = (
            f"grid cell column {col}, row {row} of a {size} by {size} grid "
            f"laid over the image (column 0 and row 0 are top-left)"
        )
        if representation == "palette":
            add(cell, "swatch", f"In this painting, which pigment fills {where}?", swatches)
        elif representation == "hsl":
            add(cell, "hue", f"Which hue family belongs at {where} in this painting?", hue_criteria)
            add(cell, "sat", f"How saturated is the color at {where} in this painting?", sat_criteria)
            add(cell, "light", f"How light is the color at {where} in this painting?", light_criteria)
        elif representation == "rgb":
            for channel in ("red", "green", "blue"):
                for plane in range(BIT_PLANES):
                    weight = 1 << (BIT_PLANES - 1 - plane)
                    add(
                        cell,
                        f"bit:{channel}:{plane}",
                        f"Is bit {weight} of the {channel} channel set at {where} in this painting?",
                        bit_criteria,
                    )
        else:  # silhouette
            add(cell, "figure", f"Is {where} part of the main subject of this painting?", figure_criteria)

    # Pack questions into bounded batches without splitting a cell's channels.
    batch: list[Question] = []
    batch_cells: set[int] = set()
    for question in plan.questions:
        opens_new_cell = question.cell not in batch_cells
        if batch and opens_new_cell and len(batch) >= MAX_QUESTIONS_PER_CALL:
            plan.batches.append(batch)
            batch, batch_cells = [], set()
        batch.append(question)
        batch_cells.add(question.cell)
    if batch:
        plan.batches.append(batch)
    return plan


def _probability_map(raw: Any, qid: str) -> dict[str, float]:
    """Normalise the two plausible probability shapes into {option: float}."""
    if isinstance(raw, Mapping):
        items = raw.items()
    elif isinstance(raw, list):
        items = []
        for entry in raw:
            if not isinstance(entry, Mapping):
                raise ProviderError(f"question {qid} returned a malformed probability list")
            name = entry.get("option", entry.get("name"))
            value = entry.get("probability", entry.get("p"))
            items.append((name, value))
    else:
        raise ProviderError(f"question {qid} returned no probabilities")

    out: dict[str, float] = {}
    for name, value in items:
        if not isinstance(name, str):
            raise ProviderError(f"question {qid} returned an unnamed option")
        try:
            probability = float(value)
        except (TypeError, ValueError):
            raise ProviderError(f"question {qid} returned a non-numeric probability") from None
        if not math.isfinite(probability) or probability < 0:
            raise ProviderError(f"question {qid} returned an invalid probability")
        out[name] = probability
    if not out or sum(out.values()) <= 0:
        raise ProviderError(f"question {qid} returned an empty distribution")
    return out


class JevProvider:
    """Typed-question backend: TypeSafe's Jev or any Jev-compatible endpoint."""

    name = "jev"

    def __init__(
        self,
        api_key: str = "",
        model: str = JEV_MODEL,
        endpoint: str = JEV_ENDPOINT,
        timeout: int = HTTP_TIMEOUT,
    ) -> None:
        self._key = api_key
        self._model = model
        self._endpoint = endpoint
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    @property
    def endpoint(self) -> str:
        return self._endpoint

    # -- wire ---------------------------------------------------------------

    def _call(self, state: str, batch: Sequence[Question]) -> dict[str, Any]:
        body = json.dumps(
            {
                "state": state,
                "model": self._model,
                "questions": {q.qid: q.payload for q in batch},
            }
        ).encode("utf-8")
        request = Request(
            self._endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self._key:
            request.add_header("Authorization", f"Bearer {self._key}")
        try:
            with _open(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise ProviderError(f"the model rejected the request (HTTP {error.code})") from None
        except (URLError, TimeoutError, OSError):
            raise ProviderError("the model could not be reached") from None
        except (ValueError, UnicodeError):
            raise ProviderError("the model returned unreadable data") from None
        if not isinstance(payload, Mapping) or not isinstance(payload.get("answers"), Mapping):
            raise ProviderError("the model returned no answers")
        return payload["answers"]

    def _ask(self, state: str, plan: QuestionPlan) -> dict[str, dict[str, float]]:
        answers: dict[str, dict[str, float]] = {}
        with ThreadPoolExecutor(max_workers=PARALLEL_CALLS) as pool:
            for result in pool.map(lambda b: self._call(state, b), plan.batches):
                for qid, raw in result.items():
                    if isinstance(raw, Mapping):
                        answers[qid] = _probability_map(raw.get("probabilities"), qid)
        missing = [q.qid for q in plan.questions if q.qid not in answers]
        if missing:
            raise ProviderError(f"the model skipped {len(missing)} of {len(plan.questions)} questions")
        return answers

    # -- generation ---------------------------------------------------------

    def generate(self, prompt: str, representation: str, size: int) -> dict[str, Any]:
        plan = build_jev_plan(prompt, representation, size)
        answers = self._ask(f"A painting of: {prompt}", plan)
        by_cell: dict[int, dict[str, dict[str, float]]] = {}
        for question in plan.questions:
            by_cell.setdefault(question.cell, {})[question.channel] = answers[question.qid]

        cells = [
            _cell_from_distributions(representation, by_cell.get(index, {}))
            for index in range(size * size)
        ]
        return make_field(
            size,
            size,
            cells,
            {"provider": self.name, "model": self._model, "representation": representation},
        )


def _cell_from_distributions(
    representation: str, channels: dict[str, dict[str, float]]
) -> list[list[float]]:
    """Collapse one cell's probability answers into weighted pigments."""
    if representation == "palette":
        distribution = channels.get("swatch", {})
        candidates = []
        for option, probability in distribution.items():
            if option.startswith("s") and option[1:].isdigit():
                index = int(option[1:])
                if 0 <= index < len(PALETTE):
                    r, g, b = PALETTE[index]
                    candidates.append([r, g, b, probability])
        return _normalise(candidates)

    if representation == "hsl":
        hue = channels.get("hue", {})
        sat = channels.get("sat", {"l2": 1.0})
        light = channels.get("light", {"l2": 1.0})

        def level_expectation(distribution: dict[str, float]) -> tuple[float, float]:
            """Return (expected level 0..1, spread) from labelled bins."""
            total = sum(distribution.values())
            mean = 0.0
            spread = 0.0
            for option, probability in distribution.items():
                if option.startswith("l") and option[1:].isdigit():
                    index = int(option[1:])
                    share = probability / total
                    level = index / (LEVEL_BINS - 1)
                    mean += share * level
                    spread += share * (level - 0.5) ** 2
            return mean, math.sqrt(spread)

        sat_mean, _ = level_expectation(sat)
        light_mean, _ = level_expectation(light)
        total = sum(hue.values())
        ranked = sorted(
            (
                (probability / total, (int(option[1:]) * 30 + 15))
                for option, probability in hue.items()
                if option.startswith("h") and option[1:].isdigit()
            ),
            reverse=True,
        )
        if not ranked:
            raise ProviderError("a hue answer carried no bins")
        candidates = [
            [*_c(hsv_to_rgb(degrees, 0.25 + 0.75 * sat_mean, 0.15 + 0.75 * light_mean)), weight]
            for weight, degrees in ranked[:3]
        ]
        return _normalise(candidates)

    if representation == "rgb":
        expected = []
        variance = 0.0
        for channel in ("red", "green", "blue"):
            mean = 0.0
            for plane in range(BIT_PLANES):
                bit = 1 << (BIT_PLANES - 1 - plane)
                distribution = channels.get(f"bit:{channel}:{plane}", {"b0": 1.0})
                total = sum(distribution.values())
                on = distribution.get("b1", 0.0) / total if total else 0.0
                mean += on * bit
                variance += on * (1 - on) * bit * bit
            expected.append(_clamp_byte(mean * (255 / (2 ** BIT_PLANES - 1))))
        sigma = min(48.0, math.sqrt(variance) * (255 / (2 ** BIT_PLANES - 1)))
        candidates = [
            [expected[0], expected[1], expected[2], 0.55],
            [
                _clamp_byte(expected[0] + sigma),
                _clamp_byte(expected[1] - sigma * 0.5),
                _clamp_byte(expected[2] - sigma),
                0.30,
            ],
            [
                _clamp_byte(expected[0] - sigma),
                _clamp_byte(expected[1] + sigma * 0.5),
                _clamp_byte(expected[2] + sigma * 0.5),
                0.15,
            ],
        ]
        return _normalise(candidates)

    # silhouette
    distribution = channels.get("figure", {"subject": 0.5, "ground": 0.5})
    total = sum(distribution.values())
    subject = distribution.get("subject", 0.0) / total
    return _normalise(
        [
            [INK[0], INK[1], INK[2], max(subject, 0.02)],
            [PAPER[0], PAPER[1], PAPER[2], max(1 - subject, 0.02)],
        ]
    )


def _c(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    return _clamp_byte(rgb[0]), _clamp_byte(rgb[1]), _clamp_byte(rgb[2])


# ------------------------------------------------------- OpenAI-compatible


_HEX = re.compile(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})")
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_FENCE = re.compile(r"```[a-zA-Z]*\s*(.*?)```", re.DOTALL)


def _parse_hex(text: str) -> tuple[int, int, int] | None:
    match = _HEX.fullmatch(text.strip())
    if not match:
        return None
    digits = match.group(1)
    if len(digits) == 3:
        digits = "".join(ch * 2 for ch in digits)
    return int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16)


def parse_grid_json(text: str, size: int) -> tuple[list[tuple[int, int, int]], list[list[int]]]:
    """Extract a palette plus an index grid from a model reply.

    Accepts fenced code blocks and tolerates prose around the JSON object.
    Rows of the wrong length are resampled to the requested size so a model
    that answers 16x16 to a 24x24 request still paints.
    """
    if not isinstance(text, str):
        raise ProviderError("the model returned no text")
    candidates = [text]
    candidates.extend(match.group(1) for match in _FENCE.finditer(text))
    blob = None
    for candidate in candidates:
        match = _JSON_BLOCK.search(candidate)
        if match:
            blob = match.group(0)
            break
    if blob is None:
        raise ProviderError("the model returned no grid")
    try:
        data = json.loads(blob)
    except ValueError:
        raise ProviderError("the model returned malformed grid data") from None

    raw_palette = data.get("palette") if isinstance(data, Mapping) else None
    raw_rows = data.get("grid") if isinstance(data, Mapping) else None
    if not isinstance(raw_palette, list) or not isinstance(raw_rows, list):
        raise ProviderError("the model grid is missing its palette or rows")
    if not raw_rows or not raw_palette:
        raise ProviderError("the model returned an empty grid")

    palette: list[tuple[int, int, int]] = []
    for entry in raw_palette:
        rgb = _parse_hex(entry) if isinstance(entry, str) else None
        if rgb is None:
            raise ProviderError("the model returned an unpaintable palette entry")
        palette.append(rgb)

    rows: list[list[int]] = []
    for row in raw_rows:
        if isinstance(row, str):
            indices = [int(ch, 16) for ch in row.strip() if ch.isalnum()]
        elif isinstance(row, list):
            indices = [int(v) for v in row]
        else:
            raise ProviderError("the model returned an unusable grid row")
        if not indices:
            raise ProviderError("the model returned an empty grid row")
        rows.append(indices)

    width, height = len(rows[0]), len(rows)
    grid = [
        [rows[min(height - 1, row * height // size)][min(width - 1, col * width // size)] for col in range(size)]
        for row in range(size)
    ]
    for row in grid:
        for index in row:
            if not 0 <= index < len(palette):
                raise ProviderError("the grid points outside the palette")
    return palette, grid


class ChatProvider:
    """Any OpenAI-compatible chat completions backend, local or hosted."""

    name = "chat"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model: str = CHAT_MODEL,
        timeout: int = HTTP_TIMEOUT,
    ) -> None:
        self._key = api_key
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    def _messages(self, prompt: str, representation: str, size: int) -> list[dict[str, str]]:
        style = {
            "palette": "Compose with a limited painter's palette of at most 16 flat pigment swatches.",
            "hsl": "Spread hues broadly; vary saturation and lightness like an impressionist study.",
            "rgb": "Use vivid, high-contrast colors with pure, unmixed channels.",
            "silhouette": "Use exactly two colors: one dark for the subject, one pale for the background.",
        }[representation]
        return [
            {
                "role": "system",
                "content": (
                    "You are a colorist who answers with data, never with prose. "
                    "Reply with one JSON object and nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Paint this scene as a coarse color grid: {prompt}\n"
                    f"{style}\n"
                    f"Return exactly this shape: "
                    f'{{"palette": ["#rrggbb", ...], "grid": ["<row of palette indices>", ...]}}. '
                    f"The grid must be {size} rows of {size} characters, each character a hex "
                    f"digit indexing the palette (0-9 then a-f). Keep the palette short and reuse colors."
                ),
            },
        ]

    def _request(self, prompt: str, representation: str, size: int, strict: bool) -> str:
        body = {
            "model": self._model,
            "messages": self._messages(prompt, representation, size),
            "temperature": 0.7,
        }
        if strict:
            body["response_format"] = {"type": "json_object"}
        request = Request(
            self._url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self._key:
            request.add_header("Authorization", f"Bearer {self._key}")
        try:
            with _open(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            if strict and error.code == 400:
                return self._request(prompt, representation, size, strict=False)
            raise ProviderError(f"the model rejected the request (HTTP {error.code})") from None
        except (URLError, TimeoutError, OSError):
            raise ProviderError("the model could not be reached") from None
        except (ValueError, UnicodeError):
            raise ProviderError("the model returned unreadable data") from None
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ProviderError("the model returned no completion") from None

    def generate(self, prompt: str, representation: str, size: int) -> dict[str, Any]:
        if representation not in REPRESENTATIONS:
            raise ProviderError(f"unknown representation: {representation}")
        if size not in GRID_SIZES:
            raise ProviderError(f"unknown grid size: {size}")
        palette, grid = parse_grid_json(self._request(prompt, representation, size, strict=True), size)

        cells: list[list[list[float]]] = []
        for row in range(size):
            for col in range(size):
                own = palette[grid[row][col]]
                if representation == "silhouette":
                    candidates = [
                        [INK[0], INK[1], INK[2], 0.62],
                        [PAPER[0], PAPER[1], PAPER[2], 0.38],
                    ] if sum(own) / 3 < 128 else [
                        [PAPER[0], PAPER[1], PAPER[2], 0.62],
                        [INK[0], INK[1], INK[2], 0.38],
                    ]
                else:
                    # The cell's own pigment dominates; neighbouring pigments
                    # leak in so the painter can blend uncertain edges.
                    neighbours = [
                        palette[grid[min(size - 1, row + dr)][min(size - 1, col + dc)]]
                        for dr, dc in ((0, 1), (1, 0))
                    ]
                    candidates = [
                        [own[0], own[1], own[2], 0.66],
                        [neighbours[0][0], neighbours[0][1], neighbours[0][2], 0.20],
                        [neighbours[1][0], neighbours[1][1], neighbours[1][2], 0.14],
                    ]
                cells.append(_normalise(candidates))
        return make_field(
            size,
            size,
            cells,
            {"provider": self.name, "model": self._model, "representation": representation},
        )


# ------------------------------------------------------------- configuration


def load_env_file(path: str | os.PathLike[str], env: dict[str, str] | None = None) -> int:
    """Load a minimal .env file into the environment. Returns keys set.

    Understands `KEY=value`, `export KEY=value`, optional single or double
    quotes, blank lines, and `#` comments. The real environment always wins
    over the file. Keys are only placed into this process's environment.
    """
    target: dict[str, str] = os.environ if env is None else env
    file = Path(path)
    if not file.is_file():
        return 0
    loaded = 0
    for line in file.read_text(encoding="utf-8-sig").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("export "):
            text = text[7:].strip()
        key, separator, value = text.partition("=")
        key = key.strip()
        if not separator or not key or " " in key:
            continue
        value = value.strip()
        if value[:1] in ('"', "'"):
            quote = value[0]
            end = value.find(quote, 1)
            value = value[1:end if end != -1 else None]
        else:
            value = value.split(" #", 1)[0].strip()
        if key not in target:
            target[key] = value
            loaded += 1
    return loaded


def _chat_backend(key: str, base_url: str, model: str) -> ChatProvider:
    return ChatProvider(
        key,
        base_url=base_url or "https://api.openai.com/v1",
        model=model or CHAT_MODEL,
    )


def _jev_backend(key: str, endpoint: str, model: str) -> JevProvider:
    return JevProvider(
        key,
        endpoint=endpoint or JEV_ENDPOINT,
        model=model or JEV_MODEL,
    )


def provider_from_env(env: Mapping[str, str] | None = None) -> JevProvider | ChatProvider | None:
    """Pick a backend from environment variables. Keys never leave this scope.

    Both backends take a key, a model name, and a URL, and both work without a
    key against local or third-party endpoints that need none:

    - Jev / System One compatible: JEV_BASE_URL (or JEV_ENDPOINT), JEV_MODEL,
      JEV_API_KEY — TypeSafe's Jev or any third-party Jev-compatible model.
    - OpenAI compatible: OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_API_KEY —
      Ollama, LM Studio, vLLM, or any hosted model.

    With PAINT_PROVIDER=auto, any Jev configuration selects the Jev backend
    first; otherwise any OpenAI-compatible configuration selects that one.
    """
    env = os.environ if env is None else env
    choice = (env.get("PAINT_PROVIDER") or "auto").strip().lower() or "auto"

    jev_key = (env.get("JEV_API_KEY") or "").strip()
    jev_url = (env.get("JEV_BASE_URL") or env.get("JEV_ENDPOINT") or "").strip()
    jev_model = (env.get("JEV_MODEL") or "").strip()

    chat_key = (env.get("OPENAI_API_KEY") or env.get("AI_API_KEY") or "").strip()
    chat_url = (env.get("OPENAI_BASE_URL") or env.get("AI_BASE_URL") or "").strip()
    chat_model = (env.get("OPENAI_MODEL") or env.get("AI_MODEL") or "").strip()

    has_jev = bool(jev_key or jev_url or jev_model)
    has_chat = bool(chat_key or chat_url or chat_model)

    if choice == "jev":
        return _jev_backend(jev_key, jev_url, jev_model) if has_jev else None
    if choice in ("chat", "openai"):
        return _chat_backend(chat_key, chat_url, chat_model) if has_chat else None
    if has_jev:
        return _jev_backend(jev_key, jev_url, jev_model)
    if has_chat:
        return _chat_backend(chat_key, chat_url, chat_model)
    return None
