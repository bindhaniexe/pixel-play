// Pixel Play studio UI: prompt composer, painting carousel, settings panel.
import { hashSeed, paintPixels } from "./paint.js";

const CANVAS = 560;
const $ = (id) => document.getElementById(id);

const ui = {
  toggle: $("toggle"),
  sidebar: $("sidebar"),
  close: $("close"),
  painter: $("painter"),
  representation: $("representation"),
  size: $("size"),
  track: $("track"),
  navigation: $("navigation"),
  previous: $("previous"),
  next: $("next"),
  position: $("position"),
  form: $("prompt-form"),
  prompt: $("prompt"),
  paint: $("paint"),
  status: $("status"),
};

const gallery = [];
let selected = 0;
let busy = false;
let job = 0;
const pending = new Map();

function say(message, isError = false) {
  ui.status.textContent = message;
  ui.status.classList.toggle("error", isError);
}

// ---- painting worker, with a main-thread fallback ----

let worker = null;
try {
  worker = new Worker("./paint.worker.js", { type: "module" });
  worker.onmessage = (event) => {
    const { id, ok, size, buffer, message } = event.data;
    const waiter = pending.get(id);
    if (!waiter) return;
    pending.delete(id);
    if (ok) waiter.resolve(new Uint8ClampedArray(buffer));
    else waiter.reject(new Error(message || "the painter stopped"));
  };
  worker.onerror = () => {
    worker = null;
  };
} catch {
  worker = null;
}

function render(field, seed) {
  const id = ++job;
  if (!worker) {
    return Promise.resolve().then(() => paintPixels(field, CANVAS, seed));
  }
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, field, size: CANVAS, seed });
    setTimeout(() => {
      if (pending.has(id)) {
        pending.delete(id);
        reject(new Error("the painter took too long"));
      }
    }, 30000);
  });
}

// ---- gallery ----

function mount(field, prompt) {
  const canvas = document.createElement("canvas");
  canvas.width = CANVAS;
  canvas.height = CANVAS;
  const context = canvas.getContext("2d");
  const image = context.createImageData(CANVAS, CANVAS);
  image.data.set(field);
  context.putImageData(image, 0, 0);

  const frame = document.createElement("div");
  frame.className = "frame";
  frame.appendChild(canvas);
  if (gallery.length === 0) ui.track.textContent = "";
  ui.track.appendChild(frame);
  gallery.push({ frame, prompt });
  show(gallery.length - 1);
}

function show(index) {
  selected = Math.max(0, Math.min(gallery.length - 1, index));
  ui.track.style.transform = `translateX(${-selected * 100}%)`;
  ui.navigation.hidden = gallery.length < 2;
  ui.position.textContent = `${selected + 1} / ${gallery.length}`;
}

ui.previous.addEventListener("click", () => show(selected - 1));
ui.next.addEventListener("click", () => show(selected + 1));

// ---- settings panel ----

function setPanel(open) {
  ui.sidebar.hidden = !open;
  ui.toggle.setAttribute("aria-expanded", String(open));
}
ui.toggle.addEventListener("click", () => setPanel(ui.sidebar.hidden));
ui.close.addEventListener("click", () => setPanel(false));

// ---- composer ----

ui.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  const prompt = ui.prompt.value.trim();
  if (!prompt) return;

  busy = true;
  ui.paint.disabled = true;
  say("Grinding pigments…");
  const started = performance.now();
  try {
    const response = await fetch("/api/paint", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        representation: ui.representation.value,
        size: Number(ui.size.value),
      }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || `the studio replied with HTTP ${response.status}`);
    }
    const pixels = await render(payload.field, hashSeed(prompt + ui.representation.value + ui.size.value));
    mount(pixels, prompt);
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    const model = payload.field.meta || {};
    say(`Painted in ${seconds}s · model ${payload.ms ?? "?"} ms · ${model.provider ?? "field"}`);
  } catch (error) {
    say(error && error.message ? error.message : "something went wrong", true);
  } finally {
    busy = false;
    ui.paint.disabled = false;
    ui.prompt.focus();
  }
});

// ---- studio status ----

(async () => {
  try {
    const response = await fetch("/api/status");
    const info = await response.json();
    ui.painter.textContent = info.configured
      ? `Painter ready: ${info.provider} (${info.model}).`
      : "No painter key on this machine yet — set JEV_API_KEY or OPENAI_API_KEY for the helper.";
  } catch {
    ui.painter.textContent = "The local helper is not answering.";
  }
})();
