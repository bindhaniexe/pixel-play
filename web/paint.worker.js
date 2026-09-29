// Off-main-thread painting: receive a field, hand back finished pixels.
import { paintPixels } from "./paint.js";

self.onmessage = (event) => {
  const { id, field, size, seed } = event.data;
  try {
    const pixels = paintPixels(field, size, seed);
    self.postMessage({ id, ok: true, size, buffer: pixels.buffer }, [pixels.buffer]);
  } catch (error) {
    self.postMessage({ id, ok: false, message: String(error && error.message ? error.message : error) });
  }
};
