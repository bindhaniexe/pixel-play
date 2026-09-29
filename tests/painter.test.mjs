// Offline painter checks: determinism, output shape, input validation.
import test from "node:test";
import assert from "node:assert/strict";
import { hashSeed, paintPixels } from "../web/paint.js";

function syntheticField(size) {
  const cells = [];
  for (let row = 0; row < size; row += 1) {
    for (let col = 0; col < size; col += 1) {
      cells.push([
        [(col * 255) / (size - 1), (row * 255) / (size - 1), 128, 0.7],
        [30, 30, 60, 0.3],
      ]);
    }
  }
  return { width: size, height: size, cells, meta: { provider: "test" } };
}

test("one (field, seed) pair paints one identical image", () => {
  const field = syntheticField(8);
  const seed = hashSeed("a lighthouse at dusk");
  const first = paintPixels(field, 256, seed);
  const second = paintPixels(field, 256, seed);
  assert.equal(first.length, 256 * 256 * 4);
  assert.ok(Buffer.from(first).equals(Buffer.from(second)));
});

test("different seeds give different canvases", () => {
  const field = syntheticField(8);
  const first = paintPixels(field, 256, hashSeed("tulips"));
  const second = paintPixels(field, 256, hashSeed("peonies"));
  assert.ok(!Buffer.from(first).equals(Buffer.from(second)));
});

test("every pixel is opaque", () => {
  const pixels = paintPixels(syntheticField(6), 128, hashSeed("opaque"));
  for (let i = 3; i < pixels.length; i += 4) {
    if (pixels[i] !== 255) assert.fail(`alpha drifted at ${i}`);
  }
});

test("hashSeed is stable and nonzero", () => {
  assert.equal(hashSeed("pixel play"), hashSeed("pixel play"));
  assert.notEqual(hashSeed("pixel play"), hashSeed("pixel play "));
  assert.ok(hashSeed("") > 0);
});

test("malformed fields are refused", () => {
  assert.throws(() => paintPixels(null, 128, 1), /field/);
  assert.throws(() => paintPixels({ width: 2, height: 2, cells: [] }, 128, 1), /field/);
  assert.throws(
    () => paintPixels({ width: 1, height: 1, cells: [[[255, 0, 0]]] }, 128, 1),
    /paintPixels/,
  );
});
