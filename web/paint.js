// Pixel Play painter: turns a pigment-probability field into a textured canvas.
//
// Pure functions over typed arrays — no DOM — so the same code runs inside the
// paint worker and inside plain Node for tests. Every random draw comes from a
// seeded xorshift stream, so one (field, seed) pair always paints one image.

const LN4 = Math.log(4);
const TAU = Math.PI * 2;

/** Stable 32-bit hash of a string, usable as a PRNG seed. */
export function hashSeed(text) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0 || 1;
}

function makeRandom(seed) {
  let s = (seed >>> 0) || 0x9e3779b9;
  return () => {
    s ^= s << 13;
    s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 4294967296;
  };
}

/** Reject anything that is not a grid of weighted pigments. */
function unusable(field) {
  if (!field || !Array.isArray(field.cells)) return true;
  const { width, height, cells } = field;
  if (!Number.isInteger(width) || !Number.isInteger(height) || width * height !== cells.length) {
    return true;
  }
  for (const cell of cells) {
    if (!Array.isArray(cell) || cell.length === 0) return true;
    for (const pigment of cell) {
      if (!Array.isArray(pigment) || pigment.length !== 4) return true;
      if (!pigment.every((value) => Number.isFinite(value)) || pigment[3] <= 0) return true;
    }
  }
  return false;
}

/**
 * Read the field once: mean pigment per cell, plus entropy of each cell's
 * pigment distribution (0 = certain, 1 = fully ambiguous) and a luma grid the
 * stroke engine follows.
 */
function analyse(field) {
  const { width: gw, height: gh, cells } = field;
  const means = new Float32Array(gw * gh * 3);
  const entropy = new Float32Array(gw * gh);
  const luma = new Float32Array(gw * gh);
  for (let i = 0; i < gw * gh; i += 1) {
    const pigments = cells[i];
    let r = 0;
    let g = 0;
    let b = 0;
    let h = 0;
    for (const pigment of pigments) {
      const w = pigment[3];
      r += pigment[0] * w;
      g += pigment[1] * w;
      b += pigment[2] * w;
      if (w > 0) h -= w * Math.log(w);
    }
    means[i * 3] = r;
    means[i * 3 + 1] = g;
    means[i * 3 + 2] = b;
    entropy[i] = Math.min(1, h / LN4);
    luma[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }
  return { means, entropy, luma, gw, gh };
}

function bilinear(values, gw, gh, x, y, channels) {
  const gx = Math.min(gw - 1.001, Math.max(0, x));
  const gy = Math.min(gh - 1.001, Math.max(0, y));
  const x0 = Math.floor(gx);
  const y0 = Math.floor(gy);
  const fx = gx - x0;
  const fy = gy - y0;
  const x1 = Math.min(gw - 1, x0 + 1);
  const y1 = Math.min(gh - 1, y0 + 1);
  const out = new Array(channels);
  for (let c = 0; c < channels; c += 1) {
    const top = values[(y0 * gw + x0) * channels + c] * (1 - fx) + values[(y0 * gw + x1) * channels + c] * fx;
    const bottom = values[(y1 * gw + x0) * channels + c] * (1 - fx) + values[(y1 * gw + x1) * channels + c] * fx;
    out[c] = top * (1 - fy) + bottom * fy;
  }
  return out;
}

/** Pick a pigment near (gx, gy): usually the cell's own, sometimes a neighbour's. */
function pickPigment(field, random, gx, gy) {
  const { width: gw, height: gh, cells } = field;
  const drift = random() < 0.66 ? 0 : random() < 0.5 ? 1 : 2;
  const col = Math.min(gw - 1, Math.max(0, Math.round(gx) + (drift === 1 ? (random() < 0.5 ? -1 : 1) : 0)));
  const row = Math.min(gh - 1, Math.max(0, Math.round(gy) + (drift === 2 ? (random() < 0.5 ? -1 : 1) : 0)));
  const pigments = cells[row * gw + col];
  let ticket = random();
  for (const pigment of pigments) {
    ticket -= pigment[3];
    if (ticket <= 0) return pigment;
  }
  return pigments[pigments.length - 1];
}

/** Unit direction that runs along a luminance contour at this grid position. */
function contourDirection(luma, gw, gh, gx, gy, random) {
  const x = Math.min(gw - 2, Math.max(1, Math.round(gx)));
  const y = Math.min(gh - 2, Math.max(1, Math.round(gy)));
  const dx = luma[y * gw + x + 1] - luma[y * gw + x - 1];
  const dy = luma[(y + 1) * gw + x] - luma[(y - 1) * gw + x];
  let vx = -dy;
  let vy = dx;
  const magnitude = Math.hypot(vx, vy);
  if (magnitude < 1e-3) {
    const angle = random() * TAU;
    return [Math.cos(angle), Math.sin(angle)];
  }
  vx /= magnitude;
  vy /= magnitude;
  const wobble = (random() - 0.5) * 0.6;
  const cos = Math.cos(wobble);
  const sin = Math.sin(wobble);
  return [vx * cos - vy * sin, vx * sin + vy * cos];
}

/** One soft elliptical dab: blends toward a pigment and raises the height map. */
function dab(data, height, px, cx, cy, radius, pigment, alpha) {
  const left = Math.max(0, Math.floor(cx - radius));
  const right = Math.min(px - 1, Math.ceil(cx + radius));
  const top = Math.max(0, Math.floor(cy - radius));
  const bottom = Math.min(px - 1, Math.ceil(cy + radius));
  for (let y = top; y <= bottom; y += 1) {
    for (let x = left; x <= right; x += 1) {
      const dx = x + 0.5 - cx;
      const dy = y + 0.5 - cy;
      const d2 = (dx * dx + dy * dy) / (radius * radius);
      if (d2 >= 1) continue;
      const falloff = (1 - d2) * (1 - d2);
      const mix = alpha * falloff;
      const index = (y * px + x) * 4;
      data[index] += (pigment[0] - data[index]) * mix;
      data[index + 1] += (pigment[1] - data[index + 1]) * mix;
      data[index + 2] += (pigment[2] - data[index + 2]) * mix;
      height[y * px + x] += falloff * alpha;
    }
  }
}

function blurHeight(height, px) {
  const scratch = new Float32Array(height.length);
  for (let pass = 0; pass < 2; pass += 1) {
    for (let y = 0; y < px; y += 1) {
      for (let x = 0; x < px; x += 1) {
        let sum = 0;
        for (let dy = -1; dy <= 1; dy += 1) {
          const yy = Math.min(px - 1, Math.max(0, y + dy));
          for (let dx = -1; dx <= 1; dx += 1) {
            const xx = Math.min(px - 1, Math.max(0, x + dx));
            sum += height[yy * px + xx];
          }
        }
        scratch[y * px + x] = sum / 9;
      }
    }
    height.set(scratch);
  }
}

/**
 * Paint a field into a square RGBA buffer.
 *
 * Stage 1 spreads the mean pigment field as an underpainting. Stage 2 drags
 * contour-following strokes whose pigments are sampled from the per-cell
 * distributions. Stage 3 lights the height map those strokes leave behind for
 * impasto relief, modulated by local ambiguity. Stage 4 adds weave and a
 * vignette so the result reads as a physical panel.
 */
export function paintPixels(field, size = 560, seed = 1) {
  if (unusable(field)) {
    throw new Error("paintPixels needs a square field of pigment cells");
  }
  const px = Math.max(32, Math.min(1024, Math.round(size)));
  const random = makeRandom(seed);
  const { means, entropy, luma, gw, gh } = analyse(field);
  const data = new Uint8ClampedArray(px * px * 4);
  const height = new Float32Array(px * px);

  // Stage 1 — underpainting.
  for (let y = 0; y < px; y += 1) {
    for (let x = 0; x < px; x += 1) {
      const [r, g, b] = bilinear(means, gw, gh, ((x + 0.5) / px) * gw - 0.5, ((y + 0.5) / px) * gh - 0.5, 3);
      const grain = (random() - 0.5) * 9;
      const index = (y * px + x) * 4;
      data[index] = r + grain;
      data[index + 1] = g + grain;
      data[index + 2] = b + grain;
      data[index + 3] = 255;
    }
  }

  // Stage 2 — strokes.
  const cellPx = px / gw;
  const strokeCount = Math.round(gw * gh * 3.4);
  for (let s = 0; s < strokeCount; s += 1) {
    const gx = random() * (gw - 1);
    const gy = random() * (gh - 1);
    const pigment = pickPigment(field, random, gx, gy);
    const [dx, dy] = contourDirection(luma, gw, gh, gx, gy, random);
    const length = cellPx * (1.1 + random() * 2.7);
    const width = cellPx * (0.2 + random() * 0.45);
    const alpha = 0.14 + random() * 0.24;
    const curve = (random() - 0.5) * 0.7;
    const steps = Math.max(2, Math.round(length / (width * 0.6)));
    const originX = ((gx + 0.5) / gw) * px;
    const originY = ((gy + 0.5) / gh) * px;
    for (let step = 0; step <= steps; step += 1) {
      const t = step / steps - 0.5;
      const bend = curve * (t * t - 0.25) * length;
      const x = originX + dx * t * length - dy * bend;
      const y = originY + dy * t * length + dx * bend;
      dab(data, height, px, x, y, width * (0.7 + 0.3 * (1 - Math.abs(t * 2))), pigment, alpha);
    }
  }

  // Stage 3 — impasto relief lit from the upper left.
  blurHeight(height, px);
  for (let y = 0; y < px; y += 1) {
    for (let x = 0; x < px; x += 1) {
      const gx = Math.min(gw - 1, Math.max(0, (x / px) * gw));
      const gy = Math.min(gh - 1, Math.max(0, (y / px) * gh));
      const ambiguity = entropy[Math.round(gy) * gw + Math.round(gx)];
      const upLeft = height[Math.max(0, y - 1) * px + Math.max(0, x - 1)];
      const downRight = height[Math.min(px - 1, y + 1) * px + Math.min(px - 1, x + 1)];
      const relief = (upLeft - downRight) * (26 + 42 * ambiguity);
      const index = (y * px + x) * 4;
      data[index] += relief;
      data[index + 1] += relief;
      data[index + 2] += relief;
    }
  }

  // Stage 4 — weave, vignette.
  for (let y = 0; y < px; y += 1) {
    for (let x = 0; x < px; x += 1) {
      const weave = 2.4 * (Math.sin(x * 1.7) + Math.sin(y * 1.7));
      const nx = (x / px) * 2 - 1;
      const ny = (y / px) * 2 - 1;
      const distance = (nx * nx + ny * ny) / 2;
      const shade = (1 - 0.34 * distance * distance) * (1 + weave / 255);
      const index = (y * px + x) * 4;
      data[index] *= shade;
      data[index + 1] *= shade;
      data[index + 2] *= shade;
      data[index + 3] = 255;
    }
  }

  return data;
}
