#!/usr/bin/env node
// Concurrency / TTFA tester for vox-serve streaming.
// Requires Node 18+ (built-in fetch, FormData, Blob). No npm install needed.
//
//   node concurrency_test.js            # uses config below
//   node concurrency_test.js 32 clone   # 32 users, cloning
//   node concurrency_test.js 64 noclone # 64 users, default voice

const fs = require("fs");

// ---------------- CONFIG: edit these ----------------
const HOST = "127.0.0.1";   // <-- your server IP, e.g. "203.0.113.5"
const PORT = 2200;          // <-- server port
const REF_WAV = "./reference_hf.wav"; // <-- clean 3-10s speech wav for cloning
const TEXT = "Hello, this is a concurrency test of the streaming text to speech system.";
let CONCURRENCY = 64;       // number of simultaneous users
let USE_CLONE = true;       // true = clone from REF_WAV, false = default voice
// ----------------------------------------------------

// optional CLI overrides: node concurrency_test.js <N> <clone|noclone>
if (process.argv[2]) CONCURRENCY = parseInt(process.argv[2], 10);
if (process.argv[3]) USE_CLONE = process.argv[3] !== "noclone";

const URL = `http://${HOST}:${PORT}/generate`;

function pct(arr, p) {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const k = (p / 100) * (s.length - 1);
  const lo = Math.floor(k), hi = Math.min(lo + 1, s.length - 1);
  return s[lo] * (1 - (k - lo)) + s[hi] * (k - lo);
}

async function oneRequest(refBytes) {
  const form = new FormData();
  form.append("text", TEXT);
  form.append("streaming", "true");
  if (refBytes) form.append("audio", new Blob([refBytes], { type: "audio/wav" }), "ref.wav");

  const start = performance.now();
  try {
    const res = await fetch(URL, { method: "POST", body: form });
    if (!res.ok) return { ok: false, err: `HTTP ${res.status}` };
    let chunkIdx = 0, ttfa = null;
    for await (const chunk of res.body) {
      chunkIdx++;
      if (chunkIdx === 1) continue;      // first chunk is the WAV header
      if (ttfa === null) ttfa = performance.now() - start;
    }
    return { ok: ttfa !== null, ttfa };
  } catch (e) {
    return { ok: false, err: String(e) };
  }
}

async function main() {
  const refBytes = USE_CLONE ? fs.readFileSync(REF_WAV) : null;
  console.log(`Target ${URL} | concurrency=${CONCURRENCY} | mode=${USE_CLONE ? "cloning" : "no-clone"}`);
  if (USE_CLONE) console.log(`reference: ${REF_WAV} (${refBytes.length} bytes)`);

  const t0 = performance.now();
  const results = await Promise.all(
    Array.from({ length: CONCURRENCY }, () => oneRequest(refBytes))
  );
  const wall = performance.now() - t0;

  const ttfas = results.filter(r => r.ok).map(r => r.ttfa);
  const fails = results.filter(r => !r.ok);

  console.log(`\nok=${ttfas.length}/${results.length}  failed=${fails.length}  wall=${wall.toFixed(0)}ms`);
  if (ttfas.length) {
    const mean = ttfas.reduce((a, b) => a + b, 0) / ttfas.length;
    console.log("TTFA (ms): " +
      `mean=${mean.toFixed(0)} p50=${pct(ttfas, 50).toFixed(0)} ` +
      `p90=${pct(ttfas, 90).toFixed(0)} p99=${pct(ttfas, 99).toFixed(0)} ` +
      `min=${Math.min(...ttfas).toFixed(0)} max=${Math.max(...ttfas).toFixed(0)}`);
  }
  if (fails.length) console.log("errors:", [...new Set(fails.map(f => f.err))]);
}

main();
