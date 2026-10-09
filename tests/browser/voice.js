/* Voice mode end to end: a fake microphone talks through a whole estimate.
 *
 * Chrome's fake audio device plays a recording of a "customer" (a different
 * voice from Sam's) into the page's microphone. Everything after that is real:
 * voiceloop's speech detector, OpenAI live transcription, the LangGraph turn,
 * signed speech, OpenAI TTS. Costs a few cents per run.
 *
 *   VOICE_WAV=/path/customer.wav BASE=http://localhost:8080 node tests/browser/voice.js
 *
 * Make the recording with tests/browser/make_customer_wav.py (see README).
 */
const { chromium } = require("playwright");

const BASE = process.env.BASE || "http://localhost:8080";
const WAV = process.env.VOICE_WAV;
if (!WAV) { console.error("Set VOICE_WAV to the customer recording."); process.exit(2); }

const fails = [];
const check = (cond, label, extra) => {
  console.log(`  ${cond ? "✓" : "✗"} ${label}${!cond && extra ? "  → " + extra : ""}`);
  if (!cond) fails.push(label);
};

const nextFriday = (() => {
  // Next WEEK's Friday, the rule in lead_form.parse_spoken_date (Los Angeles date).
  const la = new Date(new Date().toLocaleString("en-US", { timeZone: "America/Los_Angeles" }));
  const toMonday = 7 - ((la.getDay() + 6) % 7);
  const d = new Date(la); d.setDate(la.getDate() + toMonday + 4);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
})();

(async () => {
  const browser = await chromium.launch({
    args: [
      "--use-fake-ui-for-media-stream",
      "--use-fake-device-for-media-stream",
      `--use-file-for-fake-audio-capture=${WAV}`,
      "--autoplay-policy=no-user-gesture-required",
    ],
  });
  const context = await browser.newContext({ viewport: { width: 390, height: 844 }, permissions: ["microphone"] });
  const page = await context.newPage();
  page.on("pageerror", (e) => { console.log("  ✗ PAGE ERROR: " + e.message); fails.push("pageerror"); });
  const DEBUG = !!process.env.DEBUG;
  page.on("console", (m) => { if (DEBUG || /\[voice\]/.test(m.text())) console.log("    console: " + m.text()); });
  page.on("requestfinished", async (r) => {
    if (DEBUG && r.url().includes("/api/")) console.log(`    ${r.method()} ${new URL(r.url()).pathname} ${(await r.response())?.status()} ${(r.postData() || "").slice(0, 120)}`);
  });

  await page.goto(BASE);
  await page.waitForSelector("#talk-chip:not([hidden])", { timeout: 10000 });
  check(await page.isVisible("#talk"), "the mic button is offered when voice is available");

  // A log of what the voice bar shows, with timestamps, kept by the page itself.
  await page.evaluate(() => {
    window.__voice = [];
    const bar = document.getElementById("voicebar");
    const heard = document.getElementById("voice-heard");
    const t0 = performance.now();
    new MutationObserver(() => window.__voice.push({ t: performance.now() - t0, state: bar.dataset.state, heard: heard.textContent }))
      .observe(bar, { attributes: true, subtree: true, childList: true, characterData: true });
  });
  if (DEBUG) {
    let seen = 0;
    setInterval(async () => {
      const log = await page.evaluate(() => window.__voice).catch(() => []);
      for (const e of log.slice(seen)) console.log(`    [${(e.t / 1000).toFixed(1)}s] ${e.state} ${e.heard}`);
      seen = log.length;
    }, 1000).unref();
  }

  await page.click("#talk-chip");
  await page.waitForSelector("#voicebar:not([hidden])");
  check(true, "one tap starts voice mode");

  // ── A question, hands-free ─────────────────────────────────────────────
  const bubble = (re) => page.waitForFunction(
    (src) => [...document.querySelectorAll(".msg")].some((m) => new RegExp(src, "i").test(m.textContent)),
    re.source, { timeout: 60000 });
  await bubble(/two movers/);
  check(true, "the spoken question appears in the conversation");
  const mine = await page.locator(".msg.me").allInnerTexts();
  check(!mine.some((t) => /talk to me instead/i.test(t)), 'the "talk" chip starts voice, it is not sent as a message', mine.join(" | "));
  await bubble(/\$1\d\d/);
  check(true, "the answer comes back with our real rate");

  // ── The estimate, by voice ─────────────────────────────────────────────
  await page.waitForSelector(".recap", { timeout: 200000 }).catch(async (e) => {
    console.log("    thread: " + (await page.locator("#thread").innerText()).replace(/\n+/g, " | ").slice(-1500));
    console.log("    card: " + (await page.locator("#slot").innerText()).replace(/\n+/g, " | "));
    throw e;
  });
  const recap = await page.locator(".recap").innerText();
  for (const [label, value] of [
    ["name", "Jordan Lee"], ["phone", "(323) 555-0142"], ["email", "jordan@example.com"],
    ["from zip", "90026"], ["to zip", "90401"], ["move date", nextFriday], ["size", "2 bedrooms"],
  ]) check(recap.includes(value), `spoken ${label} recorded as ${value}`, recap.replace(/\n/g, " | "));
  check(!/Stairs|elevator/i.test(recap), '"skip" skipped the optional stairs question');

  await page.waitForFunction(() => {
    const all = [...document.querySelectorAll(".msg.them")];
    const recap = document.querySelector(".recap");
    return recap && all.some((m) => recap.compareDocumentPosition(m) & Node.DOCUMENT_POSITION_FOLLOWING);
  }, null, { timeout: 60000 });
  const last = (await page.locator(".msg.them").last().innerText()).trim();
  check(/manager|sent|over/i.test(last), '"yes, send it" sent the estimate', last);

  // ── Timings ────────────────────────────────────────────────────────────
  const log = await page.evaluate(() => window.__voice);
  const gaps = [];
  for (let i = 0; i < log.length; i++) {
    if (!/^Heard:/.test(log[i].heard || "")) continue;
    const spoke = log.slice(i).find((e) => e.state === "speaking");
    if (spoke) gaps.push(Math.round(spoke.t - log[i].t));
  }
  const uniq = gaps.filter((g, i) => i === 0 || g !== gaps[i - 1]);
  const median = uniq.sort((a, b) => a - b)[Math.floor(uniq.length / 2)];
  console.log(`    end of speech → Sam speaking (ms): median ${median}, all ${uniq.join(", ")}`);

  // ── Hanging up ─────────────────────────────────────────────────────────
  await page.click("#voice-end");
  await page.waitForSelector("#voicebar", { state: "hidden" });
  check((await page.getAttribute("#talk", "aria-pressed")) === "false", "End switches voice off");

  await browser.close();
  console.log(fails.length ? `\n${fails.length} failed` : "\nall passed");
  process.exit(fails.length ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(1); });
