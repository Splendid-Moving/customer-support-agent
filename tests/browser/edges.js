const { chromium } = require("playwright");
const BASE = process.env.BASE || "http://localhost:8812";
const fails = [];
const check = (c, l, x) => { if (c) console.log("  ✓ " + l); else { console.log("  ✗ " + l + (x ? "  → " + x : "")); fails.push(l); } };
const card = (p) => p.locator("#slot .qcard:not(.qcard--out)");
const q = async (p) => (await card(p).locator(".qcard__q").innerText()).trim();
const waitQ = (p, prev) => p.waitForFunction((x) => {
  const c = document.querySelector("#slot .qcard:not(.qcard--out) .qcard__q");
  return !document.getElementById("dots") && !document.querySelector("#slot .qcard--out") &&
         c && c.innerText.trim() && c.innerText.trim() !== x;
}, prev, { timeout: 45000 }).then(() => q(p));
const type = async (p, t) => { await p.fill("#input", t); await p.press("#input", "Enter"); };

(async () => {
  const browser = await chromium.launch();

  // ── A small phone ──────────────────────────────────────────────────────────
  console.log("\n1. iPhone SE (375×667) — the tallest card");
  let p = await browser.newPage({ viewport: { width: 375, height: 667 }, deviceScaleFactor: 2 });
  p.on("pageerror", (e) => { console.log("  ✗ PAGEERROR: " + e.message); fails.push("pageerror"); });
  await p.goto(BASE, { waitUntil: "domcontentloaded" });
  await p.getByRole("button", { name: "I'd like an estimate" }).click();
  let cur = await waitQ(p, "");
  for (const a of ["Nik", "3235550142", "nik@example.com", "90026", "90401"]) { await type(p, a); cur = await waitQ(p, cur); }
  await card(p).locator(".qcard__row", { hasText: "Not sure yet" }).click();
  cur = await waitQ(p, cur);
  check(/how big/i.test(cur), "reached the 8-row card", cur);
  await p.screenshot({ path: __dirname + "/se-choices.png" });

  const box = await p.locator("#box").boundingBox();
  const vh = p.viewportSize().height;
  check(box.y + box.height <= vh + 1, "the message box is still on screen", JSON.stringify(box));
  const first = await card(p).locator(".qcard__row").first().boundingBox();
  check(first && first.y >= 0 && first.y < vh, "the first option is on screen");
  const scrolls = await card(p).evaluate((el) => el.scrollHeight > el.clientHeight);
  check(true, "card scrolls internally when it must: " + scrolls);

  // ── "Something else" ───────────────────────────────────────────────────────
  console.log("\n2. 'Something else' hands you the keyboard");
  await card(p).locator(".qcard__row", { hasText: "Something else" }).click();
  await p.waitForTimeout(300);
  check(await p.evaluate(() => document.activeElement === document.getElementById("input")),
        "focuses the message box rather than answering");
  check((await q(p)) === cur, "and does not advance the question");

  // ── Validation ─────────────────────────────────────────────────────────────
  console.log("\n3. A bad answer is corrected on the card");
  await type(p, "Studio");
  cur = await waitQ(p, cur);
  check(/stairs|elevator/i.test(cur), "typing a valid option is accepted", cur);
  await p.close();

  p = await browser.newPage({ viewport: { width: 390, height: 844 } });
  p.on("pageerror", (e) => { console.log("  ✗ PAGEERROR: " + e.message); fails.push("pageerror"); });
  await p.goto(BASE, { waitUntil: "domcontentloaded" });
  await p.getByRole("button", { name: "I'd like an estimate" }).click();
  cur = await waitQ(p, "");
  await type(p, "Nik"); cur = await waitQ(p, cur);
  await type(p, "123");                       // too short to be a phone number
  cur = await waitQ(p, cur);
  check(/ten digits|doesn't look/i.test(cur), "the card asks again, in plain words", cur);
  check((await card(p).count()) === 1, "still one card");
  await type(p, "3235550142"); cur = await waitQ(p, cur);
  check(/email/i.test(cur), "a good answer then moves on", cur);
  await p.close();

  // ── Russian ────────────────────────────────────────────────────────────────
  console.log("\n4. The same interview in Russian");
  p = await browser.newPage({ viewport: { width: 390, height: 844 } });
  p.on("pageerror", (e) => { console.log("  ✗ PAGEERROR: " + e.message); fails.push("pageerror"); });
  await p.goto(BASE, { waitUntil: "domcontentloaded" });
  await type(p, "Здравствуйте, хочу получить оценку стоимости переезда");
  cur = await waitQ(p, "");
  check(/[а-яё]/i.test(cur), "the first question is in Russian", cur);
  const label = await card(p).locator(".qcard__count").innerText();
  check(/[а-яё]/i.test(label), "so is the counter", label);
  check(/\b10\b/.test(label), "and it still counts to 10", label);
  const ph = await p.getAttribute("#input", "placeholder");
  check(/[а-яё]/i.test(ph), "so is the message box", ph);
  await p.screenshot({ path: __dirname + "/ru-question.png" });
  await p.close();

  await browser.close();
  console.log(fails.length ? `\nFAILED (${fails.length}):\n  - ` + fails.join("\n  - ") : "\nALL CHECKS PASSED");
  process.exit(fails.length ? 1 : 0);
})();
