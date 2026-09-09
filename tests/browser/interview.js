/* Drives the real page in a real browser, on a phone-sized viewport. */
const { chromium } = require("playwright");

const BASE = process.env.BASE || "http://localhost:8812";
const fails = [];
const ok = (label) => console.log("  ✓ " + label);
const check = (cond, label, extra) => {
  if (cond) ok(label);
  else { console.log("  ✗ " + label + (extra ? "  → " + extra : "")); fails.push(label); }
};

const slotCards = (p) => p.locator("#slot .qcard:not(.qcard--out)");
const question = async (p) => (await slotCards(p).locator(".qcard__q").innerText()).trim();
const counter = async (p) => {
  const c = slotCards(p).locator(".qcard__count");
  return (await c.count()) ? (await c.innerText()).trim() : "";
};
const rows = async (p) => slotCards(p).locator(".qcard__row").allInnerTexts();

/*
 * A new question, and the page finished with the last one.
 *
 * The card renders a beat before the turn ends, and the page ignores input
 * while it is mid-turn — correctly, that is what stops a double submit. So
 * waiting on the text alone raced: the next Enter landed in that gap and was
 * dropped. Waiting for the thinking dots to clear and the outgoing card to be
 * gone is waiting for the page to actually be idle.
 */
async function waitForQuestion(p, previous, timeout = 45000) {
  await p.waitForFunction(
    (prev) => {
      const card = document.querySelector("#slot .qcard:not(.qcard--out) .qcard__q");
      const settled = !document.getElementById("dots") &&
                      !document.querySelector("#slot .qcard--out");
      return settled && card && card.innerText.trim() && card.innerText.trim() !== prev;
    },
    previous, { timeout }
  );
  return question(p);
}

async function type(p, text) {
  await p.fill("#input", text);
  await p.press("#input", "Enter");
}

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  page.on("pageerror", (e) => { console.log("  ✗ PAGE ERROR: " + e.message); fails.push("pageerror: " + e.message); });
  page.on("console", (m) => { if (m.type() === "error") { console.log("  ✗ CONSOLE: " + m.text()); fails.push("console: " + m.text()); } });

  await page.goto(BASE, { waitUntil: "domcontentloaded" });

  console.log("\n1. The opening screen");
  check(await page.locator("#slot").isHidden(), "no card before the interview starts");
  check((await page.locator(".chip").count()) === 6, "six suggestion chips");

  console.log("\n2. Starting the interview");
  await page.getByRole("button", { name: "I'd like an estimate" }).click();
  let q = await waitForQuestion(page, "");
  check(/name/i.test(q), "asks for a name", q);
  check((await counter(page)) === "Question 1 of 10", "counter reads 1 of 10", await counter(page));
  check((await slotCards(page).count()) === 1, "exactly one card in the slot");
  check((await page.locator(".msg.them").count()) >= 1, "Sam's opening line is in the transcript");

  console.log("\n3. Typing an answer");
  const bubblesBefore = await page.locator(".msg.me").count();
  await type(page, "Nik");
  q = await waitForQuestion(page, q);
  check(/number/i.test(q), "moves on to the phone number", q);
  check((await counter(page)) === "Question 2 of 10", "counter advances", await counter(page));
  check((await slotCards(page).count()) === 1, "still exactly one card — the old one is gone");
  check((await page.locator(".msg.me").count()) === bubblesBefore, "the answer did not stack in the transcript");
  check((await page.inputValue("#input")) === "", "the box is cleared");

  console.log("\n4. Through the contact details");
  for (const [answer, expect] of [["3235550142", /email/i], ["nik@example.com", /zip/i], ["90026", /zip/i], ["90401", /date/i]]) {
    await type(page, answer);
    q = await waitForQuestion(page, q);
    check(expect.test(q), `answered "${answer}" → ${q.slice(0, 44)}`, q);
  }

  console.log("\n5. The date step (the iOS bug)");
  let r = await rows(page);
  check(r.some((x) => /pick a date/i.test(x)), "a labelled 'Pick a date' row", JSON.stringify(r));
  check(r.some((x) => /not sure/i.test(x)), "a 'Not sure yet' row");
  // By visibility, not by text: Chrome's innerText returns the text of a
  // display:none element, so reading the rows would call a hidden row present.
  const useDate = slotCards(page).locator(".qcard__row", { hasText: "Use this date" });
  check(!(await useDate.isVisible()), "'Use this date' is hidden until a date is picked");
  const native = slotCards(page).locator("input.qcard__date");
  check((await native.count()) === 1, "the native picker covers its row");

  const before = q;
  await native.fill("2026-11-20");           // fires `change`, as the iOS wheel does
  await page.waitForTimeout(1200);
  check((await question(page)) === before, "picking a date does NOT answer the question");
  r = await rows(page);
  check(await useDate.isVisible(), "'Use this date' appears once a date is picked");
  check(r.some((x) => /nov|20/i.test(x)), "the chosen date is shown back", JSON.stringify(r));

  await useDate.click();
  q = await waitForQuestion(page, before);
  check(/how big/i.test(q), "confirming the date moves on", q);

  console.log("\n6. A multiple-choice question");
  r = await rows(page);
  check(r.length === 8, "seven options plus 'Something else'", String(r.length));
  check(/studio/i.test(r[0]), "first option is Studio", r[0]);
  check(/something else/i.test(r[r.length - 1]), "last row is 'Something else'");
  check((await page.getAttribute("#input", "placeholder")) === "Or reply directly…", "the box invites typing anyway",
        await page.getAttribute("#input", "placeholder"));
  await slotCards(page).locator(".qcard__row", { hasText: "2 bedrooms" }).click();
  q = await waitForQuestion(page, q);
  check(/stairs|elevator/i.test(q), "moves on to access", q);

  console.log("\n7. Skipping the optional ones");
  for (let i = 0; i < 2; i++) {
    const skip = slotCards(page).locator(".qcard__row", { hasText: "Skip" });
    check((await skip.count()) === 1, "an optional question offers Skip");
    await skip.click();
    q = await waitForQuestion(page, q);
  }
  check(/photo/i.test(q), "reaches the photo step", q);

  console.log("\n8. Photos, asked twice");
  check((await page.locator("#assist button", { hasText: "Add photos" }).count()) === 1,
        "the picker's buttons stay by the composer");
  await page.locator("#assist button", { hasText: "Skip for now" }).click();
  q = await waitForQuestion(page, q);
  check(/thirty seconds|worth/i.test(q), "skipping once asks a second time, with the reason", q.slice(0, 60));
  // The second skip ends the questions; what comes back is the read-back, and
  // that arrives in the transcript rather than the slot.
  await page.locator("#assist button", { hasText: "Skip for now" }).click();

  console.log("\n9. The read-back");
  await page.waitForFunction(() => document.querySelectorAll(".thread .recap").length === 1,
                             null, { timeout: 45000 });
  check((await slotCards(page).count()) === 0, "the question card is gone");
  const recap = page.locator(".thread .recap");
  check((await recap.count()) === 1, "the summary is in the transcript, where it can scroll");
  check(await recap.isVisible(), "and it is visible");
  // The one screen the customer is asked to CHECK must not be cut off.
  const clipped = await recap.evaluate((el) => {
    const r = el.getBoundingClientRect();
    return el.scrollHeight > el.clientHeight + 2 || r.height < 80;
  });
  check(!clipped, "the summary is not clipped");
  const summary = await recap.innerText();
  for (const value of ["Nik", "(323) 555-0142", "nik@example.com", "90026", "90401", "2 bedrooms"]) {
    check(summary.includes(value), `read-back shows ${value}`);
  }
  check((await page.locator("#assist button", { hasText: "Send it over" }).count()) === 1, "Send it over is offered");

  console.log("\n10. Sending");
  await page.locator("#assist button", { hasText: "Send it over" }).click();
  await page.waitForFunction(() => document.querySelectorAll("#slot .qcard").length === 0, null, { timeout: 45000 });
  check(true, "the card clears when the interview ends");
  const last = await page.locator(".msg.them").last().innerText();
  check(last.length > 10, "Sam confirms in the transcript", last.slice(0, 60));

  await page.screenshot({ path: __dirname + "/final.png", fullPage: true });
  await browser.close();

  console.log(fails.length ? `\nFAILED (${fails.length}):\n  - ` + fails.join("\n  - ") : "\nALL CHECKS PASSED");
  process.exit(fails.length ? 1 : 0);
})();
