const { chromium } = require("playwright");
const fails = [];
const check = (c, l, x) => { if (c) console.log("  ✓ " + l); else { console.log("  ✗ " + l + (x ? "  → " + x : "")); fails.push(l); } };
/*
 * Wait for the page to be genuinely finished.
 *
 * The thinking dots are removed on the FIRST streamed token, so they are not
 * the end of the turn — they are the start of the answer. Typing into that gap
 * is correctly ignored by the page, which looks exactly like a hang from out
 * here. The end of a turn is: no dots, no streaming caret, and the last reply
 * unchanged from one sample to the next.
 */
async function idle(p, timeout = 45000) {
  const started = Date.now();
  let last = null;
  while (Date.now() - started < timeout) {
    const now = await p.evaluate(() => {
      const busy = !!document.getElementById("dots") || !!document.querySelector(".caret");
      const msgs = document.querySelectorAll(".msg.them");
      return busy ? null : (msgs.length ? msgs[msgs.length - 1].innerText : "");
    });
    if (now !== null && now === last) return;
    last = now;
    await p.waitForTimeout(400);
  }
  throw new Error("page never went idle");
}

(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 390, height: 844 } });
  p.on("pageerror", (e) => { console.log("  ✗ PAGEERROR: " + e.message); fails.push("pageerror"); });

  console.log("\n1. An ordinary question still reads as a conversation");
  await p.goto(process.env.BASE, { waitUntil: "domcontentloaded" });
  await p.fill("#input", "how much for 2 movers?");
  await p.press("#input", "Enter");
  await idle(p);
  check((await p.locator(".msg.me").count()) === 1, "what I typed is in the transcript");
  check((await p.locator(".msg.me").innerText()).includes("2 movers"), "with my words");
  const answer = await p.locator(".msg.them").last().innerText();
  check(/115|125/.test(answer), "Sam answers with the published rate", answer.slice(0, 70));
  check(await p.locator("#slot").isHidden(), "no question card — this is not an interview");

  console.log("\n2. Then asking for an estimate starts one");
  await p.fill("#input", "ok can I get an estimate for my place");
  await p.press("#input", "Enter");
  await p.waitForFunction(() => document.querySelector("#slot .qcard__q"), null, { timeout: 45000 });
  check(true, "the card appears");
  check((await p.locator(".msg.me").count()) === 2, "the request itself is still in the transcript");

  console.log("\n3. Start over");
  await p.locator("#restart").click();
  await p.waitForLoadState("domcontentloaded");
  await p.waitForTimeout(700);
  check((await p.locator(".chip").count()) === 6, "back to the welcome screen");
  check((await p.locator(".msg").count()) === 0, "with nothing carried over");
  check(await p.locator("#slot").isHidden(), "and no card");

  await b.close();
  console.log(fails.length ? `\nFAILED (${fails.length}):\n  - ` + fails.join("\n  - ") : "\nALL CHECKS PASSED");
  process.exit(fails.length ? 1 : 0);
})();
