/* Exercises the real chat page with controlled streams, without model calls.
 * Checks burst pacing, approved corrections, interruptions and reduced motion.
 */
const assert = require("node:assert/strict");
const { chromium } = require("playwright");

const base = process.env.BASE || "http://127.0.0.1:8814";
const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function prepare(page, events) {
  await page.goto(base);
  await page.evaluate((events) => {
    const originalFetch = window.fetch;
    window.fetch = (url, options) => {
      if (url !== "/api/chat") return originalFetch(url, options);
      return Promise.resolve(new Response(new ReadableStream({
        async start(controller) {
          for (const item of events) {
            await new Promise((resolve) => setTimeout(resolve, item.delay || 0));
            if (item.error) {
              controller.error(new Error("Test connection lost"));
              return;
            }
            controller.enqueue(new TextEncoder().encode("data:" + JSON.stringify(item.event) + "\n\n"));
          }
          controller.close();
        },
      }), { headers: { "Content-Type": "text/event-stream" } }));
    };
  }, events);
  await page.locator("#input").fill("Tell me about my move");
  await page.locator("#send").click();
}

async function finished(page) {
  await page.waitForFunction(() => !busy);
  assert.equal(await page.locator(".caret, #dots").count(), 0, "no loading indicator remains");
}

(async () => {
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    const reply = "Our crew brings the truck, moving blankets, and everything needed to protect your furniture. We can help you pack, too.";

    await prepare(page, [
      { delay: 150, event: { event: "token", text: "Our crew " } },
      { delay: 300, event: { event: "token", text: reply.slice(9) } },
      { delay: 700, event: { event: "done", reply } },
    ]);
    await page.locator(".caret").waitFor();
    const cursor = await page.locator(".caret").elementHandle();
    await pause(400);
    assert.equal(await cursor.evaluate((node) => node.isConnected), true, "cursor persists between chunks");
    await finished(page);
    assert.equal(await page.locator(".msg.them").last().innerText(), reply);
    console.log("✓ streamed chunks retain one cursor and finish with exact text");

    const burst = (reply + "\n\n").repeat(8).trim();
    const ending = " We can help with packing as well.";
    await prepare(page, [
      { delay: 200, event: { event: "token", text: burst } },
      { delay: 600, event: { event: "token", text: ending } },
      { delay: 500, event: { event: "done", reply: burst + ending } },
    ]);
    await page.evaluate(() => {
      window.visibleLengths = [];
      function sample() {
        const reply = document.querySelector(".msg.them:not(.thinking)");
        window.visibleLengths.push(reply ? reply.textContent.length : 0);
        if (busy) requestAnimationFrame(sample);
      }
      requestAnimationFrame(sample);
    });
    await page.waitForFunction((length) => {
      const reply = document.querySelector(".msg.them:not(.thinking)");
      return reply && reply.textContent.length >= length - 30;
    }, burst.length);
    await pause(100);
    await page.locator("#thread").evaluate((node) => { node.scrollTop = 0; });
    await finished(page);
    const lengths = await page.evaluate(() => window.visibleLengths);
    assert.ok(new Set(lengths.filter((length) => length > 0 && length < burst.length)).size >= 2,
      "a network burst appears across multiple frames instead of one jump");
    assert.equal(await page.locator(".msg.them").last().innerText(), burst + ending);
    assert.equal(await page.locator("#thread").evaluate((node) => node.scrollTop), 0,
      "new words do not pull the customer back down after they scroll up");
    console.log("✓ a large network burst is paced across frames without losing text");
    console.log("✓ scrolling up keeps the customer's place while the answer finishes");

    const corrected = "A manager will confirm the exact price for your move.";
    await prepare(page, [
      { event: { event: "token", text: "Unapproved draft ".repeat(100) } },
      { delay: 50, event: { event: "done", reply: corrected } },
    ]);
    await finished(page);
    await pause(400);
    assert.equal(await page.locator(".msg.them").last().innerText(), corrected);
    console.log("✓ approved correction cancels all queued draft text");

    await prepare(page, [
      { event: { event: "token", text: "Draft that should disappear" } },
      { delay: 50, event: { event: "ask", ask: {
        type: "question", step: 1, total: 10,
        field: { name: "name", kind: "text", question: "What's your name?", required: true },
      } } },
    ]);
    await finished(page);
    assert.equal(await page.locator(".msg.them").count(), 0);
    assert.equal(await page.locator("#slot .qcard").count(), 1);
    console.log("✓ interview transition removes any streamed draft");

    await prepare(page, [
      { event: { event: "token", text: "A reply that gets interrupted" } },
      { delay: 100, error: true },
    ]);
    await finished(page);
    assert.match(await page.locator(".msg.them").last().innerText(), /connection|try again/i);
    console.log("✓ interrupted stream stops its animation and offers a retry");

    await page.emulateMedia({ reducedMotion: "reduce" });
    await prepare(page, [
      { delay: 100, event: { event: "token", text: reply } },
      { delay: 500, event: { event: "done", reply } },
    ]);
    await page.locator(".caret").waitFor();
    assert.equal(await page.locator(".msg.them").last().innerText(), reply, "reduced motion displays chunks immediately");
    await finished(page);
    assert.deepEqual(errors, []);
    console.log("✓ reduced motion skips pacing; no browser errors");
  } finally {
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
