// Screenshot the Weld Copilot UI at 1440x900 (used for docs/img/copilot_*.png).
//   COPILOT_FIXTURES=1 uvicorn copilot.app:app --port 7860 &
//   node copilot/scripts/screenshots.cjs [http://localhost:7860] [docs/img]
// Needs Playwright (require('playwright')); uses system Chrome when present (H.264 playback).
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

const BASE = process.argv[2] || "http://localhost:7860";
const OUT = process.argv[3] || path.join(__dirname, "..", "..", "docs", "img");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const exe = ["/usr/bin/google-chrome", "/usr/bin/google-chrome-stable"].find((p) => fs.existsSync(p));
  const browser = await chromium.launch({ executablePath: exe, args: ["--autoplay-policy=no-user-gesture-required"] });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  page.on("console", (m) => { if (m.type() === "error") console.log("console:", m.text()); });
  page.on("pageerror", (e) => console.log("pageerror:", e.message));
  const shot = async (name) => { await page.screenshot({ path: path.join(OUT, `copilot_${name}.png`) }); console.log("saved", name); };
  const waitIdle = async () => { await page.waitForFunction(() => !S.busy, null, { timeout: 120000 }); await sleep(400); };
  const ask = async (i) => { await page.click(`#suggest .sugg:nth-child(${i})`); await sleep(300); await waitIdle(); };
  const waitJobs = async () => page.waitForFunction(() => document.querySelectorAll("#messages .job-b").length === 0, null, { timeout: 300000 });

  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.waitForSelector(".chart-row");
  await sleep(1200);
  await shot("start");

  await ask(1);
  await ask(2);
  await page.evaluate(() => seek(12.6));
  await sleep(1200);
  await shot("why_failed");

  await ask(3);
  await ask(4);
  await ask(5);
  await waitJobs();
  await sleep(1500);
  await page.evaluate(() => { const m = document.querySelector("#messages"); m.scrollTop = m.scrollHeight; });
  await shot("whatif_sim2real");

  await page.click('#viewToggle [data-view="sim2real"]');
  await sleep(2500);
  await shot("sim2real_view");

  await ask(6);
  await page.evaluate(() => { setView("torch"); seek(13.4); });
  await sleep(1000);
  await shot("dataset_chat");

  await page.click('.tab-btn[data-tab="dataset"]');
  await page.waitForSelector("#dsTable tbody tr");
  await sleep(1500);
  await shot("dataset");

  await page.click('.tab-btn[data-tab="generate"]');
  await page.waitForSelector("#genPrompt");
  await sleep(1500);
  await shot("generate");

  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
