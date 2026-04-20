import { chromium } from 'playwright-chromium';

const OUT = '/tmp/slides-verify';
const BASE = 'http://localhost:4567';
// new slide numbering after inserting "The generative model" before Figure 2
// 1 title, 2 roadmap, 3 t-maze, 4 generative model (NEW),
// 5 figure 2 pieces, 6 A-matrix, 7 inference, 8 EFE, 9 three terms,
// 10 dirichlet, 11 precision, 12 impl choices, 13 sim, 14 fig 6, ...
const SLIDES_TO_CHECK = [4, 6, 7, 9, 10, 11, 12, 13];

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 } });
const page = await ctx.newPage();

for (const n of SLIDES_TO_CHECK) {
  await page.goto(`${BASE}/${n}?print`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1400);
  const path = `${OUT}/slide-${String(n).padStart(2, '0')}.png`;
  await page.screenshot({ path, fullPage: false });
  console.log('wrote', path);
}

await browser.close();
