// Покадровый рендер промо-ролика в MP4.
//   NODE_PATH=$(npm root -g) node render.js                 -> promo.mp4 (1080x1920, 30fps)
//   node render.js --stills 1,6,10,16,18,24,30             -> PNG-кадры для проверки
//   node render.js --bot @my_bot --brandA MYBRAND --brandB AI --model "MYBRAND CORE"
// ffmpeg: берётся из $FFMPEG, иначе из PATH.
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const path = require('path');

const args = process.argv.slice(2);
const opt = k => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : null; };

(async () => {
  const q = new URLSearchParams({ render: '1' });
  for (const k of ['bot', 'brandA', 'brandB', 'model']) if (opt(k)) q.set(k, opt(k));
  const url = 'file://' + path.join(__dirname, 'index.html') + '?' + q;
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
  await page.goto(url);
  await page.waitForFunction(() => window.__ready === true);
  const { DURATION, FPS } = await page.evaluate(() => ({ DURATION: window.DURATION, FPS: window.FPS }));

  const stills = opt('stills');
  if (stills) {
    for (const t of stills.split(',').map(Number)) {
      await page.evaluate(t => window.renderAt(t), t);
      await page.screenshot({ path: path.join(opt('dir') || __dirname, `still_${String(t).replace('.', '_')}.png`) });
    }
    return browser.close();
  }

  const out = opt('out') || path.join(__dirname, 'promo.mp4');
  const ff = spawn(process.env.FFMPEG || 'ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-i', '-',
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], { stdio: ['pipe', 'inherit', 'inherit'] });
  const total = Math.round(DURATION * FPS);
  for (let f = 0; f < total; f++) {
    await page.evaluate(t => window.renderAt(t), f / FPS);
    const buf = await page.screenshot({ type: 'jpeg', quality: 95 });
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
    if (f % 90 === 0) process.stdout.write(`frame ${f}/${total}\n`);
  }
  ff.stdin.end();
  await new Promise(r => ff.on('close', r));
  await browser.close();
  console.log('done ->', out);
})();
