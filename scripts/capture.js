// capture.js
const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer");

async function main() {
  const pagesDir = path.join(__dirname, "../slides");
  const outDir = path.join(__dirname, "../screenshots");

  // Ensure output dir exists
  await fs.promises.mkdir(outDir, { recursive: true });

  // Get all .html files in pagesDir
  const files = await fs.promises.readdir(pagesDir);
  const htmlFiles = files.filter((f) => f.toLowerCase().endsWith(".html"));

  if (htmlFiles.length === 0) {
    console.log("No .html files found in", pagesDir);
    return;
  }

  const browser = await puppeteer.launch({
    headless: "new", // or true, depending on your Puppeteer version
  });
  const page = await browser.newPage();

  // Optional: default viewport; fullPage: true will expand height as needed
  await page.setViewport({ width: 1920, height: 1080 });

  for (const file of htmlFiles) {
    const filePath = path.join(pagesDir, file);

    // Build file:// URL (normalize for Windows slashes)
    const fileUrl = "file://" + filePath.replace(/\\/g, "/");

    console.log("Capturing", fileUrl);

    await page.goto(fileUrl, {
      waitUntil: "networkidle0", // wait for network to be idle
      timeout: 0,
    });

    // If you need JS-heavy pages to finish animations, you can add:
    // await page.waitForTimeout(1000);

    const baseName = path.basename(file, path.extname(file));
    const outPath = path.join(outDir, baseName + ".png");

    await page.screenshot({
      path: outPath,
      fullPage: true, // capture whole scrollable page
    });

    console.log("Saved ->", outPath);
  }

  await browser.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});