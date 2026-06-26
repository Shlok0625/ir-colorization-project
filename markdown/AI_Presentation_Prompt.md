# MASTER PROMPT — Paste this entire block into the AI presentation tool

You are creating a 9-slide pitch deck for a **national-level hackathon idea-submission round** (ISRO Bharatiya Antariksh Hackathon 2026). The panel is technical and will shortlist based on engineering credibility, not visual flash. Every slide must read as confident and complete, but nothing on it may overstate what is actually true — vague AI-generated filler ("leveraging cutting-edge AI for seamless results") will actively hurt this team's chances with this specific panel. Follow the structure, content, and visual instructions below exactly. Do not invent facts, numbers, or claims beyond what is given here.

## Tone & Design Rules (apply to every slide)
- Professional, technical, restrained — no cartoon clip-art, no generic "rocket ship" startup graphics.
- Color palette: deep navy / space-blue background tones with white/light text, one accent color (amber or cyan) for highlights — evokes ISRO/aerospace without being kitschy.
- Use real diagrams (flowcharts, pipeline boxes, comparison tables) rather than decorative stock photography. Where a photo is appropriate (satellite imagery, IR/thermal sample images), use realistic Landsat-style imagery, not generic AI-art.
- Keep text density low per slide — bullet points, not paragraphs. The supporting detail in this prompt is for you to distill, not paste verbatim.
- Every slide must look distinct from every other slide in layout — do not reuse one template for all 9 slides.

---

## SLIDE 1 — Title & Basics
**Layout**: Full bleed dark space/Earth-observation themed background (satellite over Earth at night, or an actual thermal-to-RGB transition image as a hero visual). Centered title block.

Content:
- Team Name: **[TEAM NAME TBD — insert placeholder text "Team Name" in editable text box, do not invent a name]**
- Problem Statement: **Infrared Image Colorization & Enhancement for Improved Object Interpretation**
- Team Leader: **Vighnajit CHM**

**Visual required**: A hero image showing a side-by-side or split-screen of a grayscale/thermal infrared satellite image transitioning into a realistic colorized RGB version of the same scene. This single image is the entire pitch in one glance — prioritize getting this visual right.

---

## SLIDE 2 — Team Details
**Layout**: Clean 2x2 or vertical card grid, one card per person, each with name/college/role icon.

Content (use exactly this):
| Name | College | Role |
|---|---|---|
| Vighnajit CHM (Team Leader) | The National Institute of Engineering, Mysuru | CV/ML — model architecture & training |
| Shlok Vardhan | The National Institute of Engineering, Mysuru | Evaluation, Documentation & Process |
| Tanush Malode | The National Institute of Engineering, Mysuru | Data Pipeline & Classical CV Baselines |
| Sai Naman | [College name placeholder — leave editable, do not invent] | Web Development |

**Visual**: simple role-icon per card (satellite/code/chart/browser icon) — keep minimal, no headshots needed.

---

## SLIDE 3 — The Opportunity (USP) — THIS IS THE MOST IMPORTANT SLIDE
**Layout**: Two-column "Problem vs Our Approach" comparison, or a 3-point differentiation list with icons. This slide must land the panel's "why these people, why this solution" judgment in under 60 seconds of reading.

State the USP using these three specific, defensible differentiators (do not weaken them into generic claims):

1. **Real paired ground truth, not assumed colorization.** Most IR-colorization research has no real color ground truth to supervise against — it relies on plausibility, not accuracy. This project uses Landsat 9's genuinely co-registered Band 10 (thermal) and Bands 2/3/4 (true RGB) from the same overpass, same scene — so predicted color is supervised directly against real measured color, pixel-by-pixel. This removes the central ambiguity that limits most colorization work.

2. **Physics-constrained, not just visually plausible.** A 7-term loss function (reconstruction + structural + adversarial + perceptual + NDVI consistency + NDWI consistency + independent semantic-segmentation consistency) is explicitly weighted to prevent the generator from producing color that merely "looks real" while contradicting the underlying physical signal — directly targeting the hallucination risk that generic GAN-colorization approaches don't address.

3. **Engineering honesty as a design principle.** Every architectural choice (e.g., whether unpaired pretraining is even used) is gated behind a named, real, available dataset requirement and validated through an explicit ablation study — not included for technical buzzword density. This is a system designed to survive scrutiny, including its own.

Closing line for this slide: *"We are not claiming to remove hallucination — we are claiming to measure, bound, and report it, with a real physical ground truth to check against."*

**Visual required**: simple comparison graphic — left side "Typical IR Colorization Approaches" (no ground truth, plausibility-driven, single metric) vs right side "Our Approach" (real RGB ground truth, 7-term physics loss, multi-metric + downstream validation).

---

## SLIDE 4 — Features
**Layout**: Icon + short-label grid (4–6 features), visual-forward as instructed by the template.

List these features (each as a short label + one-line description):
- **Native-Resolution Recovery** — recovers genuine 100m thermal detail from the over-resampled 30m delivered product, rather than treating the delivered product as ground truth.
- **ESRGAN Sharpening** — recovers faint edges/textures lost to low native thermal resolution using RRDB blocks + perceptual/edge losses.
- **Physics-Constrained Colorization** — Pix2Pix-based RGB generation supervised against real Landsat color, constrained by an independent land-cover semantic signal (SegFormer) so colors are not hallucinated.
- **Automated Physics Verification Gate** — every output tile is checked for NDVI/NDWI consistency before being finalized; flagged tiles get a bounded, auditable correction pass.
- **Downstream Task Validation** — measurable object-detection improvement (YOLOv8 mAP) on colorized output vs raw IR, not just "looks better."
- **Lightweight Web Viewer** — side-by-side raw IR / enhanced / colorized / ground-truth comparison with live per-tile quality metrics.

**Visual required**: small sketch/icon per feature (as the template recommends) — e.g., a simple 4-panel pipeline sketch showing IR → sharpened IR → colorized → verified output.

---

## SLIDE 5 — Flow (Process / Use-Case Diagram)
**Layout**: Horizontal left-to-right pipeline flow diagram — this should be a clean, simplified version of the technical pipeline (full technical depth goes on Slide 7's architecture diagram instead).

Render this flow as boxes connected by arrows:
`Landsat 9 IR + RGB Input` → `Native Resolution Recovery (30m→100m)` → `ESRGAN Super-Resolution` → `Pix2Pix Physics-Constrained Colorization` → `Physics Verification Gate (NDVI/NDWI check)` → `Final RGB GeoTIFF Output` → `Evaluation (Image Quality + Downstream Detection)`

Add a small annotation under the diagram: *"Each stage is independently validated against a non-DL baseline before being trusted in the next stage."*

**Visual required**: this entire slide IS the diagram — make it the dominant visual element, minimal surrounding text.

---

## SLIDE 6 — Wireframes (Optional but include — strengthens shortlisting odds)
**Layout**: Browser-window mockup frame showing the web viewer UI.

Content to depict inside the mockup:
- A tile selector/dropdown on the left
- Four side-by-side panels: "Raw IR" / "SR-Enhanced IR" / "Colorized Output" / "Real RGB Ground Truth"
- A metrics strip below the panels showing live PSNR / SSIM / LPIPS values for the selected tile
- Caption: *"Deliberately minimal by design — frontend investment is secondary to model and evaluation rigor, which is what this problem statement is actually graded on."*

**Visual required**: a clean low-fidelity browser/app wireframe mockup, not a finished UI — this slide is explicitly meant to look like a wireframe, per the template's own instruction.

---

## SLIDE 7 — Architecture (Full Technical Diagram)
**Layout**: Dense technical block diagram — this is where full rigor is expected and rewarded by a technical panel. Use this exact structure (reproduce as labeled boxes/arrows, not prose):

```
[Landsat 9: B2,B3,B4,B10] → [Co-registration + RGB Composite, 30m]
[Landsat 9: B10] → [Native 100m TIR Recovery] → [Synthetic Degradation: 100m→200m, TRAINING ONLY]
[Synthetic 200m] → [ESRGAN SR Module: RRDB + PixelShuffle] → [SR Output: 200m→100m TIR]

(Conditional, gated by named dataset) [Unpaired IR/RGB Corpus] → [CycleGAN Pretrain] → [Pix2Pix Generator Init]

[SR Output 100m TIR] + [RGB Composite 30m] + [Pix2Pix Generator Init] 
   → [Pix2Pix Fine-Tune: paired 100m TIR → RGB]
   → [7-Term Loss: L1 + SSIM + Adversarial + LPIPS-Perceptual + NDVI + NDWI + SegFormer-Consistency]
   → [SegFormer Auxiliary Semantic Channel feeds loss]
   → [Physics Verification: NDVI/NDWI Check + Bounded Color-Correction Loop]
   → [Final RGB GeoTIFF Output, RGB channel order]

[Final Output] → [Evaluation: PSNR / SSIM / FID / LPIPS]
[Final Output] → [YOLOv8 Downstream Detection mAP]
[Final Output] → [Baselines: Bicubic SR, Classical Colorization]
→ [Lightweight Web Viewer]
```

Add a small side-note box on this slide: *"CycleGAN stage is conditional — included only if a real, named unpaired dataset is confirmed; otherwise this path is cut, not faked."* This honesty signal is a deliberate credibility point for a technical judge — keep it visible, don't hide it in small print.

**Visual required**: this should look like a genuine system architecture diagram (boxes, arrows, grouped stages with stage labels/numbers) — not a flowery infographic. Technical panels reward precision here.

---

## SLIDE 8 — Tech Stack
**Layout**: Grouped logo/label grid by category (not just a flat list).

- **Data Source**: Landsat 9 Collection 2 (Google Earth Engine API)
- **Super-Resolution**: ESRGAN (RRDB + PixelShuffle), PyTorch
- **Colorization**: Pix2Pix (U-Net generator + PatchGAN discriminator), optional CycleGAN pretrain
- **Semantic Constraint**: SegFormer (land-cover segmentation)
- **Downstream Validation**: YOLOv8 (object detection)
- **Evaluation**: PSNR, SSIM, LPIPS, FID
- **Backend**: Python, Flask API
- **Frontend**: HTML/CSS/JavaScript (lightweight viewer)
- **Geospatial Tooling**: Rasterio / GDAL for GeoTIFF I/O

**Visual required**: technology logos (PyTorch, Flask, Google Earth Engine, OpenCV/Rasterio) arranged in a clean grouped grid by pipeline stage — avoids the slide looking like a random buzzword list.

---

## SLIDE 9 — Budget (Estimated Implementation Cost)
**Layout**: Simple cost breakdown table + one summary callout number.

Content (use these figures — they are sourced from real 2026 GPU cloud market rates, not invented):
| Item | Estimate |
|---|---|
| Data acquisition (Google Earth Engine) | $0 (free tier, academic/research use) |
| GPU compute — ESRGAN training (~15–20 hrs) | ~$25–40 |
| GPU compute — Pix2Pix fine-tuning (~20–25 hrs) | ~$35–50 |
| GPU compute — CycleGAN pretrain, if used (~10–15 hrs) | ~$15–25 |
| GPU compute — SegFormer + YOLOv8 fine-tuning (~10 hrs) | ~$15–20 |
| Experimentation/rerun buffer (~1.5x) | ~$45–65 |
| **Estimated total compute cost (A100-class GPU, ~$1.5–2/hr cloud rate)** | **~$135–200 (≈ ₹11,000–₹17,000)** |
| Web hosting/demo (free-tier) | $0 |

Footnote on slide: *"Estimate based on on-demand A100 80GB cloud GPU rates (~$1.5–2/hr, 2026 market range); actual cost will vary by provider and training iteration count."*

**Visual required**: a horizontal bar breakdown chart showing the cost split by pipeline stage, with the total as a single prominent number.

---

## FINAL INSTRUCTION TO THE AI TOOL
Do not add a 10th slide, do not add a generic "Thank You / Questions" closing slide beyond what the template specifies, and do not soften or remove the conditional/honest framing on Slides 3, 5, 7, and 9 (the CycleGAN gating, the "directional not precise" framing implicit in the metrics, the real-cost-not-invented framing on budget) — these honesty markers are a deliberate credibility strategy for a technical judging panel, not hedging to be cleaned up.
