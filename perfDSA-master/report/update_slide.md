# Update slide (26 Oct)

## Slide 1 — AIF QC, SQ1 and SQ2: what's new

**QC & features**
- 408 AIFs labelled → trust flag (trusted / review / reject = 321 / 17 / 70); only 9/61 failures are segmentation errors
- Truncation: literature-based threshold (de la Rosa 2023, 10 % bound) → tail ≤ 0.15 instead of 0.30 *(to decide)*

**SQ1 – sampling segment (full ICA vs top)**
- Model-based full ICA ≈ top (Dice 0.87, ±2 %)
- Grown full ICA: AIF peak +7 % (frontal) vs −11 % (lateral); lateral CBV +32 %
- Driver: vessel overlap; top AIF is view-consistent → **keep supraclinoid ICA (perfDSA)**

**SQ2 – projection view (A vs B, same bolus, n = 43)**
- Lateral AIF ×1.32, explained by exposure (8 kV lower; whole image also ×1.32; AIF/whole image = 1.01)
- Not path length, not saturation; shape-only features agree (curve mismatch −63 % after normalisation)
- But perfusion maps: **CBV ×0.42, MTT ×0.60** in lateral → views not interchangeable

**Next:** decide on the 0.15 threshold → re-run · tissue curves per view (why does CBV drop?) · write Results

## Slide 2 — figure

`report/figures/update_slide.png`
Caption: (left) Grown vs top: direction flips between views. (middle) Same bolus, lateral AIF higher. (right) AIF difference = global exposure; CBV/MTT differ even more.
