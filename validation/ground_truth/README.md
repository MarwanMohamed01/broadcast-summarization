# Ground Truth Annotation Instructions

## What you're doing

You are creating the "gold standard" list of news headlines that actually
appeared in two 30-minute slices of the 14-hour Al Jazeera video. This
ground truth is then compared against what our pipeline extracted, to
compute Precision / Recall / F1 / CER / WER metrics for the thesis.

## The two slices

- **Slice A** = chunk 17 of the 14-hour video (hours ~8:30 – 9:00)
- **Slice B** = chunk 26 of the 14-hour video (hours ~13:00 – 13:30)

Two slices from different time periods tests whether the pipeline works
consistently over time — a stronger claim than "works on one sample."

## Files

- `slice_A_panorama.png` — a very wide PNG image showing every scrolled
  ticker frame from slice A stitched together end-to-end. Open in any
  image viewer and scroll horizontally to read.
- `slice_B_panorama.png` — same for slice B.
- `slice_A_headlines.txt` — where YOU type the ground-truth headlines
  you see in panorama A (empty file waiting for your input).
- `slice_B_headlines.txt` — same for slice B.

## How to annotate

1. Open `slice_A_panorama.png` in an image viewer that supports
   horizontal scrolling (Windows Photos, IrfanView, any browser).
2. Scroll from left to right, reading each headline as it appears.
3. Each distinct headline is separated from the next by a small `" - "`
   delimiter. You'll see the same headlines repeating in cycles — only
   write each UNIQUE headline ONCE.
4. Type each headline into `slice_A_headlines.txt`, one per line, in
   UPPERCASE. Example:

   ```
   GAZA HEALTH MINISTRY: 72,292 PALESTINIANS KILLED IN ISRAELI ATTACKS SINCE OCTOBER 2023
   ISRAELI FORCES AND SETTLERS HAVE KILLED 1,138 PALESTINIANS IN OCCUPIED WEST BANK SINCE OCTOBER 2023
   D.R. CONGO TO TEMPORARILY ACCEPT DEPORTED MIGRANTS UNDER NEW DEAL WITH THE U.S.
   ```

5. No numbering. No quotes. No bullet points. Just one headline per line.
6. Lines starting with `#` are treated as comments (ignore them).
7. Don't worry about minor OCR-level typos in the panorama — write what
   you can read clearly. If a word is unreadable, use `???` or skip
   the word.
8. When done with slice A, repeat for slice B.

## Tips

- Expect 25–40 unique headlines per slice.
- The ticker content may differ between slices if breaking news rotated
  in/out during the day — don't assume they're the same.
- You can save and continue later. The file is just a plain text file.
- If two headlines look nearly identical but with different numbers
  (e.g., "72,292 killed" vs "716 killed since ceasefire"), they are
  DIFFERENT headlines — list both.

## When you're done

Tell Claude "ground truth is ready" and it will run
`validation/validate_extraction.py` to compute the metrics.
