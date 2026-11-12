# WeldWorld — 15-minute demo script (Ameri-Fab Energy Solutions)

Runs on **real, public welding data** (both CC BY 4.0, credited in the app under *Data & credits*):
- **AIMEN SAW** — 5 coupons × 24 passes of two-wire submerged-arc welding on **52 mm SA-516 Gr.70N** pressure-vessel
  plate, with **real RT/UT inspection results** (243 indications in 33 passes) and bead photos. Closest to your vessel shop.
- **BAM tandem-GMAW** — 6-layer multi-pass welds on **30 mm S355** with **real process-camera video** synced to
  10 kHz current/voltage, gas flow, thermocouples and robot path. Not NDT-inspected.

Every number below is what the copilot answers on this data (scripted chat, replay mode). With Claude as the chat
model the wording varies; the numbers don't.

## Before the meeting (T-60 min)
- **GPU path:** `python runpod/launch.py --mode serve` (Runpod, ~10 min when the volume is warm). Open the copilot URL it
  prints. Run the six demo chips once so Cosmos 3 results are cached.
- **Laptop path (no GPU / backup):** `python runpod/download.py demo`, then `python scripts/run_copilot.py`.
  Cosmos 3 answers then come from the cache of the last GPU run; anything not cached is labelled *rule-based* or
  *placeholder*.
- Open **How it works** once — it is your agenda slide.

## 0:00–2:00 · Framing
> "You weld SA-516 vessels and pipe spools to ASME IX and B31.3. When RT rejects a weld you find out days later, you
> repair it, and you rarely learn *why*. Today: real vessel-plate welds, with the real X-ray results, flowing through an
> AI pipeline — and an assistant you can ask questions in plain English."

Show **How it works**: real weld data → Mosaico (store + query by physics) → NVIDIA Cosmos Curator (training-ready
data) → NVIDIA Cosmos 3 (inspect, what-if, new variants) → Copilot.

## 2:00–4:30 · The inspection ground truth — chip ①
**Ask:** `Which SA-516 vessel passes had NDT indications?`
**Answer:** *33 of 120 inspected passes have NDT indications: porosity ×195, metallic inclusion ×31, slag inclusion ×9,
lack of fusion ×4, crack ×4.* Shop-floor list: red NDT badges per bead.
> "This is a real radiography and UT campaign by AIMEN, on your grade of plate. Every indication is located along the
> seam, and we mapped each one to the exact second of the weld recording."

## 4:30–7:00 · The honest moment — chip ②
**Ask:** `Show porosity on PP5 bead 2 — did the arc signals change?`
The replay jumps to 11.6 s; the NDT strip under the plots marks the 14 porosity windows.
**Answer:** *Did the arc signals change? No. Inside the 14 indication windows the lead current averages 535 A vs 533 A in
the clean weld, and no signal leaves its normal band more often than chance.*
> "Most vendors would show you a red spike here. Real porosity on this coupon left **no electrical signature**. A
> current/voltage monitor alone would have missed it. That's why the pipeline joins the arc data with inspection
> results and vision — and why the assistant is built to say *'the data doesn't show a cause'* instead of inventing one."

Across all 120 passes: only 3 of 26 passes with indications show any signal change inside the windows (*How it works →
What the real data says*).

## 7:00–9:00 · Procedure compliance — chip ④
**Ask:** `Which passes ran outside their published settings?`
**Answer:** 31 SAW passes (mostly 1–4 s current excursions) and 0 BAM layers. The standout: **PP3 bead 2** ran at
**~363–433 A against a published 550 A for the whole pass** (and voltage dipped at the start).
> "Every pass is checked automatically against its settings, not just the ones that go to X-ray. Note what we *don't*
> check: AIMEN's wire-speed channel reads 7–35 % below the command for whole passes — a logger issue, so it's shown
> but not judged. Arc start and stop ramps are excluded."
Be clear: the windows are the published nominal settings ±10 %, a demo assumption — your qualified WPS ranges would
replace them.

## 9:00–10:30 · Tandem process detail — chip ③
**Ask:** `Compare lead vs trail current on WID617 layer 3`
**Answer:** lead 391 A / trail 353 A (−10 %), voltages 33.7 / 33.9 V, wire feed 20 / 15 m/min. Play the **real BAM
process video**, synced to the plots.
> "Two wires, two power sources, one robot, all on one clock. In Mosaico this is one sequence you can query by value."

## 10:30–12:30 · Cosmos 3 — chips ⑤ and ⑥
- `Inspect the bead photo of PP5 bead 3` → bead photo + Cosmos 3 Reasoner verdict (or the *rule-based* stand-in if not
  cached). Point out: *RT/UT indications are internal — usually not visible on the bead surface*; the Reasoner reads the
  photo **and** the NDT record and says so.
- `What if travel speed dropped 20 % on WID630 layer 4?` → Cosmos 3 Generator video from a real BAM frame. Say:
  *"It's a generated visual prediction, not a measurement — your PQR still decides."*
- **Generate** tab → *Transfer*: the edge map of the real BAM video → new realistic variants with the same timing, so
  the labels carry over. That's how rare defects become enough training data.

## 12:30–15:00 · Pilot proposal
> "Everything you saw is public research data. A pilot puts **your** welds through the same pipeline."
- **Weeks 1–2:** export arc data from one robotic cell (Lincoln/Miller/Fronius power-source logs), add a torch camera,
  pull your RT reports for the same joints.
- **Weeks 3–4:** Mosaico + copilot on your data; WPS windows from your qualified procedures.
- **Weeks 5–6:** Cosmos 3 Reasoner checks on your footage; measure agreement with your RT results; decide.

## Likely questions
- **"Did the AI predict the X-ray result?"** — Not from current/voltage on this data, and we show that honestly. The value
  today is traceability, automatic procedure checks and fast root-cause *questions*; prediction needs your camera data.
- **"Is the data real?"** — Yes: AIMEN (Spain) and BAM (Germany) research datasets, CC BY 4.0. *Data & credits* lists
  authors, DOIs and what we changed (resampled to 100 Hz, time-aligned, photos resized).
- **"Is ±10 % our WPS?"** — No, a demo assumption from the published nominal settings; yours replace it.
- **"Can it run on-prem?"** — Yes: Mosaico, Curator and Cosmos 3 are self-hostable; the copilot runs on a laptop in replay.

## Fallback (GPU or network trouble)
Laptop path above. Everything that came from Cosmos 3 is labelled with its source (`cosmos3-local` / cached); anything
else says *rule-based* or *placeholder* — never claim those as Cosmos output.
