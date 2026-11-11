# WeldWorld: 15-minute demo script (Ameri-Fab Energy Solutions)

**Audience:** Ameri-Fab Energy Solutions, Weatherford TX. They build ASME Section VIII pressure vessels,
B31.3 process piping and pipe spools, manifold/header skids and structural skids. Expect the
operations lead, the QA/QC manager or CWI, and possibly the owner.

**Goal of the meeting:** agree on a **6-week pilot** on one robotic or semi-automatic weld cell.

**Promise:** "Every weld you make already produces data that predicts whether it will pass RT. We capture it, make it
searchable by physics, and use NVIDIA's Cosmos world models to explain, predict and multiply it."

---

## Before the meeting (T-60 min)

| Check | Command / action |
|---|---|
| Pod up, models warm | `bash runpod/status.sh`: reasoner and generator show **healthy** |
| Showcase cache populated | `make warm-cache` ran after the last data change. The copilot answers in about 2 s from the cache |
| Copilot URL opens | `https://<podid>-7860.proxy.runpod.net` in a **fresh** browser profile, zoom 110%, dark mode off for the projector |
| Fallback ready | laptop: `make copilot-fixtures` works offline. The screen recording `docs/img/demo_fallback.mp4` (if produced) is on the desktop |
| Data story fixed | Simulator seed 7 (default), so the same joints carry the same defects every time (see the table below) |
| Second tab | MinIO console (`-9001`) logged in, for the "where does the data live" question |

**The data story (fixed seed).** All of it is synthetic. Say so up front.

| Joint / pass | What happened | RT |
|---|---|---|
| **SP-1043 J07 P3 (FCAW fill)** | shielding-gas loss → **porosity** (severity 0.8) | **reject**: the hero example |
| SP-1043 J07 P4 (cap) | undercut | visual |
| SP-1043 J06 P1 (GTAW root) | **burn-through** on the root | reject |
| SP-1043 J05 P3 | voltage excursion, minor undercut | accept |
| SP-1044 J02 P2 | **arc instability** (worn contact tip / liner) | — |
| **SP-1044 J03 P3** | **scattered porosity, within limits** | **accept**: the calibration example |
| SP-1044 J03 P4 (cap) | travel too fast → **lack of fusion** | reject |
| V-207 CS2 P3 (SAW vessel shell seam) | interpass temperature exceeded → **crack** | reject |

---

## 0:00–2:00 · Framing (no screen yet, or the title slide / dashboard idle)

Say:

> "You weld to code: Section IX procedures and qualifications, B31.3 piping, Section VIII vessels. Today a weld is
> judged **after the fact**, when RT or UT comes back. By then the spool has moved on, the welder has done 20 more
> joints, and a reject means excavation, a re-weld, a re-shoot and a schedule hit. A repair usually costs several times what the original weld cost."

Ask them (and write the numbers down; they size the pilot):
- "What's your RT reject rate on pipe and on vessel seams? Which defect shows up most: porosity, lack of fusion, slag?"
- "When a joint fails RT, how long does it take to find *why*, and can you tell from records which WPS parameters
  were actually used?"
- "How do you qualify a new welder, or a new robot program, today?"

Then point out four problems the demo addresses:
1. **QA arrives too late.** RT is a sample (for example, B31.3 Normal Fluid Service random examination) and comes after the fact.
2. **WPS compliance traceability.** The WPS gives ranges for amps, volts, travel speed, heat input and interpass. Hardly anyone can prove
   those ranges were held on *this* pass.
3. **Welder and robot qualification** is a coupon plus a bend or RT test, not a continuous record.
4. **Defect data is scarce.** Good shops make few defects, so there is too little defect footage to train any AI. Synthetic data fixes this.

---

## 2:00–4:00 · Stage 1–2: COLLECT → MOSAICO ("your welds as data")

**Click:** open the copilot. The dashboard shows the passes table (spools SP-1043, SP-1044 and vessel V-207), with RT badges
and WPS-violation counts.

Say:

> "This is a robotic pipe cell like your WC-02: A106 Gr.B pipe on a positioner, GTAW root, FCAW fill and cap.
> Every pass streams arc current and voltage at 100 Hz, wire feed, gas flow, robot joints and torch pose at 50 Hz, IR interpass
> temperature, and a torch camera. Each pass is one **sequence** in Mosaico, an open-source data platform built for
> robotics and physical AI, and it carries its spool, joint, WPS and RT result as metadata."

**Chat chip / type:** `Which passes broke the WPS this week?`
*(With `COPILOT_LLM=claude` you can be more specific: "Show me FCAW fill passes on SP-1043 that broke the WPS".)*

> "That's a query on **metadata**. Here's the one nobody can do today: a query on **physics**."

**Chat chip / type:** `Find passes where travel speed went above 7 mm/s`

**Wow #1: query by physics.** The result lists time windows inside passes, not just files.
> "Mosaico indexes the sensor values themselves, so 'show me every time gas flow dropped below 8 L/min across all cells this
> month' is one query. It ingests ROS bags and MCAP too. If your robots run ROS, or we put a small logger on the
> power source, this works on your floor."

---

## 4:00–7:00 · Root cause and the Cosmos verdict against RT (hero joint SP-1043 J07)

**Chat chip / type:** `Why did SP-1043 J07 fail RT?`

The copilot runs `wps_check`, then `assess_weld` (Cosmos 3 Reasoner), then jumps the replay to the defect.

Point at:
- **WPS findings:** gas flow out of range from about t = X s (the plot band turns red), with voltage spiking at the same time.
- **Video replay**, synchronised with the signal plots at the moment porosity starts.
- **Cosmos 3 Reasoner verdict:** *reject, porosity*, a root cause (shielding-gas loss: check the regulator, hose and
  nozzle spatter), a recommended action, and the reasoning trace.

**Wow #2: Cosmos verdict vs RT.**
> "The world model looked at the 5-second torch-camera clip and the sensor context, and called *reject, porosity*
> **before** anyone shot the joint. RT agreed. That's the shift: from 'RT found it a week later' to 'the cell flagged it
> during the pass, with the cause attached'."

**Chat chip / type:** `Show me the moment porosity started`

> "And this is your traceability record: the actual parameters, against the WPS, on every pass, for every
> spool package. Your Section IX paperwork says what *should* happen, and this shows what *did*."

---

## 7:00–8:30 · Calibration: it doesn't cry wolf (SP-1044 J03 P3)

**Chat chip / type:** `Assess SP-1044 J03 P3`

> "This pass also has porosity, but it's scattered and within acceptance limits. RT accepted it. The model says
> *accept* or *review*, not reject. We don't want a system that stops the line on every spatter. In the pilot we
> measure exactly this: how often its verdict matches your RT film."

*(If the verdict comes back `review`, say: "Review means it routes the joint to your CWI. That's the right behaviour
for a borderline case.")*

Optional, if they run SAW (needs `COPILOT_LLM=claude` or `cosmos`; the zero-key scripted router only resolves
`SP-…` references): `Why did the V-207 shell seam crack?` shows interpass over the WPS maximum on heavy wall
(thermal plot).

---

## 8:30–10:30 · What-if video (Cosmos 3 Generator)

**Chat chip / type:** `What if travel speed drops 20% on that pass?`

The copilot starts a background job, shows a progress indicator, then plays the generated clip. Live generation runs at 480p and takes
about a minute. The showcase clip comes from the warm cache.

**Wow #3: what-if video.**
> "This is a generative world model. It predicts what the weld *would look like* under a different procedure:
> slower travel, more heat input, a wider, flatter bead. Use it to train welders on 'what bad looks like' and to
> sanity-check a parameter change *before* you burn a coupon."

Be honest: "It's a visual prediction, not a metallurgical simulation. It shows the trend, and your PQR still decides."

---

## 10:30–12:30 · Sim→Real and the training-data story (Curator + Transfer)

**Chat chip / type:** `Turn this pass into photoreal footage`

**Wow #4: sim→real.** Split view: the simulator's edge or depth render on the left, Cosmos 3 Transfer's photoreal output on the right.
> "Defects are rare in a good shop, which is exactly why you can't train an AI on your own footage alone. We simulate the cell,
> inject porosity, lack of fusion and burn-through with the right sensor signatures, and Cosmos makes it look real."

**Chat chip / type:** `How many curated porosity clips do we have for training?`

> "Every clip went through NVIDIA Cosmos Curator: cut into 5-second clips, quality-filtered, captioned and embedded, and
> labelled with the sensor stats and the RT result. That's a training-ready dataset, and the same pipeline runs on
> your real footage."

---

## 12:30–15:00 · Pilot proposal and close

> "What you saw ran on synthetic data from a simulator matched to your work. The pilot answers one question:
> **does it predict your RT results on your welds?**"

**6-week pilot on one cell** (one robotic or mechanized pipe cell, or a SAW station):

| Week | Activity |
|---|---|
| 1 | Site survey; install sensors; connect the robot controller; link WPS and RT report IDs to joint IDs |
| 2–3 | Production as usual. Every pass streams into Mosaico (on-prem box or private cloud) |
| 4 | Curate; first WPS-compliance reports; first Cosmos assessments, run **blind** to RT |
| 5 | Compare against RT results as they arrive; tune prompts and thresholds with your CWI |
| 6 | Readout: prediction vs RT confusion matrix, time-to-root-cause, compliance report, go/no-go for rollout |

**Sensors to add** (all non-invasive; no change to the WPS):
- **Arc monitor**: current and voltage at 1 kHz or faster (clamp-on Hall sensor plus voltage sense leads, or the power source's
  production-monitoring data port if it has one), plus wire-feed speed.
- **Gas flow sensor** on the shielding-gas line (porosity is the most common reject, and gas loss is a common cause).
- **Torch / weld camera**: a high-dynamic-range weld camera that sees the arc and puddle.
- **IR**: a pyrometer or small thermal camera for interpass temperature (critical for heavy-wall vessel seams and impact-tested
  procedures).
- **Robot / positioner**: TCP pose and travel speed from the controller (ROS driver or vendor API).

**Success criteria to agree today:** (1) the share of RT rejects flagged before RT; (2) false-alarm rate;
(3) hours from a reject to a documented root cause; (4) a compliance report per spool package with no manual effort.

**Close:** "Can we pick the cell and a start date?"

---

## Likely questions

| Question | Answer |
|---|---|
| *Does this replace RT or the CWI?* | No. Code-required examination stays. This is in-process monitoring plus decision support that routes attention: fewer surprises at RT, faster root cause, and better records. |
| *How accurate is it?* | On this synthetic data it matches the injected labels. On *your* welds we don't know yet, and that is exactly what the pilot measures, blind against your RT film. |
| *Where does our data go?* | Wherever you want. Everything here is open source and self-hostable. Mosaico, Curator and the Cosmos models run on one GPU server on-prem or in your private cloud. The demo runs on a rented GPU pod with synthetic data only. |
| *What does it cost to run?* | This demo runs on 2× H100 (about $8/hr rented, pennies per weld). For production, one 80 GB-class GPU server per site handles QA for several cells. Curation and video generation are batch jobs. |
| *Licenses?* | Mosaico and Cosmos Curator are Apache-2.0. The Cosmos 3 models are under NVIDIA's OpenMDW-1.1 (commercial use allowed). See the README. |
| *Our robots aren't ROS / we weld manually.* | The arc monitor, gas sensor and camera work on manual stations too. Robot pose is a bonus. Mosaico takes CSV, video, ROS and MCAP. |
| *Can it check WPS compliance for welder qualification records?* | Yes. Each pass carries the welder or robot program ID, so continuity and WPQ evidence become a query. |
| *How long until it's useful?* | WPS-compliance traceability is useful in week 2. Defect prediction needs RT outcomes to compare against, so weeks 4–6. |
| *Why NVIDIA Cosmos and not a normal vision model?* | Cosmos is a *physical-AI world model*: it reasons over video and physics and can generate it. That covers QA, explanation and synthetic training data with one model family, and it can be fine-tuned on your data later. |
| *What about guardrails / safety filters?* | Cosmos ships a guardrail model that is gated. The demo runs without it. A production deployment would enable it. |

---

## Fallback plan (GPU or network trouble)

| Symptom | Do this | Say this |
|---|---|---|
| Model servers down / slow | `COSMOS_BACKEND=replay` (default when servers don't answer). Answers come from the warm cache and are labelled as cached | nothing; it looks identical |
| Pod unreachable / venue Wi-Fi blocks `*.proxy.runpod.net` | Laptop: `make copilot-fixtures` (no network, no GPU), then open `http://localhost:7860` | "I'm running a local copy, same software." |
| Live what-if takes longer than the 100-s proxy limit | The copilot runs it as a background job. If needed, use the cached showcase clip | "Full-res generation takes a few minutes; here's the one I made earlier." |
| Everything fails | Play the screen recording (`docs/img/demo_fallback.mp4`) and talk over it | — |

Rehearse the fallback once. Switching to `make copilot-fixtures` takes about 20 seconds.
