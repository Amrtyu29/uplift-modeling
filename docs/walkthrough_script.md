# Loom walkthrough — shot list

Target: **3 minutes**. Recruiters and hiring managers watch these when they will
not read code, so the goal is not a code tour. It is one idea, one validation,
one decision.

Every number below is quoted from `reports/*.json` in this repo. If you re-run
anything, re-read them before recording rather than trusting this file.

**Setup before you hit record**
- `make dashboard` running at localhost:8501, Overview tab open
- README open in a second tab, scrolled to the break-even chart
- Terminal with `scripts/07_experiment_design.py` output already printed
- Close Slack/mail. Loom captures notifications.

---

## 0:00–0:25 — The problem, with a number

> "Most targeting models predict who is likely to convert. That ranks your most
> loyal customers first — people who were going to buy anyway, so the budget
> buys nothing."

Show the zip-code table in the README.

> "You can see it in the raw data before any modelling. Rural customers have the
> highest response rate, 20.5%, and the lowest actual uplift, just over 5 points.
> Urban is the reverse — lowest response, highest uplift. A response model spends
> on rural first, and it is the worst group to spend on."

**Why this opening:** it proves the premise with observed data in 25 seconds,
before asking anyone to trust a model.

---

## 0:25–1:05 — What the model estimates, and how it is validated

> "So instead of P(convert), this estimates the causal effect of contacting each
> customer. Both datasets are randomized experiments, so treated and control are
> comparable by construction."

Switch to the dashboard Overview tab, Qini curve.

> "You cannot score this with AUC — the thing being predicted is never observed
> for any individual. There is no label. What you can check is the ranking: take
> the customers the model rates highest, and measure whether they actually show
> a bigger treated-versus-control gap on held-out randomized data. That is the
> Qini curve."

> "Best model gets 0.188 on Hillstrom and 0.733 on Criteo's full 14 million rows.
> A conventional response model scores 0.076 — with a confidence interval that
> includes zero. As a way of finding incremental effect it is not distinguishable
> from random targeting."

**Do not** explain S/T/X-learners here. Nobody watching a 3-minute video needs
the taxonomy; it is in the README if they want it.

---

## 1:05–2:00 — The business result, including the part that did not work

This is the section that separates the video from every other portfolio demo.
Do not skip the negative finding — it is the most credible thing you will say.

> "Then I priced it, and the first result was negative. Email costs a fraction of
> a cent, and at that price contacting everyone is close to optimal. The model's
> profit advantage is 2.1%, and the bootstrap interval on it contains the
> treat-everyone number. So I am not claiming a revenue lift on email."

Show the break-even chart.

> "Rather than pick a flattering cost assumption, I swept the contact cost. The
> break-even is 74 cents. Below that, blanket targeting wins and no model changes
> it. Above it, targeting the top 10% is worth 168% more profit. On Criteo, past
> 27 cents a blanket campaign actually loses money while the targeted one still
> earns $600,000 on the holdout."

> "That is the deliverable — not a lift number, a decision rule keyed to what the
> channel actually costs."

---

## 2:00–2:40 — The two things people do not build

Pick **one** of these depending on the role. Do not do both; you will run long.

**For an experimentation / product DS role:**
> "I also sized the experiment that would validate this. Powering it on response
> rate is a trap — withholding contact necessarily lowers response, so the test
> is built to conclude that targeting hurts. Powered on profit, validating on
> email needs 130 million customers. At $2 a contact it needs 3,200. So the
> recommendation is to validate on an expensive channel, not the cheap one."

**For an MLOps / platform role:**
> "And it is monitored for the failure mode specific to uplift models — the
> effect decaying while the features and conversion rates look completely normal.
> A feature-drift monitor sees nothing. This tracks the distribution of predicted
> treatment effect, and catches a 45% decay at PSI 2.9."

---

## 2:40–3:00 — Close

> "It runs end to end with one make command, serves through FastAPI, and every
> number in the README is committed as JSON in the repo so you can check any
> claim without re-running a thing. Code is linked below."

---

## Notes

- **Use the Loom share link, not the embed URL.** GitHub strips iframes; an
  embedded player will not render in a README. A plain link is the only option.
- Do not read this script. Talk from the beats — a recording that sounds read is
  worse than one with an "um" in it.
- If you fluff a section, restart that section rather than the whole video.
  Loom's trim is per-segment and this is structured to make that easy.
- One take at 3 minutes beats four takes at 90 seconds.

## Numbers cheat-sheet

| Claim | Value | Source |
|---|---|---|
| Rural response / uplift | 0.2050 / 0.0515 | `hillstrom_exploration.json` |
| Urban response / uplift | 0.1601 / 0.0633 | `hillstrom_exploration.json` |
| Best Qini, Hillstrom | 0.188 (S-learner) | `hillstrom_metrics.json` |
| Best Qini, Criteo 14M | 0.733 (X-learner) | `criteo_metrics.json` |
| Response model Qini | 0.076, CI [-0.014, 0.177] | `hillstrom_metrics.json` |
| Profit advantage | +2.1%, CI [$1,059, $15,764] | `hillstrom_simulation.json` |
| Break-even, Hillstrom | $0.74/contact → +168% | `hillstrom_simulation.json` |
| Criteo past $0.27 | blanket −$54,222 vs targeted $600,278 | `criteo_simulation.json` |
| A/B test, email | 130,047,154 customers | `hillstrom_experiment_design.json` |
| A/B test, $2/contact | 3,220 customers | `hillstrom_experiment_design.json` |
| Effect-decay detection | PSI 2.90 at 45% decay | `hillstrom_monitoring.json` |
