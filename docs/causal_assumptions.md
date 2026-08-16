# Causal assumptions, and what happens when they break

Every number in this project is a causal claim: "contacting this customer *causes*
a change in their probability of visiting." Prediction accuracy cannot support a
claim like that. What supports it is a set of assumptions, plus a randomized
experiment that makes those assumptions credible rather than merely stated.

This document is deliberately explicit about which assumptions are satisfied by
construction here, which are assumed, and which would need re-examining before
this ran against live traffic.

---

## The fundamental problem

For each customer we want

    tau(x) = P(visit | contacted, X = x) - P(visit | not contacted, X = x)

but each customer is only ever observed under one of those two conditions. The
individual difference is never observable — not for a single person, not once.
Everything below exists to make the *group-level average* of that difference
estimable, which is the strongest thing available.

This is also the honest answer to "how do you know your uplift predictions are
right?" You do not, at the individual level, ever. You validate that customers
your model ranks highly show a larger treated-vs-control gap on held-out
randomized data than customers it ranks low. That is a claim about the ranking,
and it is testable — which is what the Qini curve tests.

---

## Assumption 1 — Unconfoundedness

**Statement.** Treatment assignment is independent of the potential outcomes,
given the observed features.

**Status here: satisfied by design.** Both datasets come from randomized
experiments. Hillstrom randomly assigned 64,000 customers across two email
creatives and a no-email control; Criteo randomized ad exposure. Randomization
makes assignment independent of *everything*, observed or not, which is a far
stronger guarantee than any statistical adjustment can provide.

**Verified, not assumed.** `scripts/01_explore.py` computes a covariate balance
table. On Hillstrom every feature has |SMD| < 0.014, well inside the conventional
0.1 threshold. Note that t-test p-values are the wrong tool at this sample size —
with 64,000 rows they flag differences far too small to bias anything, which is
why standardized mean difference is the decision rule.

**What breaks it in production.** This is the assumption most likely to fail once
the model is live, and it fails quietly:

- The campaign team excludes "unsubscribed" or "low engagement" customers before
  the file reaches the model. Assignment now depends on engagement, which also
  drives the outcome.
- The model's own recommendations decide who gets contacted. After one cycle,
  treatment is a function of predicted uplift, and the next training run is
  confounded by its own predecessor.
- Delivery failures. Bounced email is a non-random subset correlated with
  account age and activity.

The mitigation is the same in all three cases: hold back a permanent randomized
control group — 5-10% of the file, contacted or not by coin flip regardless of
model score. It is the only thing that keeps the next training set clean, and it
is what makes the monitoring in `04_monitor.py` able to close the loop at all.

---

## Assumption 2 — SUTVA / no interference

**Statement.** One customer's treatment does not affect another customer's
outcome, and there is only one version of the treatment.

**Status here: assumed, and partially questionable.**

The no-interference half is reasonable for email: one household receiving a
catalogue rarely changes another household's purchasing. It is weaker than it
looks for the Criteo advertising data, where auction dynamics mean showing an
ad to one user can change what other users see.

The single-version half is *violated in an interesting way* by Hillstrom, and
this project exploits rather than ignores it. There are two treatments — a men's
creative and a women's creative — and they are not interchangeable:

| Shopper type | Men's email | Women's email |
|---|---|---|
| Men's-only buyer | +0.069 | +0.011 |
| Women's-only buyer | +0.071 | +0.074 |
| Buys both | +0.134 | +0.071 |

Sending the women's creative to a men's-only buyer captures about 16% of the
lift the men's creative would have produced. Collapsing "email" into one binary
treatment averages over that. `scripts/05_multi_treatment.py` models the arms
separately for exactly this reason.

---

## Assumption 3 — Overlap / positivity

**Statement.** Every customer had a non-zero probability of landing in either arm.

**Status here: satisfied by design**, and worth stating because it is what makes
the X-learner's propensity weighting well-behaved. Hillstrom is roughly 2:1
treated to control; Criteo is far more lopsided at about 85:15. In both cases the
propensity is bounded away from 0 and 1 by construction, since assignment was a
coin flip and not a function of features.

The implementation still clips estimated propensities to [0.05, 0.95] in
`XLearner.predict_uplift`. That is not because overlap fails here, but because
an unclipped weight would hand the entire estimate to whichever arm has almost no
data at that point in feature space — a real hazard when this code is pointed at
observational data.

---

## Assumption 4 — The economics

Not a causal assumption, but the one most likely to be wrong in practice, and the
one that decides the recommendation.

The simulation converts incremental conversions into money using an assumed value
per conversion and cost per contact. These are inputs, not measurements. Rather
than picking a flattering pair, `scripts/03_simulate.py` sweeps the contact cost
and reports the break-even point: on Hillstrom, selective targeting only beats
contacting everyone above **$0.74 per contact**, against an incremental revenue
of **$0.583 per contact** at the optimal depth.

At email's real cost of roughly $0.001-0.01 per send, contacting nearly everyone
is the correct policy, and the model's ranking is worth having for a different
reason — knowing who to drop first when a budget, a send limit, or a
customer-fatigue constraint binds.

---

## Assumption 5 — Stability over time

**Statement.** The treatment effect estimated on historical data still holds when
the policy runs.

**Status: assumed, and actively monitored because it is the one that decays.**

Offers lose novelty, competitors respond, and repeatedly-targeted segments
saturate. This failure is invisible to a conventional monitor: the feature
distribution is unchanged, the outcome rate is unchanged, and only the *response
to treatment* has moved. `src/uplift/monitoring.py` therefore tracks the
distribution of predicted uplift itself, and `realized_effect` compares predicted
against observed on the randomized holdout once labels arrive.

`scripts/04_monitor.py` plants three failures — covariate shift, a broken
feature, and effect decay — and detects all three.

---

## What would be needed to trust this in production

1. **A permanent randomized holdout.** Non-negotiable. Without it, every future
   retraining is confounded by the current model's own decisions, and there is no
   way to measure whether the policy is still working.
2. **A live A/B test of the policy itself.** The simulation here estimates what
   the policy *would have* earned on historical randomized data. That is the
   right way to choose a policy; it is not the same as having run it.
   `scripts/07_experiment_design.py` sizes that test, and the result is a
   constraint rather than a formality: powered on profit per customer, the
   comparison needs **130 million customers** at email's contact cost, against
   **3,220** at $2.00 per contact. Powering it on response rate instead is worse
   than useless here — every subgroup has a positive effect, so withholding
   contact necessarily lowers the response rate and the test is built to
   conclude that targeting hurts.
3. **Real unit economics** from finance, replacing the assumed values, along with
   the cost of a customer-fatigue unsubscribe — which is the term most often
   missing and the one that makes suppression valuable.
4. **Effect-decay alerting wired to a retraining trigger**, not just a log line.
5. **Honest reporting of the confidence intervals.** On the Hillstrom holdout,
   the profit advantage of the best policy over treating everyone has a 95%
   interval that includes zero. The point estimate favours the model; the data
   do not rule out no difference. Both facts belong in the same sentence.

---

## Known limitations of this implementation

- **Segments are a decision rule, not recovered types.** Persuadable / sure thing
  / lost cause / sleeping dog are defined by pairs of potential outcomes that are
  never jointly observed. The labels here are quantile cuts on predicted effect
  and predicted baseline. What *is* verified is that each labelled group shows
  the claimed behaviour on randomized holdout data.
- **The conversion outcome is too sparse to model heterogeneity.** Hillstrom's
  conversion rate is 0.6% in the control arm. The campaign clearly moves it —
  ATE +0.0050, 95% CI [0.0035, 0.0064] — so the limitation is not that the
  effect is absent but that ~110 control conversions in a 19K holdout cannot
  support estimating how that effect *varies* across customers. Visits are
  modelled instead, and the gap between "caused a visit" and "caused a purchase"
  is a real limitation, not a modelling detail.
- **Qini is undefined without an overall effect.** The coefficient divides by the
  total incremental outcome, so on an experiment with no effect it returns noise
  divided by noise. `qini_coefficient` returns NaN when the total effect is not
  statistically distinguishable from zero, rather than a flattering number.
- **Model selection is noisy.** Bootstrap Qini intervals on a 19K holdout overlap
  for every learner tested, which is why selection uses repeated cross-validation
  and reports whether the winner is actually separated from the runner-up.
- **The uncertainty estimates are themselves miscalibrated.** The Bayesian
  bootstrap in `src/uplift/bayesian.py` gives standardized residuals with sd 1.77
  where a calibrated posterior gives 1.0. The variance decomposition shows
  posterior variance is only ~5.5% of the residual denominator, so widening the
  intervals does not fix it — the excess is bias in the point estimates, which
  needs a better model or more data rather than wider error bars. The method
  propagates uncertainty from resampling the data and says nothing about whether
  the model class is right; that gap is exactly what the residuals detect.
