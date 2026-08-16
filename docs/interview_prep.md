# Talking about this project

Numbers here are the ones the pipeline actually produces. If you re-run with
different settings, re-read them off `reports/*.json` before using them.

---

## Resume bullet

> Built and deployed an uplift modeling system estimating heterogeneous treatment
> effects of marketing offers using meta-learners (S/T/X-learner, class
> transformation) and causal forests, validated on randomized holdout data;
> achieved a Qini coefficient of **0.19 on Hillstrom (64K customers) and 0.73 on
> Criteo (14M-row ad experiment)** against **0.08 for a conventional response
> model whose confidence interval included zero**; added Bayesian-bootstrap
> uncertainty on individual treatment effects with a calibration test, quantified
> the break-even contact cost at which selective targeting beats blanket
> campaigns (**$0.74/contact**, worth **+168% incremental profit** above it), and
> **power-analysed the validating A/B test** — showing the policy needs 130M
> customers to validate on email but only 3.2K at $2/contact; served via FastAPI
> with a drift-monitoring layer that detects treatment-effect decay invisible to
> feature-based monitors.

Shorter, if space is tight:

> Built and deployed a causal uplift modeling system (S/T/X-learners, Qini/AUUC
> validation on randomized holdouts) that separates customers a campaign actually
> persuades from those who convert anyway; Qini 0.19 vs 0.08 for a conventional
> response model, with a FastAPI service and treatment-effect drift monitoring.

**One caveat to keep you honest:** do not claim a headline revenue lift. On this
data the profit advantage over treating everyone has a 95% interval that includes
zero, and the power analysis independently confirms it is far too small to
validate on email. Claim the ranking quality, the break-even analysis, and the
experiment design — all three hold up, and the last one is the rarest thing on a
DS portfolio.

---

## The 60-second version

> Most targeting models predict who will convert. That is the wrong question,
> because it ranks your most loyal customers first — the people who would have
> bought anyway. You spend budget and get credited with conversions you already
> had.
>
> This system estimates the *causal* effect of contacting each customer: how much
> their conversion probability changes *because* of the offer. I used randomized
> experiment data, so the treated and control groups are comparable by
> construction, and I compared five approaches — S, T and X-learners, a
> class-transformation baseline, and a causal forest.
>
> You cannot score this with AUC, because the thing being predicted is never
> observed for any individual. Instead I used the Qini curve, which checks
> whether customers the model ranks highly actually show a bigger
> treated-vs-control gap on held-out randomized data. The best model got 0.19 on
> the email dataset and 0.73 on Criteo's 14-million-row ad experiment. A
> conventional response model scored 0.08 with a confidence interval that
> included zero.
>
> Then I priced it. The interesting result was negative at first: with email
> costing a fraction of a cent, contacting everyone is nearly optimal, and the
> model does not beat it. So instead of picking a flattering cost assumption, I
> swept the contact cost and found the break-even — $0.74 per contact. Above that,
> targeting the top 10% is worth 168% more profit than blanket targeting. That is
> the number a marketing team can actually act on.
>
> Then I sized the experiment that would actually validate it, and that changed
> the recommendation: on email the difference is 0.8 cents a head and needs 130
> million customers, but at $2 a contact the same test needs 3,200. So the answer
> is to validate on an expensive channel, not the cheap one.
>
> It is deployed as a FastAPI service that returns a contact/hold decision rather
> than a raw score, with drift monitoring on the treatment-effect distribution —
> because uplift models fail by the effect decaying while features look perfectly
> normal.

---

## Questions you will get

### "Why not just use a normal classifier?"

Because it answers a different question. A response model ranks by P(convert |
contacted). That puts sure things at the top — people who convert either way —
and the money spent on them buys nothing.

I can show this without any modelling. In the raw Hillstrom data, rural customers
have the *highest* response rate (20.5%) and the *lowest* uplift (+0.052). Urban
customers are the reverse. A response model spends the budget on rural first, and
it is the worst group to spend on.

And when I scored the response model with the same Qini machinery, it got 0.076
with a CI of [-0.014, 0.177] — as a way of finding incremental effect, it is not
distinguishable from random targeting.

### "How do you know your uplift estimates are correct? You can't observe the counterfactual."

At the individual level I do not, and neither does anyone else — that is the
fundamental problem of causal inference, not a gap in the implementation.

What I can validate is the *ranking*, and that is testable. Because assignment
was randomized and the ranking uses pre-treatment features only, any group the
model selects contains treated and control customers who are comparable. So I
take the top 10%, measure the actual treated-minus-control gap, and compare it to
the bottom 10%. That is exactly what the Qini curve accumulates across all
targeting depths.

The decile calibration chart is the direct version: predicted uplift plotted
against observed uplift per decile, with confidence intervals. And
`realized_effect` closes the loop — predicted +0.083 against observed +0.079 on
the top 30%, a calibration ratio of 0.95.

### "What could go wrong in production?"

Three things, in order of likelihood:

1. **Unconfoundedness breaks, quietly.** The moment the model's own output
   decides who gets contacted, next quarter's training data is confounded by this
   quarter's model. Campaign teams also filter the file before it reaches the
   model — dropping unsubscribes, say — and that filter correlates with the
   outcome. The fix is a permanent randomized holdout, 5-10% assigned by coin
   flip regardless of score. It is not optional; it is what keeps every future
   retraining valid.
2. **Effect decay.** Offers lose novelty and audiences saturate. This is invisible
   to a normal monitor: features unchanged, conversion rate unchanged, only the
   response to treatment moved. That is why the monitoring tracks the distribution
   of predicted uplift. In testing it catches a 45% effect decay at PSI 2.90 while
   a feature-drift monitor would see nothing.
3. **Sleeping dogs, if they exist.** They did not on this data — I checked, and
   every subgroup showed a positive effect, so I reported zero rather than
   manufacturing a segment. On the full 14M-row Criteo data the model flags
   0.6%, but their observed holdout interval includes zero, so I did not claim
   it either.

### "What's the business impact?"

I would answer this carefully, because the honest answer is more interesting than
a big number.

At the optimum the uplift policy earns $8,810 on the holdout against $8,631 for
contacting everyone — 2.1% better, and the bootstrap interval on that is
[$1,059, $15,764], which contains the treat-everyone figure. So on email, I would
not claim a revenue lift.

The reason is economics, not modelling: incremental revenue is $0.583 per
contact, and email costs a fraction of a cent. When contact is nearly free,
contacting everyone is close to optimal and no ranking changes that.

What the model is worth is knowing *who to drop first* when something binds — a
budget, a send limit, fatigue. I quantified exactly where that kicks in: above
$0.74 per contact, targeting the top 10% is worth 168% more profit. On Criteo the
break-even is $0.039, and past $0.27 per contact a blanket campaign actually
loses money (-$54K) while the targeted one still earns $600K on a 4.2M-row
holdout. That is the deliverable — a decision rule keyed to the channel's real
cost.

I would avoid quoting a percentage near that crossover, incidentally: when the
blanket campaign's profit approaches zero, the ratio swings from +300% to +1000%
across two cents of assumed cost. The dollar figures are the stable ones.

### "Why did the S-learner win? That's not what the literature says."

It surprised me too, and it is a good illustration of why I cross-validated
instead of trusting one split.

The S-learner pools both arms into one model. The usual objection is that a
regularized model can decline to split on the treatment feature and return zero
uplift everywhere. That failure did not happen here — the treatment effect is
large (+6 percentage points on a 10.6% base rate) so the model does split on it.

Meanwhile the T- and X-learners pay a real cost: each arm's model sees only part
of the 44,800 rows, and the noise in two separate models *adds* when you subtract
them rather than cancelling. With only 11 features and a fairly homogeneous
effect, there is not enough heterogeneity to repay that variance.

On Criteo the ordering changes — X and T-learner come out on top, which fits the
theory, since Criteo has 14M rows and a lopsided 85:15 split where the
X-learner's propensity weighting earns its keep.

That one is worth telling properly, because I ran it twice. On a 10% sample the
top three scored 0.6697 / 0.6690 / 0.6686 — gaps of 0.001, indistinguishable — so
I broke the tie on reasoning rather than the decimal. Running the full 14M rows
narrowed the confidence intervals 3.8x and the gaps grew tenfold, which separated
the X-learner from the S-learner at 2.2 standard errors but still not from the
T-learner at 0.81. So the tie-break stood, and more data confirmed the reasoning
rather than overturning it. Knowing which of those two situations you are in is
the point of reporting the standard error at all.

### "How did you pick the hyperparameters?"

By cross-validated Qini, and the answer was counterintuitive enough that I kept
the script (`scripts/02a_tune.py`). The base learners are regularized far harder
than you would for classification — 7 leaves, 800 minimum samples per leaf,
against a typical 31 and 20. Going from the standard settings to these roughly
doubles CV Qini for every learner.

The reason is that uplift is a *difference* of two small probabilities. Variance
in either arm's estimate lands directly on the difference and does not cancel.
Models that fit the outcome better rank the treatment effect worse, so optimizing
for outcome accuracy actively hurts.

### "You quantified uncertainty — was it any good?"

No, and I measured that rather than assuming it. The Bayesian bootstrap gives a
posterior on each customer's effect, and the least-certain customer's interval is
5.1x wider than the most certain, which is already useful for knowing where the
model is guessing.

But the calibration test says the intervals are too narrow: standardized
residuals come out with sd 1.77 where a calibrated posterior gives 1.0. The
obvious fix is to widen them, and that fix is wrong. The variance decomposition
shows posterior variance is only 5.5% of the residual denominator — observation
noise is the rest — so inflating it barely moves the statistic, and on held-out
data it went 1.40 to 1.34. The excess is bias in the point estimates, not
understated sampling uncertainty. Widening the bars would have made a known
error look like a quantified one.

Worth being precise about what the method does and does not capture: the Bayesian
bootstrap propagates uncertainty from resampling the data. It says nothing about
whether the model class is right, which is exactly the gap the residuals are
picking up.

### "How would you actually validate this in production?"

I sized the test, and the answer changed my recommendation.

First, the metric. Powering the test on response rate is a trap here: every
subgroup in this data has a positive effect, so withholding contact from anyone
can only lower the overall response rate. The difference at the optimal depth is
minus 0.0039. A test powered that way is designed to conclude that targeting
hurts — correct about the metric it measured, useless for the decision, because
the gains are entirely in costs avoided.

Powered on profit per customer instead, at email's contact cost the difference is
0.8 cents a head and needs 130 million customers. That is not a test anyone is
going to run. At $2.00 per contact the same comparison needs 3,220.

So the recommendation is to validate on an expensive channel, not the cheap one.
That also matches the break-even analysis from the other direction, and it is
the same conclusion the profit confidence interval reached in the simulation —
three independent routes to "the edge on email is real in point estimate and too
small to prove".

I would also size the always-on randomized holdout, which is the piece that keeps
future retraining valid: 13,515 customers to detect a 30% decay in effect.

### "Did the causal forest help?"

It came fourth, and I kept it in the comparison for that reason. `econml`'s
`CausalForestDML` is the method built specifically for treatment-effect
heterogeneity, and on this data it scores 0.103 with a confidence interval that
includes zero, at 20x the fit time of the winning S-learner.

With 44,800 rows and 11 features there is not enough heterogeneity to repay its
flexibility. That is a finding, not a failure — the useful version of "we tried
the sophisticated method" is knowing when it does not pay.

The scaling story is the better half of the answer. I measured what it costs on
Criteo: 225s at 1.4M rows, 452s at 2.8M, and at 5.6M it hit a 17.5 GB peak memory
footprint on an 8 GB machine and was killed after 110 minutes of thrashing.
Memory grows at roughly n^1.6. The four simpler learners handle all 14M rows in
under 30 seconds each at 1.7 GB.

So the method built specifically for heterogeneous treatment effects is the one
that cannot run on the dataset large enough to measure them precisely — and at
20%, where it does fit, it comes third of five and 211x slower than the S-learner
that beats it. I would rather report that than quietly leave it out of the
comparison.

### "What would you do differently with more time?"

- Model purchases rather than visits. The email does move purchases — ATE
  +0.0050, CI [0.0035, 0.0064] — but with a 0.6% control rate there are only
  ~110 control conversions in the holdout, nowhere near enough to estimate how
  that effect varies by customer. Visits are the modelled outcome for that
  reason, and the gap between causing a visit and causing a purchase is real.
- Fix the calibration properly. The intervals are miscalibrated by bias rather
  than by width, so the honest routes are a better-specified model, conformal
  prediction (which targets coverage directly rather than assuming the model is
  right), or simply more data in the high-variance regions the posterior already
  identifies.
- Run the validating experiment on a channel where it is affordable, and treat
  the email result as the design constraint it is rather than something to
  explain away.
