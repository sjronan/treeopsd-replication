# X thread — TreeOPSD verdict

POSTED 2026-10-24 evening PT as 11 replies under
https://x.com/SJRonanMD/status/2114189557843529741 (reply ids 2114190439674998877
.. 2114190453197443427; thread post 1 was included as reply 1). Verified live:
numbers reply carries fig2-gaps.png with alt text; texts match this file.
The closing reply promises the replication note, preregistration, and result
packages — GitHub link reply still to add, arXiv link after announcement.

Each block is one post. Post them in order as replies to the one above.

---

We replicated the Tsinghua TreeOPSD paper (arXiv:2610.23181) on Qwen3-4B-Thinking-2507. Preregistered, 5 arms, 3 seeds, frozen retrieval, fixed step-100 checkpoint. Two GB300 Stations. The short version: it passes against its own control, and the control is the problem.

---

The paper's control is TreePO with no clip on the update. We added the same TreePO with ordinary ratio clipping (arm E). That one arm separates "the posterior teacher works" from "any bound on the update works."

---

TreeOPSD minus the clipped control, 3 seeds, same hardware: QA average +1.98 EM [+1.03, +2.94]. BC+, the long-horizon track where the paper claims +2.9: +0.08 [-1.24, +1.41]. Per seed on QA it goes 3.78, 2.10, 0.05. At 3 seeds the seed interval includes zero.

---

Against the paper's unclipped control, TreeOPSD passes both preregistered bars: QA +3.76 [+2.81, +4.74], BC+ +3.45 [+2.13, +4.78]. Half that gap is one control seed breaking in its last 10 steps. Length-clip hit 0.27, tool calls per question nearly doubled.

---

We measured the mechanism itself. TreeOPSD's update target differs from TreePO's by a median 1.05% at the paper's dose (1.56% in the last quarter). We preregistered 2% as the inactivity line. Then we turned the dose up.

---

At 2x dose the drift roughly doubled and the accuracy gap was -0.8 on our QA subset. At 4x dose the unclipped control collapsed in training. Reward 0.002. 99% of responses hit the length cap. It repeated in a second seed.

---

Ordinary clipping at the same 4x dose: reward 0.49 to 0.64 all the way through, no collapse. TreeOPSD also held. Unclipped collapses, anything bounded holds. You do not need the posterior construction to get that property.

---

Budget: the tree methods match GRPO in generated tokens (1.03x) and not in trained tokens (1.48x). Shared prefixes get re-scored per branch in training. Token-matched GRPO at G=12 removed 44% of the QA gap. Count the tokens you train on, not just the ones you sample.

---

Best-checkpoint selection did not rescue the claim either. Picking each arm's best checkpoint in the run shrinks the gap toward zero (+0.3 QA vs +1.5 at step 100), because it rescues the control seed before it started breaking.

---

What holds up: bounded tree backup trains a 4B search agent well. Untrained base 22.8 EM on our QA subset; 40.6 to 42.6 mean QA EM across the bounded arms at 3 seeds, full sets. The training works. The credit-assignment story is what our data does not support.

---

3 seeds, one 4B model, 830 BC+ questions. Those limits are in the note, with every arm's numbers. Replication note, preregistration, and result packages go up with this. Measure which part does the work. Here, it was the bound.
