---
name: Experiment Issue Template
about: 'Use this template for any empirical study: simulation runs, algorithm comparisons,
  metric evaluations, exploratory trials, etc.'
title: "[Experiment]: <brief descriptive title>"
labels: experiment
assignees: ''

---

<!--
=======================================================
 Experiment Issue Template
=======================================================
Use this template for any empirical study: simulation runs,
algorithm comparisons, metric evaluations, exploratory trials,
etc.  It is general enough for guide‑wire navigation tests,
RL hyper‑parameter sweeps, XAI probes, or any future experiment.

Filling hints (only visible while editing):
· Keep sections concise but complete.
· Convert checklist items to *sub‑issues* if they grow.
· Track reproducibility: always note code version / commit.
· When everything is done, **add a comment** under the issue
  summarising results and linking to code, logs, figures, docs.
-->

## 📄 Name  
<!-- Replace the placeholder below with a concise name. -->
<Experiment title here>

---

## 🧐 Context / Objective  
<!-- Why are you running this experiment?  Briefly describe the goal
     or hypothesis and how it fits into the master‑thesis project. -->

---

## 🔧 Preconditions  
<!-- List anything that must already be set up (leave blank if none). -->
- SOFA compiled & runs ✓
- Baseline model available ✓
<!-- Add / remove as needed -->

---

## ✅ Tasks / Sub‑Issues  
<!-- Break the work down. Convert any line to a sub‑issue via “⋯ > Convert to sub‑issue”. -->
- [ ] Task 1 – …
- [ ] Task 2 – …
- [ ] Task 3 – …

---

## 📏 Metrics / Evaluation Criteria  
<!-- What will you measure?  Success rate, time, collisions, etc. -->

---

## 📦 Deliverables  
<!-- Tangible outputs expected from this experiment. -->
- [ ] Code / notebook
- [ ] Logs / CSV result files
- [ ] Plots / figures
- [ ] Summary report

---

## 🗂️ Version & Environment  
<!-- Ensure reproducibility. -->
- **Code commit / tag:** `<hash or tag>`
- **Branch (if any):** `<branch-name>`
- **Software versions:** SOFA v …, Python …, RL‑lib …, …
- **Hardware (opt.):** `<GPU/CPU details>`

---

<!--
🎯  Post‑completion instructions (keep this comment!):
After all deliverables are ticked off, the *author of the
issue* should add a new comment containing:
1. Links to commits or pull requests that implemented the work.
2. Links to generated logs, plots, docs, or data.
3. A brief summary of the results, interpretation, and next steps.
-->
