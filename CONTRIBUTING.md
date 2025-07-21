# Contributing Guidelines for **gw‑tool‑recommender**

> A concise, professional guide that lets future contributors (including my future‑self) ramp up in minutes and keeps the history clean and reproducible.

---

## 1  Branching Model  (“*main / dev / feature*”)

| Branch          | Rule                                                                                                                 | Merge target                                                                               |
|-----------------| -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **`main`**      | Always *deployable* / reproducible. Holds only tested code & tagged checkpoints (e.g. `v0.1`, `paper‑camera‑ready`). |  ✗ (direct commits **forbidden**)Merge **only** via fast‑forward from `dev` or `hotfix/*`. |
| **`dev`**             | Integration branch. All day‑to‑day work is merged here.                                                              | `main`                                                                                     |
| **`feature/<topic>`** | One logical change (new device model, XAI module …). Derive from ``.                                                 | `dev` via PR                                                                               |
| **`hotfix/<issue>`** | Critical patch on `main`.                                                                                            | `main` & `dev`                                                                             |

> *Default branch**: set **`dev`** as the default branch in GitHub (Repo → Settings → Branches) so that forks and PRs point to the integration branch.

### Create a feature branch

```bash
# ensure dev is current
git checkout dev && git pull

# new work
git checkout -b feature/multi-guidewire-device
# … commit, push …

git push -u origin feature/multi-guidewire-device
# open PR → dev
```

---

## 2  Commit Message Convention  — *Conventional Commits v1*

```
<type>(<scope>): <subject>

<body>   # optional, wrap 72 characters
```

### Allowed `<type>` values

- **feat**   : new functionality / experiment
- **fix**    : bug fix or regression
- **docs**   : documentation only (README, diagrams, wiki)
- **refactor**: internal code change that neither fixes a bug nor adds a feature
- **chore**  : CI, build system, packaging, dependency bump
- **test**   : add or update tests

### Examples

```
feat(device): add flexible‑tip stiffness parameter
fix(training): prevent NaN loss when reward explodes
docs(readme): clarify quick‑start instructions
chore(ci): enable pytest smoke test in GitHub Actions
```

*Subject ≤ 72 chars, written in imperative mood (“Add”, “Fix”, “Refactor”).*

---

## 3  Pull‑Request Checklist ✅
- [ ] **Branch** rebased on latest `dev` (`git pull --rebase origin dev`).
- [ ] `pre‑commit run --all-files` passes (black, ruff, end‑of‑file newline …).
- [ ] Unit / smoke tests pass (`pytest -q`).
- [ ] Docs / examples updated if public interface changed.
- [ ] No large binary files (> 50 MB) committed (use Git LFS if needed).
- [ ] Commits squashed / rebased into logical units.

*Merge strategy:* **Squash & merge** PR into `dev`. After a milestone is stable, fast‑forward `main`:
```bash
git checkout main && git pull && git merge --ff-only dev && git push
```

---

## 4  Versioning & Tags

- **Semantic Versioning 2.0** once project reaches reproducible baseline.
- Tag created on `main`, e.g.
  ```bash
  git checkout main
  git pull
  git tag -a v0.1 -m "Baseline SAC‑LSTM benchmark"
  git push --tags
  ```

---

## 5  Coding Style & Tooling

- **black** (PEP 8 compliant, 120 char line length).
- **ruff** linter, rule‑set `ruff: select = ALL, ignore = E501` …
- **pre‑commit** config is provided; install once:
  ```bash
  pre-commit install
  ```

---

## 6  Directory Overview

```
third_party/        # git submodules (read‑only; update via submodule workflow)
dev/                # project‑specific source code (importable as 'gwtool')
docs/               # Sphinx or markdown docs, images
experiments/        # one‑off notebooks, scratch scripts
```

---

*Questions? Create an issue or ping ****@HannesDeittert****.*