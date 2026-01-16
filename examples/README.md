# Examples

These scripts use the installed package. Run them from the repo root after:

```bash
pip install -e .
```

- `python examples/run_evaluation.py` (uses `docs/eval_example.yml`)
- `python examples/generate_anatomy_dataset.py`
- `python examples/play_agent.py --tool <model>/<wire> --checkpoint <path> --arch-record <id>`
- `python examples/train_paper.py --tool <model>/<wire> -d cpu -nw 2 -n demo_run`
