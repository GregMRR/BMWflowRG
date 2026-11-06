# Development

## Tests

`tests/test_model_generated.py` checks `ModelGenerated` on the vendored `tests/fixtures/KPZ_LO.py` against `ModelKPZ` (LO,
d = 1, 2); `tests/test_model_generated_lambda.py` checks `tests/fixtures/KPZLambda_LO.py` (KPZ with a flowing f_λ);
`tests/test_splines.py` checks the splines. Run them in a virtual environment:

```bash
python -m venv .venv && .venv/bin/pip install -r flow/requirements.txt pytest
.venv/bin/pytest tests                      # --module / --lambda-module <file> check other exports
```

## Documentation

- Markdown is the source. GitHub renders it; a claude.ai artifact renders the same files as one site.
- `docs/reference/` is generated from the docstrings in `flow/` (NumPy style): never edit it. Edit the docstring, then run
  `python3 tools/docgen.py reference`. The git pre-commit hook does this for you (enable it once per clone:
  `git config core.hooksPath .githooks`).
- `python3 tools/docgen.py check` fails when the reference is stale or a relative link is broken. GitHub Actions runs it on
  every push (`.github/workflows/docs.yml`).
- `docs/docgen.toml` lists the source files of the reference.
