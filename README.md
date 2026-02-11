# fem_portfolio

A small finite-element method (FEM) portfolio with a 2D solver and CLI.

## Requirements

- Python 3.8+

## Install

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run the CLI (module)

From the repo root:

```bash
python3 -m src.cli --pde poisson --domain square
```

## Run the CLI (console script)

After `pip install -e .`:

```bash
fem_solver --pde poisson --domain square
```

## Notes

- `analyses/` contains example scripts.
- `archive/` contains older experiments.
