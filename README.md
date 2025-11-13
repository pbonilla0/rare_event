# Rare Event Simulator

This repository contains tools to sample rare error configurations and estimate logical error rates for quantum error-correcting circuits using a splitting / Metropolis approach.

Contents
- `RareEventSimulator.py` - core simulator class and helpers.
- `example.ipynb` - interactive notebook demonstrating usage (quick demo + instructions for full runs).
- `run_full_sampling.py` - script to run large offline sampling jobs.
- `requirements.txt` - pinned Python dependencies found in the development environment.

Quick start
1. Create a virtual environment and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the example notebook (recommended)

Running a full experiment
- Use `run_full_sampling.py` for long runs (it will create `./samples` and `./weights`):

```bash
python run_full_sampling.py --help
```

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{rare_event,
  title={Rare Event Simulator},
  author={Bonilla Ataides, J. Pablo},
  year={2025},
  url={https://github.com/pbonilla0/rare_event}
}
```

License
- This project is licensed under the MIT License - see `LICENSE`.
