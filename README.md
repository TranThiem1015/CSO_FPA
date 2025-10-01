# CSO_FPA

Research code for comparing Cat Swarm Optimization (CSO) and Flower Pollination Algorithm (FPA)
for a K-user interference-channel power allocation problem.

## Contents
- `CSO_FPA.py` : single-file experimental harness implementing CSO and FPA, plotting, and CSV outputs.
- `design/CSO_FPA_algorithm_design.txt` : design document describing algorithms, pseudocode and evaluation plan.
- `design/flowchart.puml` : PlantUML activity flowchart for the main experiment script.

## Quick start

1. Create a Python 3 virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install numpy matplotlib pandas
# Optional: nicer tables
pip install tabulate
```

3. Run the experiment (small smoke test):

```bash
python CSO_FPA.py
```

The script runs experiments for P_max values [1,10,100] by default and saves PNG/CSV outputs in the working directory.

### Windows helper

If you are on Windows, a convenience batch script `requirements.bat` is provided to create a virtual environment and install dependencies:

```bat
requirements.bat
```

This script will create a `.venv` folder, activate it, upgrade pip and install the packages listed in `requirements.txt`.

## Outputs
- `convergence_P{P}.png` : convergence plot comparing CSO and FPA for each P_max
- `convergence_metrics_P{P}.csv` : per-algorithm convergence speed metrics
- `population_summary_P{P}.csv/.txt` : snapshot tables of populations at selected iterations

## Development notes
- The code is a research harness and stores rich per-iteration histories for post-hoc analysis and plotting.
- Comments use a JavaDoc-like brief format encoded as Python comments per project preference.

## License
Add a license of your choice (e.g., MIT) or remove this section.

## Contact
Add maintainer/contact details here if desired.

Made with ❤️ by [Trần Thị Thiêm]
