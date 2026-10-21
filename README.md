# MetaQuant Problems

**Turn a problem description into a structured optimization model, compare CPU and GPU solutions, and understand the measured results with Meta Llama.**

Project domain: **metaquantproblems.com**.

MetaQuant Problems is an independent experimental application for binary optimization. It combines Meta Llama, served through Fireworks, with a classical CPU baseline and an on-demand NVIDIA GPU solver running on Modal.

The interface uses a white background and blue text, with a single RUN button and separate result panels for both engines. Quantum-inspired optimization runs on classical GPU hardware; the application does not use a quantum computer.

## What it does

- Formulates supported optimization problems from natural-language descriptions using Meta Llama.
- Keeps structured problem data internal and validates it before execution.
- Provides 18 synthetic scenarios covering festival sessions, charging zones, habitat restoration, warehouse dispatch, and sensor modes.
- Runs simulated annealing on CPU and Simulated Bifurcation on GPU in separate, concurrent processes.
- Displays each engine's result as soon as it becomes available.
- Performs three runs per engine and reports total solver time, median time, objective value, and constraint checks.
- Independently verifies solution energies on CPU.
- Exports measurements and solutions as JSON, with binary decisions available as CSV.
- Uses Meta Llama to explain completed measurements.

English and Spanish are fully covered in the main interface. French and German include the main interface labels, with English fallback for some technical messages.

## Architecture

| Component | Responsibility |
| --- | --- |
| Web interface | Problem selection, natural-language input, results, and downloads |
| Render / FastAPI | Serves the interface, validates requests, tracks jobs, and calls external services |
| Meta Llama / Fireworks | Proposes structured problem formulations and explains measured results |
| Modal / NVIDIA L4 | Runs the CPU baseline and GPU solver in isolated processes |
| Optimization engines | Build the mathematical problem, solve it, and verify objective values |

Predefined examples do not require a model API call. Natural-language formulation and result explanations do.

The GPU starts on demand. The default configuration allows one GPU container, scales down to zero, and keeps a container warm for up to 300 seconds after activity. The GPU worker does not expose a public HTTP endpoint.

## Problem models

| Scenario | Synthetic benchmark model |
| --- | --- |
| Festival sessions | Assign events to two sessions while minimizing pairwise audience conflicts |
| Charging zones | Split requests between two zones, penalizing load imbalance and resource contention |
| Habitat restoration | Select actions using individual benefits and signed pairwise interaction costs |
| Warehouse dispatch | Assign shipments to two hubs using handling and pairwise transport costs |
| Sensor modes | Optimize binary sensor decisions with local biases and signed pairwise interactions |

Large examples contain 512 to 4,096 binary decisions, depending on the scenario. They use generated data and are explicitly labeled as synthetic.

User-supplied data supports bounded models such as weighted two-way partitioning, load balancing, action selection with forbidden pairs, small one-to-one assignment, and general QUBO matrices. Limits depend on the model: most actual-data inputs support up to 128 decisions, while assignment cost matrices support up to 10 rows.

Generated benchmarks and actual-data models can have different constraints. Check that the model description matches your intended problem before running. The formulation prompt instructs Llama to request missing data and reject unsupported constraints rather than silently replace them. Schema validation checks structure and numeric limits; users must still verify that the proposed formulation captures their intended problem.

## Measurement methodology

Both engines receive the same mathematical problem. Each engine runs three times, and the best objective value is retained alongside the timing of every run.

Large synthetic benchmarks use these fixed profiles:

- CPU: simulated annealing with 8 reads and 200 sweeps per run.
- GPU: Simulated Bifurcation with 128 agents and 1,000 steps per run, without early stopping.

These are different algorithms with different computational budgets. A timing ratio does not demonstrate quantum advantage, equal work, or equal solution quality. Compare objective values and constraint checks as well as speed.

Solver times exclude model API calls, network latency, and initial warm-up. They include the representation conversion, transfers, and solution extraction performed inside the measured solver calls. End-to-end request time is displayed separately.

GPU execution requires CUDA and verifies the actual tensor devices. There is no silent CPU fallback. Large benchmark instances have no proven optimum; small supported instances can receive an exact reference check.

## Local setup

Use Python 3.12. From the repository root on Linux or Windows WSL:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
COOKIE_SECURE=0 python -m uvicorn server.main:app --reload
```

Open `http://127.0.0.1:8000`.

Without provider credentials, you can inspect the interface. GPU execution and model requests remain unavailable until configured.

`.env.example` documents the settings. The application does not automatically load that file: export variables in your shell or configure them in Render.

## Deploy the GPU worker

Create a Modal account at [modal.com](https://modal.com/), configure billing as required for GPU access, and authenticate:

```bash
python -m modal setup
python -m modal deploy --env main modal_app.py
```

`modal_app.py` defines the NVIDIA L4 worker and its CUDA dependencies. Use the same Modal workspace for deployment and the credentials configured in Render.

The current package retains the internal Modal application name `metaquant-forge` and function name `compare_job`. These internal identifiers must match between `modal_app.py` and `server/main.py`; they do not determine the public domain or project name.

## Configure Meta Llama

Sign in at **[app.fireworks.ai](https://app.fireworks.ai/)** and create an API key.

The default configuration is:

```text
META_API_BASE_URL=https://api.fireworks.ai/inference/v1
META_MODEL=accounts/fireworks/models/llama-v3p3-70b-instruct
```

Set `META_API_KEY` to your Fireworks key. Model access and available credit must be confirmed in your provider account.

## Deploy on Render

1. Upload the project contents to a GitHub repository. Keep `render.yaml`, `requirements.txt`, `modal_app.py`, `server/`, `optimizer/`, and `dist/` at the repository root.
2. Open [Render](https://dashboard.render.com/) and select **New → Blueprint**.
3. Connect the repository and select the branch to deploy.
4. Enter `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, and `META_API_KEY` when prompted.
5. Review the service plan and create the Blueprint. `SESSION_SECRET` is generated automatically.
6. Open the assigned HTTPS URL and check `/health`.
7. Run a small problem before trying the large benchmarks.
8. Add `metaquantproblems.com` under **Settings → Custom Domains**, then configure the DNS records supplied by Render.

The Blueprint uses the free Render plan initially. Choose a paid plan if your demonstration requires the web service to remain awake.

For a manual Web Service deployment:

```text
Runtime: Python
Build command: pip install -r requirements.txt
Start command: uvicorn server.main:app --host 0.0.0.0 --port $PORT --workers 1
Health check path: /health
```

Use one Render instance and one server worker. The GPU dependencies belong on Modal, not Render.

### Environment variables

| Variable | Purpose |
| --- | --- |
| `MODAL_TOKEN_ID` | Modal token identifier |
| `MODAL_TOKEN_SECRET` | Modal token secret |
| `MODAL_ENVIRONMENT` | Modal environment; defaults to `main` |
| `META_API_KEY` | Fireworks API key |
| `META_API_BASE_URL` | Model provider's compatible API base URL |
| `META_MODEL` | Meta model identifier |
| `SESSION_SECRET` | Persistent random secret for signing session cookies |
| `COOKIE_SECURE` | `1` for HTTPS deployment; `0` for local HTTP development |
| `PYTHON_VERSION` | Python version selected by Render |

Keep all credentials server-side. Never commit actual tokens or API keys.

## First test

Choose a predefined example, select the smallest available size, and press **RUN** once. Predefined examples skip Fireworks formulation. Wait for the initial container startup. Each result panel updates when that engine's result arrives; completion order depends on the actual run.

For a formulation test, explicitly request synthetic data, for example:

> Create a synthetic benchmark with 512 festival events assigned to two sessions, without capacity constraints.

## Windows PowerShell: deploy from this complete bundle

Extract the ZIP into a new folder. Open PowerShell in the folder containing `modal_app.py`, `requirements.txt`, and the complete `optimizer` directory. With Python 3.12 installed, run each command separately:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m modal setup
.\.venv\Scripts\python.exe -m modal deploy --env main modal_app.py
```

During authentication, choose the same workspace used by Render's Modal credentials. Wait for `App deployed!` before testing a new job. If a command fails, stop there and capture its error output. The virtual environment is created locally; it is not shipped in this ZIP.

The corrected Modal worker serializes progress and results as plain JSON-compatible data so that Render does not need PyTorch to deserialize them. A GitHub/Render update does not deploy the worker to Modal; the final command above performs that separate update.

## Tests and validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Local verification covered problem models, API behavior, job ownership, Modal progress polling, plain-data serialization, and preservation of completed results when the other engine fails. Browser checks used simulated responses. These checks do not establish real GPU speed or end-to-end production success.

The deployed worker still needs to be verified in the owner's Modal account using a new job after deployment.

## Operational limits

This is a demonstration prototype with one concurrent comparison per gateway and one GPU container. Job tracking is held in memory and is lost on restart; download results you want to retain. The Modal function is configured with a 600-second execution timeout.

The application has basic request limits and session-based job ownership, but no user-account authentication or guaranteed monetary spending cap. Sustained public use requires durable job storage, persistent quotas, and access controls. An abruptly interrupted gateway may leave a remote job running until its timeout.

Natural-language input and explanation data are sent to Fireworks. Executed problem specifications are sent to Modal. Provider charges depend on usage, including applicable container loading, warm time, CPU, memory, GPU, and model API costs.

## Project files

- `dist/`: web interface and translations.
- `server/main.py`: FastAPI gateway and job tracking.
- `server/meta_ai.py`: Meta model formulation and explanation adapter.
- `optimizer/`: problem models, numerical engines, and verification.
- `modal_app.py`: on-demand GPU deployment.
- `render.yaml`: Render deployment configuration.
- `tests/`: local automated tests.
- `INSTALACION.txt`: Windows PowerShell deployment instructions in Spanish.

The public project name is MetaQuant Problems. The internal Modal application name remains `metaquant-forge`.

## License

See [LICENSE](LICENSE).
