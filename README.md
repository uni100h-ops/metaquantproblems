# MetaQuant Problems: Modal GPU availability patch

This patch changes only `modal_app.py`. Copy that file over the existing file
in your **complete project folder**, next to `requirements.txt` and `optimizer/`.
Do not deploy from this patch ZIP's extraction folder alone.

## Confirmed diagnosis

Modal logged that `compare_job` was waiting for an available `GPU_L4` worker.
The diagnostic request queued at 20:38:58 UTC on October 21, 2026 did not start
until 20:43:21 UTC: a 4 minute 23 second queue delay. Render submitted the call
to Modal successfully. The request was cancelled; it did not produce a solution.
Earlier cancelled calls also had tracebacks inside the first PyTorch import.
Those tracebacks identify where the interruption occurred, but do not establish
a separate import deadlock.

## Changes

- Request L4 first, with T4 as an allowed fallback when L4 is unavailable.
- Import PyTorch during container initialization through `image.imports()`.
  This avoids first importing it inside the input handler. CUDA verification
  still occurs in the worker, and PyTorch is not required on your computer.
- Preserve the one-container limit, zero minimum containers, five-minute
  idle window, solver profiles, problem complexity and partial result delivery.
- Report the GPU actually used. The GPU solver does not fall back to the CPU.

The fallback reduces dependence on L4 capacity; it cannot guarantee immediate
GPU availability. Cold startup and solver time still apply. A different GPU
can change measured performance.

## Deploy from Windows PowerShell

After replacing `modal_app.py`, use your existing complete project directory:

```powershell
cd "C:\Users\Utilisateur\Downloads\MetaQuantProblems_Completo_Actualizado"
if (!(Test-Path ".\.venv\Scripts\python.exe")) { py -3.12 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m modal deploy --env main modal_app.py
```

If your complete project has a different folder name, change only the `cd` path.
If Modal explicitly reports that authentication is missing, run:

```powershell
.\.venv\Scripts\python.exe -m modal setup
```

Then repeat the deploy command. Use the same `uni100h` workspace and `main`
environment as your existing `metaquant-forge` deployment.

Uploading this file to GitHub alone does not update the Modal deployment.
Also replace it in the repository so future deployments retain this change.
The current Render code and frontend do not need replacing for this patch.

## Validation and remaining step

The existing 17 Modal progress and Render gateway tests pass, including worker
reuse, partial results and serialization without PyTorch on Render. The Modal
SDK accepts the updated deployment definition locally.

This patch has **not been deployed to Modal or verified on a live GPU yet**.
After a successful deployment, run one example on the website and check the
new Modal function call. Do not assume a fixed response time until measured.

Official references:
- https://modal.com/docs/examples/gpu_fallbacks
- https://modal.com/docs/guide/images
