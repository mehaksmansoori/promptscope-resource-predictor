---
title: PromptScope
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
python_version: "3.10"
---

# PromptScope resource predictor

**Live app:** https://huggingface.co/spaces/AL-ALGO/promptscope-resource-predictor

Enter a prompt, choose Mistral or Qwen (or compare both), and set a generation token budget. The app estimates generation latency, total latency, peak memory, GPU memory, RAM usage, and throughput. It loads only the regression model and tokenizers, not the 7B LLM weights.

Estimates apply to the original benchmark hardware/configuration. They are not live measurements or utilization percentages. Output budgets outside the benchmark range may be unreliable.

## Deploy

Create a Gradio Space at https://huggingface.co/new-space. Upload `README.md`, `app.py`, `features.py`, `requirements.txt`, `build_model.py`, `model_config.json`, and `training_features.csv` to the Space root. Hugging Face builds the app from these files. The current deployment uses ZeroGPU and includes its required `spaces.GPU` wrapper. The regression computation itself uses the CPU. Hosting eligibility and any charges are governed by your account and the platform.

The server reconstructs `model.joblib` on first startup from feature-only training rows (no raw prompt text). The model was rebuilt using the supplied notebook's feature extraction, grouped 80/20 split, seed, and ExtraTrees pipeline on the matching measured Mistral/Qwen CSVs. It preserves the notebook's training split. The hosted app uses Python 3.10 compatible dependencies and reconstructs the estimator there to avoid pickle version mismatches. Only load a trusted model artifact.

## Run locally

Use Python 3.10 or newer:

```sh
pip install -r requirements.txt
python app.py
```

Open http://localhost:7860. The regression model is reconstructed at first startup, and tokenizers download from Hugging Face on first prediction. The `spaces.GPU` wrapper is for the hosted ZeroGPU runtime.

## Project files

- `app.py`: Gradio interface, prompt tokenization and prediction.
- `features.py`: prompt feature extraction from the original notebook.
- `build_model.py`: rebuilds the regression pipeline and caches it locally.
- `training_features.csv`: original training split's numeric/categorical features and resource targets; no raw prompt text.
- `model_config.json`: feature ordering, target names, tokenizer IDs and seed.
- `requirements.txt`: dependencies compatible with the hosted runtime.

The deployment was tested with one prompt in Compare both mode and returned all six metrics for Mistral and Qwen. This verifies the application flow, not prediction accuracy on new hardware. Independently predicted metrics may not satisfy physical relationships such as total latency being greater than generation latency.


