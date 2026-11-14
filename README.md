# Ethical Intention Booking Assistant

Code for a 2026 KTH master's thesis. The system handles vague, multi-turn booking requests: it extracts a structured intention, then decides whether to clarify or suggest. The language model sees that intention summary, not the raw dialogue or personal data.

Thesis: *Ethical AI for Task-Oriented User Intention Inference and Planning under Vague Input*

https://github.com/sswvictor/Prompt-Feed

## Layout

- **M1**: one utterance → category / time / urgency / location cues / sentiment
- **M2**: update a belief over bookable goals; choose clarify or suggest
- **M3**: store aggregate counts only; do not send raw text to the LLM
- **Dialogue**: generate replies from the intention block. Without `OPENAI_API_KEY`, a mock is used.

## Setup

Python 3.10 or newer.

```bash
git clone https://github.com/sswvictor/Prompt-Feed.git
cd Prompt-Feed
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put keys in `.env` or in the process environment, not in `config.yaml`. Tests and offline dialogue run without a key.

## Tests and demos

```bash
pytest
python scripts/demo_streamlit_smoke.py
python scripts/run_m1_eval.py --mode pipeline
python scripts/run_m3_sanity.py
streamlit run scripts/demo_pre_streamlit.py
```

Defence dining demo: `streamlit run scripts/demo_defence.py`

Train M1: `python scripts/run_m1_train.py`  
Thesis condition eval: `python scripts/run_thesis_condition_eval.py --dataset data/annotations/eval_scenarios_v2.json`  
ATIS subset: `python scripts/run_m1_public_benchmark.py`

Full public dumps (SGD, LUCID, …) are not in this repo. To rebuild the composite train set, download them into `data/external/` and run `scripts/build_m1_composite_train.py`. See `NOTICE.md` and `docs/m1_composite_mapping.md`.

```
src/        M1 / M2 / M3 / dialogue / pipeline
tests/
scripts/
data/       synthetic labels, synthetic users, ATIS mapped subset
docs/       module contracts
config.yaml
```

`.env` is gitignored. Data here is synthetic; user ids look like `persona_*`. Code is MIT (`LICENSE`). Data and third-party corpora: `NOTICE.md`.
