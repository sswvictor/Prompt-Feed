# Ethical Intention Booking Assistant

KTH 硕士论文代码（2026）。处理模糊、多轮的预约说法：先抽出结构化意图，再决定澄清还是给建议。回复用的大模型只看到意图摘要，看不到原始对话和个人信息。

论文：*Ethical AI for Task-Oriented User Intention Inference and Planning under Vague Input*

https://github.com/sswvictor/Prompt-Feed

## 结构

- **M1**：一句话 → 类别 / 时间 / 紧急度 / 地点线索 / 情感
- **M2**：在可预约目标上更新信念，选择澄清或建议
- **M3**：只存类别和时间等计数，不把原文交给 LLM
- **Dialogue**：按意图块生成回复。没有 `OPENAI_API_KEY` 时走 Mock

## 安装

Python 3.10 及以上。

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

密钥写在 `.env` 或系统环境变量里，不要写进 `config.yaml`。不填 key 也可以跑测试和离线对话。

## 测试和演示

```bash
pytest
python scripts/demo_streamlit_smoke.py
python scripts/run_m1_eval.py --mode pipeline
python scripts/run_m3_sanity.py
streamlit run scripts/demo_pre_streamlit.py
```

答辩用餐演示：`streamlit run scripts/demo_defence.py`

训练 M1：`python scripts/run_m1_train.py`  
论文条件评估：`python scripts/run_thesis_condition_eval.py --dataset data/annotations/eval_scenarios_v2.json`  
ATIS 子集：`python scripts/run_m1_public_benchmark.py`

SGD、LUCID 等完整公开集不放在这个仓库。要重做复合训练集，自己下载到 `data/external/`，再跑 `scripts/build_m1_composite_train.py`。说明在 `NOTICE.md` 和 `docs/m1_composite_mapping.md`。

```
src/        M1 / M2 / M3 / dialogue / pipeline
tests/
scripts/
data/       合成标注、合成用户、ATIS 映射子集
docs/       模块约定
config.yaml
```

`.env` 已忽略。`data/` 里是合成数据，用户 id 是 `persona_*`。代码 MIT，见 `LICENSE`；数据和第三方语料见 `NOTICE.md`。
