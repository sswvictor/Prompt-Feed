import json
from pathlib import Path
from collections import Counter

rep = json.loads(Path("artifacts/thesis_eval/v1/thesis_condition_eval.json").read_text(encoding="utf-8"))
ic = rep["conditions"]["Intention-conditioned"]["scenarios"]
mem = rep["conditions"]["Memory-augmented"]["scenarios"]
llm = rep["conditions"]["LLM-only"]["scenarios"]

print("=== Intention-conditioned per scenario ===")
for s in ic:
    m1 = [t["m1_category"] for t in s["turns"]]
    pred_path = [t["pred_category"] for t in s["turns"]]
    top1 = [t.get("top1_category_prob", t.get("top1_prob")) for t in s["turns"]]
    print(
        f"{s['scenario_id']}: gold={s['gold_category']} final={s['final_pred_category']} "
        f"match={s['gold_match']} clarify={s['clarify_turns']}/{s['n_turns']} "
        f"drop={s['entropy_drop']} m1={m1} pred={pred_path} top1_cat={top1}"
    )

same = sum(1 for s in ic if s["turns"][-1]["pred_category"] == s["turns"][-1]["m1_category"])
print(f"\nIC final pred == last M1: {same}/{len(ic)}")

# M1 last-turn match to gold under IC path (extractor quality on final utterance)
m1_final_match = sum(1 for s in ic if s["turns"][-1]["m1_category"] == s["gold_category"])
print(f"IC last-turn M1 == gold: {m1_final_match}/{len(ic)}")

# LLM-only uses M1 as pred
llm_match = sum(1 for s in llm if s["gold_match"])
print(f"LLM-only match (M1 final): {llm_match}/{len(llm)}")

ic_miss = {s["scenario_id"] for s in ic if not s["gold_match"]}
mem_miss = {s["scenario_id"] for s in mem if not s["gold_match"]}
print("recovered by Mem:", sorted(ic_miss - mem_miss))
print("both miss:", sorted(ic_miss & mem_miss))

# Confusion pairs for IC
pairs = Counter()
for s in ic:
    if not s["gold_match"]:
        pairs[(s["gold_category"], s["final_pred_category"])] += 1
print("IC confusion pairs:", pairs.most_common())

# Entropy rising?
rising = sum(1 for s in ic if s["entropy_drop"] < 0)
print(f"IC entropy_drop < 0 (rising): {rising}/{len(ic)}")
