# M1M2 运行测试与功能归因分析（阶段 A+B）

## 1) 可复现实验命令
- `pytest tests/m1_nlu -q`
- `pytest tests/m2_reasoning -q`
- `python scripts/run_m1_eval.py --mode pipeline`
- `python scripts/run_m2_sanity.py`
- `python scripts/run_m1_ab_robustness.py`
- `python scripts/run_m1_m2_attribution.py`

## 2) 基线结果（不依赖 GPT）
- M1 单测：7/7 通过。
- M2 单测：5/5 通过。
- M1 pipeline 评估：`intent_accuracy=0.8`，`slot_f1_macro=0.2`（当前数据集较小，槽位指标偏低）。
- M2 sanity：可观察到 clear/ambiguous/conflict 案例的不同熵水平，并且 feedback 后 `top1_prob` 提升（0.055715 -> 0.09961）。

结论：第一创新点的“可运行性与可复现性”已经在本地闭环，不需要连接 GPT 才能完成核心验证。

## 3) C0/C1/C2 对照（统一输入，15 条样本）
- C0（LLM-only，无 M2）：`intent_accuracy=0.6667`
- C1（M1 pipeline + M2）：`intent_accuracy=0.8`
- C2（M1 llm + M2）：`intent_accuracy=0.6667`

M2 指标（均值）：
- C1：`belief_entropy_drop_mean=0.877516`，`top1_prob_gain_mean=0.012398`，`candidate_reduction_rate_mean=0.972222`
- C2：`belief_entropy_drop_mean=0.523676`，`top1_prob_gain_mean=0.006527`，`candidate_reduction_rate_mean=0.972222`

归因结论：
- C1 > C0，说明你的结构化提取 + 概率推理机制带来了可测收益。
- C2 未超过 C1，说明当前阶段的增益并非来自“LLM 自带能力”，而是主要来自你的方法链路（尤其 M2 推理逻辑与 pipeline 稳定提取）。

## 4) 模糊度分桶结果（来自 attribution 脚本）
- C1 分桶准确率：`high=0.7692`，`low=1.0`，`medium=0.0`
- C2 分桶准确率：`high=0.6923`，`low=0.5`，`medium=0.0`

备注：当前样本集中 `medium` 桶样本数为 0，因此该桶准确率不具备统计意义。

## 5) 错误类型（error taxonomy）
- C1：`m1_category_mismatch=3`，`m2_not_recovered=3`，`ambiguous_input_error=1`
- C2：`m1_category_mismatch=4`，`m2_not_recovered=4`，`ambiguous_input_error=1`

可见主要误差仍来自“上游类别错判 + M2 未恢复”，这为下一步优化给出方向：
- 提升 ambiguous 检测精度（`category_confidence < 0.4` 的触发策略可进一步校准）。
- 扩展中等模糊样本并重测，避免两极样本分布导致结论偏置。

## 6) 对“是否需要连接 GPT”的最终建议
- 当前阶段（验证第一个创新点）：不需要。
- 需要连接 GPT 的时机：当你要在论文中回答“与真实通用 LLM 方案相比是否更优”时，再开启真实 API 对照（而非 mock fallback）。

## 7) 产物路径
- `artifacts/m1_m2/m1_m2_attribution_report.json`
- `artifacts/m1_m2/m1_m2_attribution_report.md`
- `artifacts/m1_m2/m1_m2_analysis_cn.md`
