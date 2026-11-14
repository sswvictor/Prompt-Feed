# M1M2M3 运行分析（M3 阶段）

- C1 intent_accuracy: 0.6364
- C3 intent_accuracy: 0.8636
- C3 profile_aligned_accuracy: 1.0 (n=19)

## 结论（pilot）
- C3 在 PipelineConfig(enable_m3=True) 下自动串联 M1→M3→M2。
- 样本量较小，结论仅供阶段验收；扩样后重跑 attribution。