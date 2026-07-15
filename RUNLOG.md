Train 1
Hypothesis: Baseline run.
Changes: None.
BPB Score: 2.3718
{"bpb": 2.3718, "n_params": 1339840, "steps": 2000, "tokens_in_eval": 159225, "tokens_scored": 159224}

Train 2
Hypothesis: Increased the model parameters to 1.9M
Changes: block_size = 256, n_layer = 6, dropout = 0.1, tie_weights = True
BPB Score: 2.8672
{"bpb": 2.8672, "n_params": 1937920, "steps": 2000, "tokens_in_eval": 159225, "tokens_scored": 159224}

Train 3
Hypothesis: Decreased to original model size
Changes: dynamic scheduling
BPB Score: 2.6382
{"bpb": 2.6382, "n_params": 1298880, "steps": 2000, "tokens_in_eval": 159225, "tokens_scored": 159224}

Train 4
Hypothesis: Reduced matrix sizes and implemented gradient accumulation (micro_batch=4, accum_steps=4) to fix training time bottleneck while maintaining stability.
Changes: block_size = 128, n_layer = 4, n_embd = 128, OneCycleLR + AdamW + Gradient Clipping.
BPB Score: 2.4569
Conclusion: Training time dropped to ~6.5 minutes and the loss curve was perfectly stable, but the model capacity (842k params) was too small to beat the baseline. Need to scale width and depth back up using the freed memory.

Train 5
Hypothesis: Scaled the optimized architecture back up to the parameter limit (~1.9M) to maximize capacity, relying on the gradient accumulation and OneCycleLR to maintain stability.
Changes: block_size = 128, n_layer = 6, n_embd = 160. Biases removed from all Linear layers. Tiled embedding weights.
BPB Score: 2.3511
Conclusion: The model successfully converged and decisively beat the 2.3718 baseline. The architectural efficiencies allowed for a much deeper and wider network within the strict 2,000,000 parameter budget.