# Training Run Log

### Train 1
* **Hypothesis:** Baseline run to establish a starting metric.
* **Changes:** None. Ran the starter code as provided.
* **BPB Score:** 2.3718 (1.34M params)
* **Conclusion:** The baseline is heavily undertuned. The tokenizer is byte-level, and the optimizer is a basic Adam without scheduling. 

### Train 2
* **Hypothesis:** Scaling up the model dimensions while tying weights will immediately yield better results due to higher capacity.
* **Changes:** `block_size = 256`, `n_layer = 6`, `dropout = 0.1`, `tie_weights = True`. Parameters bumped to ~1.9M. 
* **BPB Score:** 2.8672
* **Conclusion:** Worse than baseline. The batch size of 8 was too small for the increased dimensions, leading to noisy gradients, and the standard initialization caused the residual variance to explode. 

### Train 3
* **Hypothesis:** Returning to the original model size but introducing dynamic scheduling will stabilize learning.
* **Changes:** Added OneCycleLR scheduler and switched to AdamW for weight decay. 
* **BPB Score:** 2.6382
* **Conclusion:** The loss curve was smoother, but the model capacity is still too small to beat the baseline without further structural efficiency.

### Train 4
* **Hypothesis:** To solve the CPU bottleneck and gradient noise, I need to implement gradient accumulation and reduce the block size, while keeping the advanced optimizer.
* **Changes:** Reduced `block_size = 128`, `n_layer = 4`, `n_embd = 128` (842k params). Implemented gradient accumulation (micro_batch=4, accum_steps=4) for an effective batch of 16. Added gradient clipping.
* **BPB Score:** 2.4569
* **Conclusion:** Training time plummeted to ~6.5 minutes and the loss curve was perfectly stable. However, the 842k parameter count heavily underfit the data. I now have the stable framework needed to safely scale back up.

### Train 5
* **Hypothesis:** Scaling the optimized framework back up to the 2M limit while removing biases will maximize parameter efficiency and beat the baseline.
* **Changes:** `block_size = 128`, `n_layer = 6`, `n_embd = 160`. Removed biases (`bias=False`) from all Linear layers. Kept gradient accumulation and OneCycleLR. 
* **BPB Score:** 2.3511 (1.9M params)
* **Conclusion:** Success. The stable optimizer combined with the deeper/wider bias-free network decisively beat the 2.37 baseline. 

### Train 6 (Final Submission)
* **Hypothesis:** The byte-level tokenizer is wasting context window on 3-byte Devanagari characters, and standard MLPs/LayerNorms are less parameter-efficient than modern LLM variants (Llama-style). 
* **Changes:** 1. Replaced LayerNorm with RMSNorm to skip mean-centering (faster on CPU). 
  2. Replaced standard MLPs with SwiGLU blocks for higher expressivity per parameter.
  3. Implemented a custom Byte-Pair Encoding (BPE) tokenizer trained *only* on a 256KB subset of the corpus to prevent CPU hanging while still compressing Hindi characters.
  4. Adjusted dimensions (`vocab_size = 1920`, `n_layer = 4`, `n_embd = 160`) to perfectly hit ~1.86M params. 
* **BPB Score:** 2.2484 
* **Conclusion:** The massive jump in performance proves that sequence compression (via BPE) and modern architectural blocks (SwiGLU/RMSNorm) provide vastly superior language modeling capabilities than standard GPT-2 mechanics under strict compute/parameter constraints.