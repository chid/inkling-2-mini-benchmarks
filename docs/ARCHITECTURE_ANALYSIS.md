# Looped MoE Architecture & Hardware Execution Analysis

## 1. Architectural Overview: Thinking Machines Inkling 2 Mini

Inkling 2 Mini is a 26.4B total parameter Mixture-of-Experts (MoE) model with dynamic routing across 128 experts, activating ~3.4B parameters per forward token. Unlike standard feedforward architectures (like Mixtral, Qwen, or LLaMA), Inkling 2 Mini implements **Recurrent Core Loop Iteration**.

### The Recurrent Layer Geometry
- **Total Physical Layers**: 24 transformer decoder blocks.
- **Base Layers (Initial Forward)**: Layers 0 through 5 (6 physical passes).
- **Recurrent Core (Loop Pass)**: Layers 6 through 17 (12 physical layers, traversed **3 consecutive times per token**).
- **Tail Layers (Final Forward)**: Layers 18 through 23 (6 physical passes).

$$\text{Effective Logical Depth} = 6 + (3 \times 12) + 6 = \mathbf{48\text{ sequential layer passes per token}}$$

---

## 2. The Arithmetic of Decode Latency

In autoregressive token generation (batch size = 1), compute intensity is memory-bandwidth bound. Every token emitted requires streaming active weights from VRAM to compute cores.

### Breakdown at 21.79 tokens/second (Native ROCm):
$$\text{Per-Token Latency} = \frac{1000\text{ ms}}{21.79\text{ tokens}} \approx \mathbf{45.89\text{ ms / token}}$$

Dividing across the 48 sequential layer visits:
$$\text{Per-Layer Visit Latency} = \frac{45.89\text{ ms}}{48\text{ visits}} \approx \mathbf{0.956\text{ ms / visit}}$$

### Operations Performed in 0.956 Milliseconds:
Within each sub-millisecond layer invocation, the hardware must execute:
1. **Layer Normalization**: RMSNorm activation.
2. **Attention Projections**: Query, Key, Value tensor projections with Rotary Positional Embedding (RoPE).
3. **Sliding Window Attention (SWA)**: Self-attention against past KV cache positions (1536 window).
4. **MoE Gating / Routing**: Top-k router logits calculation over 128 experts.
5. **Expert Gathering & Execution**: Matrix multiplication across chosen active experts (~3.4B active parameters).
6. **Residual Sum & Writeback**: Accumulate residual state and write activations to KV memory.

At ~0.95 ms per invocation, generic inference runtimes operate directly at the physical memory streaming and driver kernel-launch threshold.

---

## 3. Runtime Dispatch Comparison: Vulkan vs. ROCm vs. Strata

### Mesa RADV Vulkan Bottleneck (13.79 t/s)
- Vulkan executes compute passes via SPIR-V shaders.
- Standard Vulkan engines dispatch each layer visit as a distinct command buffer or pipeline barrier.
- For 48 sequential layer passes, 48 Vulkan pipeline barriers and non-contiguous memory bindings are evaluated per token:
  $$48 \times 1.51\text{ ms} \approx 72.5\text{ ms / token} \implies \mathbf{13.79\text{ t/s}}$$
- Mesa RADV on Navi 48 (GFX1201 / RDNA4) additionally incurs non-conformant extension emulation overhead.

### Native ROCm HIP Direct Dispatch (21.79 t/s)
- Direct Kernel Fusion Driver (`/dev/kfd`) submission bypasses the Vulkan command buffer state machine.
- Direct HIP stream kernel launches reduce dispatch overhead from 1.51 ms down to 0.95 ms per layer visit:
  $$48 \times 0.956\text{ ms} \approx 45.89\text{ ms / token} \implies \mathbf{21.79\text{ t/s}}$$

### Strata Fused Multi-Visit Shape Engine (50.00 t/s)
- Rather than dispatching 48 individual kernel launches per token, `strata-inkling` (0.1.67 GFX1201) implements:
  1. **Hot Expert Arena**: Pre-allocates a contiguous 10.62 GiB memory arena directly on the GPU, avoiding per-expert pointer chasing.
  2. **Fused Multi-Visit Graph Execution**: Collapses the recurrent middle loop ($3 \times 12 = 36$ passes) into a unified compute graph executed in a single driver call.
  3. **Result**: Per-token decode latency drops from 45.9 ms to **20.0 ms**:
     $$\text{Throughput} = \frac{1000\text{ ms}}{20.0\text{ ms}} = \mathbf{50.0\text{ t/s}}$$

---

## 4. Speculative Decoding Horizons (DSpark Drafter)

To achieve **150–210+ tokens/second**, speculative decoding separates generation into two asymmetric components:
1. **Drafter Pass**: A small ~400 MB model (`dspark-Inkling-2-Mini-Q8_0.gguf`) rapidly guesses 3–5 tokens in parallel.
2. **Base Verification Pass**: The 48-pass base model evaluates all candidate tokens in a **single batched forward pass** (which runs at prefill speeds >200–700 t/s).
3. **Effective Yield**: If 3 draft tokens are accepted per verification step, effective generation throughput reaches $3 \times 50\text{ t/s} = \mathbf{150\text{ t/s}}$, bypassing the memory bandwidth wall.
