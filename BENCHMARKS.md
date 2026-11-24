# Hardware Benchmarks: Inkling 2 Mini (26.4B Looped MoE)

Comprehensive benchmark progression and empirical observations across runtime backends and hardware platforms.

---

## 1. Master Hardware Comparison

| Hardware Platform | Backend & Version | Model Quantization | VRAM State | Prompt Processing (`pp128`) | Token Generation (`tg64`) | Latency / Tok | Bottleneck Analysis |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Z2 Handheld**<br>AMD Ryzen Z2 (8C/16T)<br>Radeon 780M (GFX1103) | Mesa RADV Vulkan<br>`llama-b12626` | `Q4_K_M`<br>(18.66 GB) | Unified LPDDR5x<br>(21.0 GiB addressable) | **91.34 t/s** | **11.11 t/s** | 90.0 ms | Memory bandwidth ceiling of 128-bit bus (~75 GB/s sustained) |
| **Workstation**<br>NVIDIA RTX 3080 10GB<br>Ryzen 5 5600X | CUDA 12.8<br>`llama-b12626` | `AD-IQ3_S`<br>(13.34 GB) | Hybrid Spill<br>(10 GB VRAM + DDR4) | ~150–180 t/s | **~10–12 t/s** | ~95.0 ms | **PCIe Wall**: Spilling ~3.5 GB into host DDR4 throttles serial layer passes |
| **Workstation**<br>AMD Radeon RX 9070 XT 16GB<br>Ryzen 5 5600X | Mesa RADV Vulkan<br>`llama-b12626` | `AD-IQ3_S`<br>(13.34 GB) | Shared VRAM<br>(5.1 GB Free / 7.1 GB GTT) | **127.82 t/s** | **12.83 t/s** | 77.9 ms | Background `qwen-heretic` held 11.7 GB VRAM, forcing PCIe GTT fallback |
| **Workstation**<br>AMD Radeon RX 9070 XT 16GB<br>Ryzen 5 5600X | Mesa RADV Vulkan<br>`llama-b12626` | `AD-IQ3_S`<br>(13.34 GB) | Isolated Dedicated<br>(16.3 GB Free) | **143.39 t/s** | **13.79 t/s** | 72.5 ms | **SPIR-V Dispatch Bound**: ~65 ms/tok spent across 48 sequential Vulkan pipeline barriers |
| **Workstation**<br>AMD Radeon RX 9070 XT 16GB<br>Ryzen 5 5600X | **Native ROCm HIP**<br>HIP 7.1 (`core-10.1`) | `AD-IQ3_S`<br>(13.34 GB) | Isolated Dedicated<br>(16.3 GB Free) | **722.88 t/s** 🚀 | **21.79 t/s** ⚡ | 45.9 ms | Direct KFD kernel dispatch removes Vulkan barrier overhead; hits physical GDDR6 streaming limit |
| **Workstation**<br>AMD Radeon RX 9070 XT 16GB<br>Ryzen 5 5600X | **Strata Engine**<br>v0.2.21 (`strata-inkling`) | `AD-IQ3_S`<br>(13.34 GB) | Isolated Dedicated<br>(10.62 GB Arena) | **234.60 t/s** | **50.00 t/s** 🔥 | 20.0 ms | **Fused Loop Dispatch**: Multi-visit fused kernels and hot expert arena cut decode latency by 2.3× |

---

## 2. ROCm HIP Parameter Optimization Sweep (AMD RX 9070 XT)

Synthetic benchmark sweep executing `llama-bench -m Inkling-2-Mini-Uncensored-AD-IQ3_S.gguf --device ROCm0 -ngl 99 -r 1 -p 128 -n 64` on dedicated VRAM:

| CPU Threads (`-t`) | Flash Attention (`-fa`) | Prompt Processing (`pp128`) | Token Generation (`tg64`) | Delta vs 12T Baseline | Notes |
| :---: | :---: | :---: | :---: | :--- | :--- |
| **4** | 1 (ON) | 646.35 t/s | 21.64 t/s | -10.6% prompt | Compute underutilization during batch prefill |
| **6** | 1 (ON) | 714.73 t/s | 21.77 t/s | -1.1% prompt | Matches 6 physical Zen3 cores |
| **8** | 1 (ON) | **722.88 t/s** | **21.79 t/s** | **+19.6% prompt** | **Optimal Configuration**: Balanced Zen3 cache and thread dispatch |
| **12** | 1 (ON) | 661.66 t/s | 21.75 t/s | Baseline | SMT thread contention slightly degrades prefill throughput |
| **4** | 0 (OFF) | 653.16 t/s | 21.57 t/s | -9.6% prompt | - |
| **6** | 0 (OFF) | 681.38 t/s | 21.69 t/s | -5.7% prompt | -33 t/s slower prompt processing without FA |
| **8** | 0 (OFF) | 675.78 t/s | 21.67 t/s | -6.5% prompt | -47 t/s slower prompt processing without FA |
| **12** | 0 (OFF) | 685.87 t/s | 21.68 t/s | - | - |

---

## 3. Live Single-Turn Generation Comparison

Prompt: `"Explain quantum tunneling in two clear paragraphs."` (Prompt tokens: ~41, Generated tokens: ~128)

| Backend & Configuration | Prompt Processing Speed | Generation Speed | Time to First Token (TTFT) | Total Elapsed Time |
| :--- | :---: | :---: | :---: | :---: |
| **Z2 Handheld (Vulkan0)** | 31.9 t/s | 11.1 t/s | 1,280 ms | 12.81 s |
| **RX 9070 XT (Vulkan1 Dedicated)** | 35.9 t/s | 13.3 t/s | 1,142 ms | 10.76 s |
| **RX 9070 XT (ROCm0 Standard, Threads=8, FA=on)** | 89.5 t/s | 21.9 t/s | 458 ms | 6.30 s |
| **RX 9070 XT (ROCm0 + N-Gram Speculative)** | 88.1 t/s | **22.1 t/s** | 465 ms | 6.25 s |
| **RX 9070 XT (Strata 0.1.67 Dedicated)** | 234.6 t/s | **50.0 t/s** | 210 ms | **2.77 s** |
