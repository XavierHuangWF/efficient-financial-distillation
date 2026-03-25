# teacher_student_llms_plot_corrected.py
# Teacher vs Student LLMs: Parameters vs FP16 Weights Memory (log-log)
# Uses FP16 weights-only memory (≈ 2 bytes/parameter) for a consistent y-axis.
# Mixtral: show BOTH "total" (memory footprint if all experts resident) and "active" (~13B) as the compute-per-token proxy.

import matplotlib.pyplot as plt

# ===================== DATA (parameters in B, memory in GB) =====================
models = [
    # Teachers (disclosed)
    {"name": "LLaMA 3 (8B)",                 "params_b": 8.0,    "vram_gb": 16.0,   "group": "teacher"},
    {"name": "LLaMA 3 (70B)",                "params_b": 70.0,   "vram_gb": 140.0,  "group": "teacher"},
    {"name": "LLaMA 3 (405B)",               "params_b": 405.0,  "vram_gb": 810.0,  "group": "teacher"},
    # Mixtral MoE: total params vs. active params
    {"name": "Mixtral 8×7B (active ≈13B)",   "params_b": 13.0,   "vram_gb": 26.0,   "group": "teacher"},  # FP16 weights for active experts
    {"name": "Mixtral 8×7B (total 46.7B)",   "params_b": 46.7,   "vram_gb": 93.4,   "group": "teacher"},  # FP16 weights for all experts
    {"name": "Phi-3 (3.8B)",                 "params_b": 3.8,    "vram_gb": 7.6,    "group": "teacher"},
    {"name": "Phi-3 (14B)",                  "params_b": 14.0,   "vram_gb": 28.0,   "group": "teacher"},

    # Students (practical memory rounded up so points are visible)
    {"name": "TinyBERT (~14M)",              "params_b": 0.014,  "vram_gb": 1.0,    "group": "student"},
    {"name": "DistilBERT (~66M)",            "params_b": 0.066,  "vram_gb": 3.0,    "group": "student"},
    {"name": "ModernBERT (~150M)",           "params_b": 0.15,   "vram_gb": 6.0,    "group": "student"},
]

# ===================== LABEL OFFSETS =====================
offsets = {
    "LLaMA 3 (8B)"               : (1.00, 0.78),
    "LLaMA 3 (70B)"              : (1.00, 1.15),
    "LLaMA 3 (405B)"             : (0.88, 0.78),
    "Mixtral 8×7B (active ≈13B)" : (1.00, 1.25),
    "Mixtral 8×7B (total 46.7B)" : (1.00, 0.78),
    "Phi-3 (3.8B)"               : (1.00, 0.78),
    "Phi-3 (14B)"                : (1.00, 0.7),
    "TinyBERT (~14M)"            : (1.20, 0.78),
    "DistilBERT (~66M)"          : (1.00, 0.78),
    "ModernBERT (~150M)"         : (1.00, 0.78),
}

# ===================== PLOT =====================
plt.figure(figsize=(11, 7.5))

for m in models:
    x, y = m["params_b"], m["vram_gb"]
    if m["group"] == "teacher":
        color, size = "royalblue", 30   # teachers slightly larger
    else:
        color, size = "seagreen", 30    # students smaller
    plt.scatter(x, y, s=size, color=color, alpha=0.85,
                label=m["group"] if m["name"] in ["LLaMA 3 (8B)", "TinyBERT (~14M)"] else "")

    dx, dy = offsets.get(m["name"], (1.05, 1.10))
    lx, ly = x * dx, y * dy
    plt.text(lx, ly, m["name"], fontsize=9, ha="center")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Model Parameters (Billions, log scale)", fontsize=12)
plt.ylabel("FP16 Weights Memory (GB, log scale)", fontsize=12)
plt.title("Teacher vs Student LLMs: Parameters vs FP16 Weights Memory", fontsize=14)
plt.grid(True, which="both", linestyle="--", alpha=0.45)
plt.legend(title="Model Type", loc="upper left")
plt.tight_layout()
plt.savefig("../outputs/teacher_vs_student_llms_fp16weights.png", dpi=600)
plt.show()


