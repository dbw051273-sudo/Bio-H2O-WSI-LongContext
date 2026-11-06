"""
Generates Figure 0: High-Resolution System Workflow & Architecture Diagram for Bio-H2O WSI Harness
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_DIR = r"C:\Users\felix\.gemini\antigravity\scratch\wsi_long_context_project"
FIG_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def draw_workflow():
    fig, ax = plt.subplots(figsize=(13.6, 3.3), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 26)
    ax.axis('off')

    def draw_box(x, y, w, h, title, lines, fc="#F4F7FB", ec="#2B579A", title_fc="#2B579A"):
        # Main rounded box
        box = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.35,rounding_size=0.8",
            facecolor=fc, edgecolor=ec, linewidth=1.6
        )
        ax.add_patch(box)
        # Header banner
        header_h = 4.2
        header = mpatches.FancyBboxPatch(
            (x, y + h - header_h), w, header_h, boxstyle="round,pad=0.2,rounding_size=0.6",
            facecolor=title_fc, edgecolor=ec, linewidth=1.2
        )
        ax.add_patch(header)
        ax.text(x + w / 2, y + h - header_h / 2, title, color="white",
                fontsize=8.8, fontweight="bold", ha="center", va="center")
        # Body text
        body_text = "\n".join(lines)
        ax.text(x + w / 2, y + (h - header_h) / 2, body_text, color="#1E293B",
                fontsize=7.6, ha="center", va="center", linespacing=1.38)

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>,head_length=0.6,head_width=0.4", color="#334155", lw=1.6)
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.1, label,
                    fontsize=7.2, color="#0F172A", fontweight="bold", ha="center", va="bottom",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=0.8))

    # Stage 1: WSI Optical Preprocessing & Stain Deconvolution
    draw_box(
        1.0, 2.5, 20.5, 21.5,
        "① WSI 光学解卷积与网格序列化",
        [
            "• 物理标定: 20× 物镜 (MPP = 0.50 μm/px)",
            "• Beer-Lambert 光学密度变换:",
            "  OD = -log10(I / I0), I0 = 246",
            "• Ruifrok-Johnston H&E 染色分离:",
            "  提取苏木精核浓度 H_conc 空间分布",
            "• 2D-RoPE 旋转位置编码 (x_i, y_i)",
            "• 序列规模: N = 4,096 ~ 150万 Patches"
        ],
        fc="#EEF4FF", ec="#1D4ED8", title_fc="#1D4ED8"
    )

    # Stage 2: Bio-H2O Heavy-Hitter Oracle Router
    draw_box(
        25.5, 2.5, 21.5, 21.5,
        "② Bio-H2O 病理先验路由与淘汰器",
        [
            "• 稀疏全局探针: M=32 均匀锚点 Query",
            "• 融合评分函数 (Stain + Attention):",
            "  S_i = γ·A_probe(i) + (1-γ)·OD_H(i)",
            "  核心参数: γ = 0.55, 预算 B = 20%",
            "• 三元组 Token 动态分流:",
            "  1) 空间锚点 Sink Tokens (N_sink=16)",
            "  2) 高核质比病灶 Heavy-Hitters (HH)",
            "  3) 间质/空白背景 Evicted Tokens (80%)"
        ],
        fc="#FFF7ED", ec="#C2410C", title_fc="#C2410C"
    )

    # Stage 3: Dual-Stream Hybrid Attention (Top & Bottom Split)
    draw_box(
        51.0, 13.8, 22.5, 10.2,
        "③-A 精确稀疏 KV Cache 流 (20% Token)",
        [
            "• 保留 Sink + 恶性高核密 HH 键值对",
            "• 精确 Softmax 注意力保留微转移灶高频边界",
            "• 支持 FP16 / INT4 分组量化 (节省 80%+ 显存)"
        ],
        fc="#ECFDF5", ec="#047857", title_fc="#047857"
    )

    draw_box(
        51.0, 2.5, 22.5, 9.8,
        "③-B 线性核残差压缩流 (80% 淘汰 Token)",
        [
            "• 核特征映射: φ(x) = ELU(x) + 1",
            "• 恒定空间状态矩阵: S_lin = Σ φ(K_j)^T V_j",
            "• 复杂度 O(d²) (d=32), 零间质上下文丢失!"
        ],
        fc="#F0FDF4", ec="#15803D", title_fc="#15803D"
    )

    # Stage 4: Gated Fusion, Diagnosis & Task Resumption Harness
    draw_box(
        77.5, 2.5, 21.5, 21.5,
        "④ 门控融合、诊断输出与断点续接",
        [
            "• 双流自适应门控融合 (α = 0.85):",
            "  O = α·O_exact + (1-α)·O_linear",
            "• 多尺度病灶概率预测与残差审计图",
            "• 任务中断续接 (Task Resumption):",
            "  序列化存储 {Idx_keep, KV_kept,",
            "  S_lin_fp32, z_norm} (仅 426.9 KB)",
            "• 恢复耗时: 19.2 ms (提速 9.1× ~ 51×)"
        ],
        fc="#F5F3FF", ec="#6D28D9", title_fc="#6D28D9"
    )

    # Arrows connecting stages
    draw_arrow(21.7, 13.2, 25.3, 13.2, "Patch + OD")
    draw_arrow(47.2, 16.8, 50.8, 18.8, "Top-20% HH")
    draw_arrow(47.2, 9.5, 50.8, 7.4, "80% 间质")
    draw_arrow(73.7, 18.8, 77.3, 15.5, "O_exact")
    draw_arrow(73.7, 7.4, 77.3, 10.8, "O_linear")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig0_workflow_architecture.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved fig0_workflow_architecture.png successfully.")


if __name__ == "__main__":
    draw_workflow()
