# LLM Learning: 從零白盒解剖到後 Transformer 原創實驗的實戰路徑

本專案以 **Google Colab (NVIDIA L4 GPU)** 為雲端算力引擎，結合本地權重管理與 GitHub 版本控制，從最底層的隨機初始化權重（Pre-training）出發，不僅完整實作現代主流 LLM 四大訓練階段（Pretrain / SFT / LoRA / DPO），更在 **Stage 1 白盒沙盒** 中針對自迴歸 Transformer 的底層限制展開了 **5 輪後 Transformer 架構演進實驗（Diffusion LLM $\rightarrow$ Block Diffusion & 局部修圖 $\rightarrow$ 意念球量化與聯想 $\rightarrow$ 信心閘門雙腦與高知識密度架構）**。

📄 **完整研究報告與 Benchmark 對決總表**：請參閱 [`docs/stage1_research_journey_and_benchmarks.md`](docs/stage1_research_journey_and_benchmarks.md)

---

## 🗺️ 四階段實戰學習與研究路徑 (Learning & Research Roadmap)

```
[階段 1: 白盒拆解與架構突圍 (MiniMind 5.46M ~ 30.03M)]  ✅ 已完成！
   ├─ Part A (主流基線)：Pre-train (猜下一個字) -> Full SFT (-100 遮罩) -> LoRA (0.43% 便利貼微調) -> DPO (偏好對齊)
   └─ Part B (破框研究)：
        ├─ Lab 04: 雙向 Diffusion-MiniMind (LLaDA 架構，拍立得平行顯影，速度快 4~10 倍、克漏字 100%)
        ├─ Lab 05: Block-Diffusion + 動態擴充畫布 ([□] 佔位符自動停機) + 文字版 Photoshop 局部修圖 (修改/擴展/移除)
        ├─ Lab 06: 意念空間量化 (512 維球面座標) + 純向量聯想跳躍 + 單一意念雙風格平行映射 (快 14.23 倍)
        ├─ Lab 07: Entropy-Gated Dual-Brain (最弱字元信心閘門自動切換「肌肉記憶」與「意念觸類旁通」+ 熟能生巧固化)
        └─ Lab 08: 知識密度與智商大對決 (5.46M 迷你雙腦模型以 9.56 倍知識密度越級擊敗 30.03M 原版 AR/Diffusion)
          ↓
[階段 2: 工業級預訓練模型嫁接與對齊實戰 (Qwen2.5-0.5B)]  🔥 進行中！
   ├─ Lab 01: 微創雙腦嫁接大對決 (`Dual-Brain Qwen2.5-0.5B [86%]` vs `Qwen2.5-0.5B + 標準 LoRA [48%]`，未知系統零樣本推演 7/8 vs 0/8！)
   ├─ SFTTrainer + Chat Template 工業標準格式化
   └─ DPOTrainer 偏好對齊 + FlashAttention-2 / SDPA 顯存優化
          ↓
[階段 3: 外掛底層記憶庫 (RAG & Memory)]
   ├─ 零框架手寫 RAG (意念向量檢索 + FAISS)
   └─ 對話狀態管理器 (Sliding Window + Summary Buffer)
          ↓
[階段 4: 前沿極限 (高知識密度合成數據 & RL 推理)]
   ├─ Textbook 級別合成數據微調 (極限壓榨小模型容量)
   └─ Rule-based RL (GRPO) 激發長思維鏈 (<think>...</think>)
```

---

## 📊 Stage 1 核心實驗成果摘要 (Colab L4 實測)

| 編號 | 實驗主題 | Notebook 連結 | 本地備份權重 (`checkpoints/stage1_minimind/`) | 核心實測突破指標 (NVIDIA L4) |
| :---: | :--- | :--- | :--- | :--- |
| **01** | **Pre-training 預訓練** | [`01_minimind_pretrain_colab.ipynb`](notebooks/01_stage1_minimind/01_minimind_pretrain_colab.ipynb) | `pretrain_512.pth` (`66.6 MB`) | 30.03M 參數從隨機雜訊學中文語感，Loss `8.80 -> 6.04` |
| **02** | **Full SFT 指令微調** | [`02_minimind_sft_colab.ipynb`](notebooks/01_stage1_minimind/02_minimind_sft_colab.ipynb) | `full_sft_512.pth` (`66.6 MB`) | 驗證 `-100` Prompt Masking，學會多輪對話格式，Loss `5.40 -> 4.03` |
| **03** | **LoRA 與 DPO 對齊** | [`03_minimind_lora_dpo_colab.ipynb`](notebooks/01_stage1_minimind/03_minimind_lora_dpo_colab.ipynb) | `lora_identity_512.pth` (`0.27 MB`)<br/>`dpo_512.pth` (`66.6 MB`) | LoRA 僅動 `0.43%` 參數植入專屬記憶；DPO 實現人類偏好二選一對齊 |
| **04** | **Diffusion vs AR 對決** | [`04_diffusion_minimind_vs_ar_benchmark.ipynb`](notebooks/01_stage1_minimind/04_diffusion_minimind_vs_ar_benchmark.ipynb) | `diffusion_minimind_512.pth` (`60.1 MB`) | 拍立得平行顯影吞吐量達 `6,372 tok/s` (**快 10.36 倍**)，雙向克漏字 **100%** (vs AR 36.4%) |
| **05** | **Block Diffusion 與局部修圖** | [`05_block_diffusion_minimind_optimization.ipynb`](notebooks/01_stage1_minimind/05_block_diffusion_minimind_optimization.ipynb) | `block_diffusion_minimind_512.pth` (`60.1 MB`) | `8->16->24` 動態擴充 + `[□]` 自動收尾；支援局部修改/擴展/刪除縫合；2048 長文本 **快 4.24 倍** |
| **06** | **意念量化與連續聯想** | [`06_associative_thought_minimind.ipynb`](notebooks/01_stage1_minimind/06_associative_thought_minimind.ipynb) | `associative_thought_minimind_512.pth` (`70.1 MB`) | 思考過程 0 文字 Token，純 512 維球面座標跳躍 + 雙語氣自然映射，比 36 步 AR **快 14.23 倍** |
| **07** | **信心閘門雙腦與記憶固化** | [`07_entropy_gated_dual_brain_minimind.ipynb`](notebooks/01_stage1_minimind/07_entropy_gated_dual_brain_minimind.ipynb) | `entropy_dual_brain_minimind_512.pth` (`63.9 MB`) | 電話號碼 `99.2%` 信心走肌肉記憶 (100% 精準)；未知醫學題 `31.1%` 信心自動轉意念層零樣本推導，練習後固化為 `91.8%` 肌肉記憶 |
| **08** | **知識密度與智商大對決** | [`08_knowledge_density_iq_benchmark.ipynb`](notebooks/01_stage1_minimind/08_knowledge_density_iq_benchmark.ipynb) | `tiny_dual_brain_256.pth` (`11.0 MB`) | **`5.46M` 迷你雙腦模型** 在 4 大未知領域類比與 6 組死背號碼拿 **12/12 滿分**，知識密度達 `30M` 原版模型的 **9.56 倍**！ |
| **09** | **40題非飽和極限壓力測試** | [`09_non_saturated_stress_benchmark.ipynb`](notebooks/01_stage1_minimind/09_non_saturated_stress_benchmark.ipynb) | *(壓力極限測試套件)* | 打破滿分飽和！拉開 `Dual-Brain (68%)` vs `Pure-Thought LCM (40%)` vs `Pure-Diffusion (10%)` 差距，並找出 `5.46M` 臨界甜蜜點与 RAG 需求邊界 |

---

## 🚀 Stage 2 工業級預訓練模型嫁接與對決成果 (Qwen2.5-0.5B on Colab L4)

| 編號 | 實驗主題 | Notebook 連結 | 本地備份權重 (`checkpoints/stage2_qwen/`) | 核心實測突破指標 (NVIDIA L4) |
| :---: | :--- | :--- | :--- | :--- |
| **01** | **Qwen2.5-0.5B 微創雙腦嫁接 PK** | [`01_qwen25_vs_dual_brain_qwen_pk.ipynb`](notebooks/02_stage2_alignment/01_qwen25_vs_dual_brain_qwen_pk.ipynb) | `dual_brain_qwen25_05b_head.pth` (`41.6 MB`) | 100% 凍結 `Qwen2.5-0.5B` (494M)，嫁接 `896->128` 窄瓶頸正交意念漏斗與謙遜閘門；在 8 大未知系統零樣本推演拿下 **`7/8 (87.5%)`**，完勝標準 `Qwen2.5-0.5B + LoRA` 的 **`0/8 (0%)`**（總分 **86% vs 48%**）！ |
| **02** | **Mini Open-LLM Leaderboard 公開基準擂台** | [`02_mini_open_llm_leaderboard_mmlu_arc_gsm8k.ipynb`](notebooks/02_stage2_alignment/02_mini_open_llm_leaderboard_mmlu_arc_gsm8k.ipynb) | `dual_brain_qwen25_open_llm_cribnote.pth` (`4.06 MB`) | 實測 140 題官方 `MMLU + ARC-Challenge + GSM8K`：雙腦小抄版在 **`ARC-Challenge` 科學推理暴增 `+8.0%` (`46% -> 54%`)** 且 **`MMLU` 0% 遺忘 (`40%`)**；標準 LoRA 則嚴重災難性遺忘 (`MMLU 30%`, `GSM8K 崩潰至 5%`)！ |

---

## 🔬 開放研究議題追蹤 (Open Research Issues)

我們在 Stage 1 實驗中發掘出的 3 個後 Transformer 世代核心開放問題，已建立於 GitHub Issues 持續深入探索：
- 📌 **[Issue #1：無監督野生語料的「自動結構與意念對齊」](https://github.com/EndeavorYen/LLM-learning/issues/1)** *(Unsupervised Structural Alignment in the Wild)*
- 📌 **[Issue #2：多層次意念的「動態顆粒度伸縮」](https://github.com/EndeavorYen/LLM-learning/issues/2)** *(Dynamic & Hierarchical Thought Granularity)*
- 📌 **[Issue #3：超長聯想鏈的「腦波流形飄移與錨點校準」](https://github.com/EndeavorYen/LLM-learning/issues/3)** *(Latent Drift in Long-Horizon Thought Chains)*
- 📌 **[Issue #4：資訊瓶頸效應與高維度意念空間抗噪校準（為何 5.5M 小雙腦擊敗 32M 大雙腦？）](https://github.com/EndeavorYen/LLM-learning/issues/4)** *(Information Bottleneck & Dimensional Noise Scaling)*

---

## 🎯 各階段進度追蹤 (Progress Tracking)

| 階段 | 核心目標 | 實作重點 | 狀態 |
| :--- | :--- | :--- | :---: |
| **Stage 1** | 白盒拆解 MiniMind 與後 Transformer 架構突圍 | Pretrain/SFT/LoRA/DPO + Diffusion/Block-Diffusion/意念雙腦 8 大實驗 | ✅ **已完成** |
| **Stage 2** | 工業級對齊實戰 | Qwen2.5-0.5B + TRL SFT / DPO、顯存優化與架構移植探索 | 🟡 準備進入 |
| **Stage 3** | 外掛 RAG 與記憶 | FAISS 原生檢索、Sliding Window / Summary Memory | ⚪ 待開始 |
| **Stage 4** | 合成數據與 RL 推理 | Schema 結構化輸出、Rule-based RL 思維鏈 (`<think>`) | ⚪ 待開始 |
