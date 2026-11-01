# LLM Learning: 從零白盒解剖到後 Transformer 原創實驗的實戰路徑

> [!WARNING]
> **稽核完成（2026-11-01）：原始成果數據經修正後重跑，主要宣稱不成立；本頁以下的舊數字保留作為歷史紀錄。** 原始腳本補上後的程式碼稽核發現：
>
> | 宣稱 | 問題 | 追蹤 |
> | :--- | :--- | :--- |
> | 未知領域零樣本推演 100%（Stage 1 / 2） | 未知領域的答案句在解碼頭訓練資料中，實為「階段分類 + 查表」 | [#14](https://github.com/EndeavorYen/LLM-learning/issues/14) |
> | 5.46M 小雙腦知識密度 9.56 倍、Lab 09 臨界質量 | 建立在上述洩漏、AR 輸出長度被截斷、`SLOT_LEN` 截斷答案之上 | [#13](https://github.com/EndeavorYen/LLM-learning/issues/13) [#16](https://github.com/EndeavorYen/LLM-learning/issues/16) |
> | MMLU 0% 遺忘 / ARC +8% | MMLU 依任務標籤寫死退回原版；ARC 只對雙腦組加計分項 | [#15](https://github.com/EndeavorYen/LLM-learning/issues/15) |
> | 逆向克漏字 6/6 vs 0/6、AR 基線 0 分 | 對照組的評測協定讓它不可能答對；前 5 字比對過寬 | [#16](https://github.com/EndeavorYen/LLM-learning/issues/16) |
> | 速度 10.36x / 4.24x / 14.23x / 8.11x | 未在同等品質下量測或比較對象不對等 | [#12](https://github.com/EndeavorYen/LLM-learning/issues/12) |
>
> 修正後的重跑結果見 [`results/original_labs_rerun/`](results/original_labs_rerun/README.md)；完整的稽核與驗證總結見 **[`docs/audit_findings.md`](docs/audit_findings.md)**。
>
> **目前經事先登錄驗證的結論（2026-11-01）：**
> - **在概念層多步推理（聯想跳躍）有效**：8 個全新領域上，跳躍只看前兩句的 64 維狀態往後推 3 步，選對分支 **85%**（機率 50%，Qwen 0.5B 直接排序 54–58%）；需採兩段式訓練（先訓練漏斗並凍結，再訓練跳躍），原本的聯合訓練只有 59%（[#21](https://github.com/EndeavorYen/LLM-learning/issues/21)）。
> - **不死背、能泛化**：需要依序推理的狀態追蹤題，跳躍只用 1–4 步訓練，推到 10 步仍 99.8%；一次讀完全文的模型只剩 32–48%（見過的領域；[#23](https://github.com/EndeavorYen/LLM-learning/issues/23)）。
> - **瓶頸在「觸類旁通」**：換到新領域時，模型認不出新領域的句子對應哪個抽象概念（句子辨識 41–69%），所有方法都失敗；64 維漏斗也沒有比原始 896 維特徵更好或更穩（[#22](https://github.com/EndeavorYen/LLM-learning/issues/22)）。下一步專攻這一點。

本專案以 **Google Colab (NVIDIA L4 GPU)** 為雲端算力引擎，結合本地權重管理與 GitHub 版本控制，從最底層的隨機初始化權重（Pre-training）出發，不僅完整實作現代主流 LLM 四大訓練階段（Pretrain / SFT / LoRA / DPO），更在 **Stage 1 白盒沙盒** 中針對自迴歸 Transformer 的底層限制展開了 **5 輪後 Transformer 架構演進實驗（Diffusion LLM $\rightarrow$ Block Diffusion & 局部修圖 $\rightarrow$ 意念球量化與聯想 $\rightarrow$ 信心閘門雙腦與高知識密度架構）**。

📄 **完整研究報告與 Benchmark 對決總表**：
- 📘 **Stage 1 白盒拆解與後 Transformer 演進報告**：[`docs/stage1_research_journey_and_benchmarks.md`](docs/stage1_research_journey_and_benchmarks.md)
- 📗 **Stage 2 工業級 Qwen2.5-0.5B 微創雙腦嫁接、公開基準擂台與「三合一完整體」深度技術報告**：[`docs/stage2_qwen_grafting_and_trinity_analysis.md`](docs/stage2_qwen_grafting_and_trinity_analysis.md)

---

## 🗺️ 四階段實戰學習與研究路徑 (Learning & Research Roadmap)

```
[階段 1: 白盒拆解與架構突圍 (MiniMind 1.31M ~ 31.93M)]  ✅ 已完成！
   ├─ Part A (主流基線)：Pre-train (猜下一個字) -> Full SFT (-100 遮罩) -> LoRA (0.43% 便利貼微調) -> DPO (偏好對齊)
   └─ Part B (破框研究)：
        ├─ Lab 04: 雙向 Diffusion-MiniMind (LLaDA 架構，拍立得平行顯影，速度快 4~10 倍、克漏字 100%)
        ├─ Lab 05: Block-Diffusion + 動態擴充畫布 ([□] 佔位符自動停機) + 文字版 Photoshop 局部修圖 (修改/擴展/移除)
        ├─ Lab 06: 意念空間量化 (512 維球面座標) + 純向量聯想跳躍 + 單一意念雙風格平行映射 (快 14.23 倍)
        ├─ Lab 07: Entropy-Gated Dual-Brain (最弱字元信心閘門自動切換「肌肉記憶」與「意念觸類旁通」+ 熟能生巧固化)
        ├─ Lab 08: 知識密度與智商大對決 (5.46M 迷你雙腦模型以 9.56 倍知識密度越級擊敗 30.03M 原版 AR/Diffusion)
        └─ Lab 09: 40題非飽和極限壓力測試 (找出 5.46M 最小臨界質量與資訊瓶頸定律 Issue #4)
          ↓
[階段 2: 工業級預訓練模型微創嫁接與公開基準對決 (Qwen2.5-0.5B)]  ✅ 已完成！
   ├─ Lab 01: 微創雙腦嫁接對決 (`Dual-Brain Qwen2.5 [86%]` vs `標準 LoRA [48%]`，未知系統零樣本推演 7/8 vs 0/8)
   ├─ Lab 02: 國際公開基準擂台 (`MMLU` 0% 遺忘 + `ARC-Challenge` 科學推理暴增 +8.0% + 驗證單點壓縮 vs 多步算術 `GSM8K`)
   └─ Lab 03: 三合一完整體 `Full-Trinity` (凍結 Qwen 896D + 128D 意念軌道 + 2層雙向 Block-Diffusion：未知推演 8/8、逆向克漏字 6/6、快 8.11 倍、總分 96% vs 29%！)
          ↓
[階段 3: 野生語料自監督意念對齊 & 內建極致知識密度 (攻克 Issue #1 & Issue #4)]  🔜 下一步
   ├─ 無監督野生文本自動結構與意念對齊 (擺脫人工表格標籤，從普通文本自動聚類分離領域 d 與規律 s)
   └─ 內建稀疏微記憶槽 (Internal Sparse Memory，不靠場外作弊 RAG，在裸機內部以 O(1) 極低算力 Cost 接手海量死背常識，釋放 100% 推理容量)
          ↓
[階段 4: 動態多節點思維鏈 (s1 -> s2 -> ... -> sk) & 流形錨點校準 (攻克 Issue #2 & Issue #3)]
   ├─ 動態意念顆粒度伸縮 (將複雜多步數學題 GSM8K 自動拆解為多顆連續意念節點，一步一節點推演)
   └─ 超長聯想鏈之球面流形錨點校準 (Manifold Anchor Calibration，消除 5~10 步長鏈跳躍飄移)
```

---

## 📊 Stage 1 核心實驗成果摘要 (Colab L4 實測)

> ⚠️ **已重跑，下表為原始數字（歷史紀錄）**：受 [#13](https://github.com/EndeavorYen/LLM-learning/issues/13)–[#16](https://github.com/EndeavorYen/LLM-learning/issues/16) 的問題影響，修正後的重跑結果見 [`results/original_labs_rerun/`](results/original_labs_rerun/README.md)，主要宣稱不成立。

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

> ⚠️ **已重跑，下表為原始數字（歷史紀錄）**：受 [#13](https://github.com/EndeavorYen/LLM-learning/issues/13)–[#16](https://github.com/EndeavorYen/LLM-learning/issues/16) 的問題影響，修正後的重跑結果見 [`results/original_labs_rerun/`](results/original_labs_rerun/README.md)，主要宣稱不成立。

| 編號 | 實驗主題 | Notebook 連結 | 本地備份權重 (`checkpoints/stage2_qwen/`) | 核心實測突破指標 (NVIDIA L4) |
| :---: | :--- | :--- | :--- | :--- |
| **01** | **Qwen2.5-0.5B 微創雙腦嫁接 PK** | [`01_qwen25_vs_dual_brain_qwen_pk.ipynb`](notebooks/02_stage2_alignment/01_qwen25_vs_dual_brain_qwen_pk.ipynb) | `dual_brain_qwen25_05b_head.pth` (`41.6 MB`) | 100% 凍結 `Qwen2.5-0.5B` (494M)，嫁接 `896->128` 窄瓶頸正交意念漏斗與謙遜閘門；在 8 大未知系統零樣本推演拿下 **`7/8 (87.5%)`**，完勝標準 `Qwen2.5-0.5B + LoRA` 的 **`0/8 (0%)`**（總分 **86% vs 48%**）！ |
| **02** | **Mini Open-LLM Leaderboard 公開基準擂台** | [`02_mini_open_llm_leaderboard_mmlu_arc_gsm8k.ipynb`](notebooks/02_stage2_alignment/02_mini_open_llm_leaderboard_mmlu_arc_gsm8k.ipynb) | `dual_brain_qwen25_open_llm_cribnote.pth` (`4.06 MB`) | 實測 140 題官方 `MMLU + ARC-Challenge + GSM8K`：雙腦小抄版在 **`ARC-Challenge` 科學推理暴增 `+8.0%` (`46% -> 54%`)** 且 **`MMLU` 0% 遺忘 (`40%`)**；標準 LoRA 則嚴重災難性遺忘 (`MMLU 30%`, `GSM8K 崩潰至 5%`)！ |
| **03** | **三合一完整體 (Dual-Brain + Block-Diffusion + Qwen2.5)** | [`03_full_trinity_dual_brain_block_diffusion_qwen25.ipynb`](notebooks/02_stage2_alignment/03_full_trinity_dual_brain_block_diffusion_qwen25.ipynb) | `full_trinity_dual_brain_block_diff_qwen25.pth` (`73.8 MB`) | 補齊 2 層雙向 Block-Diffusion 顯影頭 (`m1 ∪ ~m1`)：未知系統零樣本推演 **`8/8 (100%)`**、逆向克漏字修圖 **`6/6 (100%)`**、推論速度 **快 `8.11 倍`**，24 題總分 **`96% vs 29%`** 輾壓標準 AR LoRA！ |

---

## 🔬 開放研究議題追蹤 (Open Research Issues)

我們在 Stage 1 實驗中發掘出的 3 個後 Transformer 世代核心開放問題，已建立於 GitHub Issues 持續深入探索：
- 🎯 **[Issue #25：B1 觸類旁通——把新領域的句子可靠地對到抽象概念](https://github.com/EndeavorYen/LLM-learning/issues/25)**（目前主線）
- 🗺️ **[Issue #30：下一階段規劃（Lab 17 補強辨識 → Lab 18 測試領域正式考試）與方向盤點](https://github.com/EndeavorYen/LLM-learning/issues/30)**，完整內容見 [`docs/plan_b1_next_phase.md`](docs/plan_b1_next_phase.md)
- 📌 **[Issue #1：無監督野生語料的「自動結構與意念對齊」](https://github.com/EndeavorYen/LLM-learning/issues/1)** *(Unsupervised Structural Alignment in the Wild)*
- 📌 **[Issue #2：多層次意念的「動態顆粒度伸縮」](https://github.com/EndeavorYen/LLM-learning/issues/2)** *(Dynamic & Hierarchical Thought Granularity)*
- 📌 **[Issue #3：超長聯想鏈的「腦波流形飄移與錨點校準」](https://github.com/EndeavorYen/LLM-learning/issues/3)** *(Latent Drift in Long-Horizon Thought Chains)*
- 📌 **[Issue #4：資訊瓶頸效應與高維度意念空間抗噪校準（為何 5.5M 小雙腦擊敗 32M 大雙腦？）](https://github.com/EndeavorYen/LLM-learning/issues/4)** *(Information Bottleneck & Dimensional Noise Scaling)* — **已結案：公平重跑後反常現象消失，「低維度比較抗噪」未獲支持**

---

## 🎯 各階段進度追蹤 (Progress Tracking)

| 階段 | 核心目標 | 實作重點 | 狀態 |
| :--- | :--- | :--- | :---: |
| **Stage 1** | 白盒拆解 MiniMind 與後 Transformer 架構突圍 | Pretrain/SFT/LoRA/DPO + Diffusion/Block-Diffusion/意念雙腦 9 大實驗 | ✅ **已完成** |
| **Stage 2** | 工業級預訓練模型微創嫁接與公開基準對決 | `Qwen2.5-0.5B` 嫁接 128D 意念腦 + 雙向 Block-Diffusion (`Full-Trinity 96% vs 29%`, `ARC +8%`) | ✅ **已完成** |
| **稽核與驗證** | 從測試集開始重建可信的驗證流程 | 修正洩漏 / 截斷 / 不公平對照、事先登錄的類比與聯想測試 v2–v5、拆解實驗、狀態追蹤前導（[總結](docs/audit_findings.md)） | ✅ **已完成** |
| **B1：觸類旁通** | 把新領域的句子可靠地對到抽象概念 | 句子 → 抽象事件 / 規律的跨領域辨識，目標單句 ≥ 95%，再回到新領域的多步推理測試 | 🔄 **進行中**：診斷最好 83.7%（[v1](results/recognition_v1/README.md)）；新領域給一個例子 92.7%（[v2](results/recognition_v2/README.md)），尚未達 95% |
| **Stage 3** | 野生語料自監督意念對齊 & 內建極致知識密度 | 攻克 [Issue #1](https://github.com/EndeavorYen/LLM-learning/issues/1) & [Issue #4](https://github.com/EndeavorYen/LLM-learning/issues/4)：免人工標籤對比聚類 + 內建 $O(1)$ 稀疏微記憶槽（提升裸機密度、降低 Cost） | 🔜 **準備進入** |
| **Stage 4** | 動態多節點思維鏈 & 流形錨點校準 | 攻克 [Issue #2](https://github.com/EndeavorYen/LLM-learning/issues/2) & [Issue #3](https://github.com/EndeavorYen/LLM-learning/issues/3)：多步數學/邏輯動態節點拆解 ($s_1 \to \dots \to s_k$) + 長鏈防飄移校準 | ⚪ 待開始 |
