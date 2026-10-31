# 實驗可復現性、原始腳本索引、資料集規範與 Loss / 門檻設計誠實揭露報告 (Reproducibility & Design Specification)

本文件針對 Code & Research Review 提出的四項核心稽核要求（**#5 原始程式碼與執行紀錄**、**#7 訓練與評測資料集**、**#8 漏斗 Loss 公式與正負樣本取樣**、**#10 `85%` 與 `35%` 門檻設定與測試集調參揭露**），提供 100% 原始無修改之對照紀錄與誠實揭露。

---

## 1. 原始執行程式碼索引（對應 Issue #5）

> **[稽核更新 #13 #17]** `scripts/` 已依稽核修正長度截斷、寫死字串與未固定版本等問題，下表行號對應**修正後**的腳本。修正前、從對話紀錄逐字擷取的原始版本保留在 commit `684a82a`（分支 [`archive/original-scripts`](https://github.com/EndeavorYen/LLM-learning/tree/archive/original-scripts/scripts)）。

所有在對話過程中實際送入 Colab L4 執行之原始 Python 腳本，已從系統底層對話紀錄（`transcript_full.jsonl`）透過 AST 原樣提取至 [`scripts/`](../scripts/) 目錄（**零遺失、零事後重寫**）：

| 階段與實驗 | 原始執行腳本路徑 (`scripts/`) | Transcript Step | 原始 Colab Session 名稱 | 包含之關鍵類別 / 函式與行號 |
| :--- | :--- | :---: | :---: | :--- |
| **Stage 1 Lab 01** (Pretrain) | [`scripts/stage1_lab01_pretrain.py`](../scripts/stage1_lab01_pretrain.py) | `Step 228` | `stage1` | `MiniMindForCausalLM` Pretrain 迴圈 |
| **Stage 1 Lab 02** (Full SFT) | [`scripts/stage1_lab02_sft.py`](../scripts/stage1_lab02_sft.py) | `Step 243` | `stage1_sft` | SFT `-100` Prompt Masking 訓練迴圈 |
| **Stage 1 Lab 03** (LoRA & DPO) | [`scripts/stage1_lab03_lora_dpo.py`](../scripts/stage1_lab03_lora_dpo.py) | `Step 263` | `stage1_lora_dpo` | `apply_lora`, `dpo_loss` 訓練與驗證 |
| **Stage 1 Lab 04** (Diffusion vs AR) | [`scripts/stage1_lab04_diffusion_vs_ar.py`](../scripts/stage1_lab04_diffusion_vs_ar.py) | `Step 288` | `stage1_diff_v2` | `is_causal = False`, 雙向遮罩去噪與克漏字對決 |
| **Stage 1 Lab 05** (Block Diffusion) | [`scripts/stage1_lab05_block_diffusion.py`](../scripts/stage1_lab05_block_diffusion.py) | `Step 310` | `stage1_block_diff_v2` | `SLOT_PAD_ID=4`, 互補遮罩 `m1 ∪ ~m1`, 局部文字修圖 |
| **Stage 1 Lab 06** (Associative Thought) | [`scripts/stage1_lab06_associative_thought.py`](../scripts/stage1_lab06_associative_thought.py) | `Step 339` | `stage1_thought_assoc_v3` | `AssociativeThoughtMiniMind`, `encode_thought`, `forward_thought_hop` |
| **Stage 1 Lab 07** (Entropy-Gated Dual-Brain) | [`scripts/stage1_lab07_entropy_gated_dual_brain.py`](../scripts/stage1_lab07_entropy_gated_dual_brain.py) | `Step 385` | `stage1_entropy_dual_brain_v3` | **`EntropyGatedDualBrainMiniMind`** (L30), `forward_fast_muscle_memory` (L62), `encode_thought` (L69), `forward_struct_hop` (L78), `solve_via_thought_analogy` (L126), `consolidate_to_muscle_memory` (L143) |
| **Stage 1 Lab 08** (12題知識密度對決) | [`scripts/stage1_lab08_knowledge_density_12q.py`](../scripts/stage1_lab08_knowledge_density_12q.py) | `Step 397` | `stage1_iq_density_bench` | `PureDiffusionMiniMind` (L126), `DualBrainMiniMind` (L161), 5模型訓練與12題評測 |
| **Stage 1 Lab 09** (40題非飽和極限壓測) | [`scripts/stage1_lab09_non_saturated_stress_40q.py`](../scripts/stage1_lab09_non_saturated_stress_40q.py) | `Step 418` | `stage1_stress_bench` | `StressDualBrainModel` (L177), 6模型訓練與 Tier 1~4 共40題壓測 |
| **Stage 2 Lab 01 v1** (初版 `bfloat16` 無謙遜閘門) | [`scripts/stage2_lab01_v1_bfloat16_no_humility.py`](../scripts/stage2_lab01_v1_bfloat16_no_humility.py) | `Step 451` | `stage2_qwen_pk` | `LoRALinear` (L113), **`DualBrainGraftedQwen`** (L185), **`domain_funnel`** (L194), **`struct_funnel`** (L195), **`law_brain`** (L199), **`extract_ds`** (L216), **`hop_law`** (L221) |
| **Stage 2 Lab 01 v2** (進階 `float32` + 謙遜閘門) | [`scripts/stage2_lab01_v2_float32_with_humility.py`](../scripts/stage2_lab01_v2_float32_with_humility.py) | `Step 458` | `stage2_qwen_pk_v2` | 同上 + **謙遜閘門正則化 `loss_humility`** (L259-264) |
| **Stage 2 Lab 02 v1** (150題 4選1 LogProb 初版) | [`scripts/stage2_lab02_v1_logprob_150q.py`](../scripts/stage2_lab02_v1_logprob_150q.py) | `Step 484` | `stage2_open_llm_leaderboard` | `DualBrainCribNoteQwen` (L231), `reasoning_router` (L262) |
| **Stage 2 Lab 02 v2** (140題標準 MC + 生成式 GSM8K) | [`scripts/stage2_lab02_v2_authentic_140q.py`](../scripts/stage2_lab02_v2_authentic_140q.py) | `Step 488` | `stage2_open_llm_authentic` | `DualBrainCribNoteQwen` (L237), `make_crib_note` (L264), 端到端 `loss_crib_lm` (L337) |
| **Stage 2 Lab 03** (三合一完整體 Full-Trinity) | [`scripts/stage2_lab03_full_trinity_24q.py`](../scripts/stage2_lab03_full_trinity_24q.py) | `Step 512` | `stage2_full_trinity_qwen_v2` | **`FullTrinityBlockDiffusionQwen`** (L176), `bidirectional_denoiser` (L198), `denoise_block` (L211), `iterative_block_generate` (L231) |

---

## 2. 訓練資料與評測 Benchmark 完整檔案索引（對應 Issues #5, #7）

我們已調整 [`.gitignore`](../.gitignore)（加入 `!benchmarks/` 與 `!benchmarks/**`），並將所有訓練資料、評測題目、標準答案與公開資料集抽樣規範完整存放於 [`benchmarks/`](../benchmarks/) 目錄：

1. **[`benchmarks/stage1_lab04_06_datasets.json`](../benchmarks/stage1_lab04_06_datasets.json)**：Stage 1 Lab 04–06 之訓練子句、雙向克漏字題目、局部文字修圖（修改/擴展/移除）與雙風格（詳細/精簡）意念鏈完整文本。
2. **[`benchmarks/stage1_lab07_dual_brain_dataset.json`](../benchmarks/stage1_lab07_dual_brain_dataset.json)**：Stage 1 Lab 07 之 6 組肌肉記憶問答 + `3 領域 × 4 階段` 雙風格因果句。
3. **[`benchmarks/stage1_lab08_12q_iq_density_benchmark.json`](../benchmarks/stage1_lab08_12q_iq_density_benchmark.json)**：Stage 1 Lab 08 之 6 組死背號碼 + `6 領域 × 4 階段 = 24` 句因果句 + **12 道評測題目與完整標準答案**。
4. **[`benchmarks/stage1_lab09_40q_stress_benchmark.json`](../benchmarks/stage1_lab09_40q_stress_benchmark.json)**：Stage 1 Lab 09 之 16 組高混淆單字元差異序號 + 4 組領域急救碼 + `6 領域 × 2 規律 × 4 階段 = 48` 句網格 + **40 道非飽和評測題目（Tier 1: 16題、Tier 2: 8題、Tier 3: 8題、Tier 4: 8題）與完整標準答案**。
5. **[`benchmarks/stage2_lab01_03_54_train_and_21q_24q_eval.json`](../benchmarks/stage2_lab01_03_54_train_and_21q_24q_eval.json)**：Stage 2 Lab 01 與 Lab 03 之 **54 句訓練資料**（6 組機密號碼 + `6 領域 × 2 規律 × 4 階段 = 48` 句網格，並列出 Competitor A `sft_texts` 的 34 條拼接字串）+ **Lab 01 的 21 題完整題目與答案** + **Lab 03 的 24 題完整題目與答案（含 6 題逆向克漏字遮罩位置）**。
6. **[`benchmarks/stage2_lab02_mmlu_arc_gsm8k_140q_sampling.json`](../benchmarks/stage2_lab02_mmlu_arc_gsm8k_140q_sampling.json)**：Stage 2 Lab 02 之 `MMLU`、`ARC-Challenge`、`GSM8K` 140 題抽樣規範：
   - **Random Seed**：`random.seed(42)`、`torch.manual_seed(42)`。
   - **抽樣方式（確定性前綴切片 Deterministic Slice，無隨機洗牌）**：
     - **`MMLU (50題)`**：`load_dataset("cais/mmlu", "all", split="test[:100]")` 取前 50 題 `list(ds_mmlu_test)[:50]`。
     - **`ARC-Challenge (50題)`**：
       - 訓練集：`load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train[:200]")` 過濾 `len(x['choices']['text']) == 4` 後取前 100 題。
       - 測試集：`load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test[:100]")` 過濾 `len(x['choices']['text']) == 4` 後取前 50 題。
     - **`GSM8K (40題)`**：
       - 訓練集：`load_dataset("openai/gsm8k", "main", split="train[:200]")` 取前 100 題（LoRA SFT 使用前 100 題，Crib-Note LM 訓練使用前 40 題）。
       - 測試集：`load_dataset("openai/gsm8k", "main", split="test[:50]")` 取前 40 題。

---

## 3. 設計說明：漏斗完整 Loss 公式、權重、正負樣本取樣與門檻設定揭露（對應 Issues #8, #10）

### 3.1 漏斗與雙腦的完整 Loss 數學公式（以 Stage 2 Lab 01 v2 與 Lab 03 為基準）

設凍結之 `Qwen/Qwen2.5-0.5B` 第 24 層平均池化輸出特徵為 $h \in \mathbb{R}^{896}$（`float32`）。

#### (1) 正交意念漏斗提取 (`extract_ds`)
$$d = \frac{\text{MLP}_{\text{domain}}(h)}{\|\text{MLP}_{\text{domain}}(h)\|_2} \in \mathbb{R}^{64}, \quad s = \frac{\text{MLP}_{\text{struct}}(h)}{\|\text{MLP}_{\text{struct}}(h)\|_2} \in \mathbb{R}^{64}$$
其中 $\text{MLP}_{\text{domain}}$ 與 $\text{MLP}_{\text{struct}}$ 結構皆為 `Linear(896, 256) -> GELU() -> Linear(256, 64)`。

#### (2) 正負樣本取樣與正交解耦損失 $\mathcal{L}_{\text{dis}}$ (`loss_dis`)
將 48 句單句（6 個領域 $\times$ 2 條規律 $\times$ 4 個階段）依序排列並 reshape 為三維張量：
$$D_{\text{grid}} \in \mathbb{R}^{6 \times 2 \times 4 \times 64}, \quad S_{\text{grid}} \in \mathbb{R}^{6 \times 2 \times 4 \times 64}$$

- **領域正樣本（Domain Positives）**：以每個領域 $i \in \{0\dots 5\}$ 的第一句 $d_{i,0,0}$ 為錨點（Anchor，`d_anchor = d_grid[:, 0, 0, :]`），拉近該領域下全部 $2 \times 4 = 8$ 個子句的領域向量：
  $$\mathcal{L}_{d\text{-pos}} = \frac{1}{6 \times 2 \times 4} \sum_{i=0}^{5}\sum_{l=0}^{1}\sum_{t=0}^{3} \left(1 - d_{i,l,t} \cdot d_{i,0,0}\right)$$
- **領域負樣本（Domain Negatives）**：取 6 個領域錨點兩兩之間的 Gram 矩陣 $G_d = d_{\text{anchor}} d_{\text{anchor}}^\top \in \mathbb{R}^{6 \times 6}$ 的所有非對角線元素（共 $6 \times 5 = 30$ 對跨領域負樣本），強制餘弦相似度小於 $-0.1$：
  $$\mathcal{L}_{d\text{-neg}} = \frac{1}{30} \sum_{i \neq j} \text{ReLU}\left(d_{i,0,0} \cdot d_{j,0,0} + 0.1\right)$$
- **結構規律正樣本（Structure Positives）**：以第 0 個領域（水利管網）的 $2 \times 4 = 8$ 個結構狀態 $s_{0,l,t}$ 為錨點（`s_anchor = s_grid[0, :, :, :]`），拉近所有 6 個領域在相同規律 $l$、相同階段 $t$ 的結構向量：
  $$\mathcal{L}_{s\text{-pos}} = \frac{1}{6 \times 2 \times 4} \sum_{i=0}^{5}\sum_{l=0}^{1}\sum_{t=0}^{3} \left(1 - s_{i,l,t} \cdot s_{0,l,t}\right)$$
- **結構規律負樣本（Structure Negatives）**：將 8 個結構錨點展平為 $S_{\text{flat8}} \in \mathbb{R}^{8 \times 64}$，取其 Gram 矩陣 $G_s = S_{\text{flat8}} S_{\text{flat8}}^\top \in \mathbb{R}^{8 \times 8}$ 的所有非對角線元素（共 $8 \times 7 = 56$ 對不同階段/不同規律之負樣本），強制餘弦相似度小於 $-0.1$：
  $$\mathcal{L}_{s\text{-neg}} = \frac{1}{56} \sum_{u \neq v} \text{ReLU}\left(S_{\text{flat8}, u} \cdot S_{\text{flat8}, v} + 0.1\right)$$
- **總解耦損失**：
  $$\mathcal{L}_{\text{dis}} = \mathcal{L}_{d\text{-pos}} + \mathcal{L}_{d\text{-neg}} + \mathcal{L}_{s\text{-pos}} + \mathcal{L}_{s\text{-neg}}$$

> ⚠️ **重要設計限制揭露（Oracle Grid Reshaping）**：
> 在上述 `d_vec.view(6, 2, 4, 64)` 與 `s_vec.view(6, 2, 4, 64)` 中，雖然**第 3 項損失 $\mathcal{L}_{\text{hop}}$（節點跳躍）只使用前 2 個已見領域 `s_grid[:2]`**，但**第 2 項損失 $\mathcal{L}_{\text{dis}}$（正交漏斗對齊）與第 4 項損失 $\mathcal{L}_{\text{diff}}$（單句顯影）使用了全部 48 句單句在 `6×2×4` 網格中的位置索引**（即模型在訓練漏斗時，已知哪個單句屬於哪個領域的第幾階段，但未知領域從未進入 `hop_law` 訓練）。這正是我們在 [GitHub Issue #1](https://github.com/EndeavorYen/LLM-learning/issues/1) 明確指出的限制：必須在 Stage 3 改為無監督聚類，才能擺脫人工 `6×2×4` 網格索引。

#### (3) 意念節點跳躍損失 $\mathcal{L}_{\text{hop}}$ (`loss_hop`)
僅取前 2 個已見領域（水利管網、高速交通）共 $2 \times 2 = 4$ 條軌跡 $S_{\text{seen}} \in \mathbb{R}^{4 \times 4 \times 64}$。將前 3 個階段 $S_{\text{seen}}[:, 0:3, :]$ 輸入因果遮罩 Transformer `hop_law` 預測下一階段座標 $\hat{S}[:, 1:4, :]$，並對目標端做 `detach()` 防止目標坍縮：
$$\mathcal{L}_{\text{hop}} = \frac{1}{4 \times 3} \sum_{b=0}^{3}\sum_{t=1}^{3} \left(1 - \hat{s}_{b,t} \cdot \text{sg}(s_{b,t})\right)$$

#### (4) 系統一快思損失 $\mathcal{L}_{\text{s1}}$ 與謙遜閘門損失 $\mathcal{L}_{\text{humility}}$
- **快思記憶損失 $\mathcal{L}_{\text{s1}}$**：對 10 個已見提示（6 組機密問答 + 4 組已見系統前提）計算 12 個格子的 Cross-Entropy。
- **未知領域謙遜損失 $\mathcal{L}_{\text{humility}}$（OOD Humility Gate）**：
  每一訓練步從 48 個單句特徵中隨機抽取 8 個向量 $h_{\text{perm}}$（注意：單句特徵長度與雙句拼接前提不同，且加入比例為 $0.35$ 的高斯雜訊），並縮放至與正樣本相同的平均範數：
  $$h_{\text{ood}} = \frac{h_{\text{perm}} + 0.35 \epsilon}{\|h_{\text{perm}} + 0.35 \epsilon\|_2} \cdot \overline{\|h_{\text{rote}}\|_2}, \quad \epsilon \sim \mathcal{N}(0, I)$$
  將 $h_{\text{ood}}$ 輸入系統一快思頭，取前 4 個格子的最大預測機率 $p_{\max} = \max_v \text{Softmax}(\text{logits}_{\text{ood}})$，懲罰超過 $0.35$ 的過度自信：
  $$\mathcal{L}_{\text{humility}} = \text{Mean}\left(\text{ReLU}(p_{\max} - 0.35)\right)$$

#### (5) 系統二語言渲染 / 雙向互補遮罩擴散損失 $\mathcal{L}_{\text{diff}}$
- 在 **Lab 01** 中為單步無遮罩 Cross-Entropy $\mathcal{L}_{\text{s2}}$。
- 在 **Lab 03 (`Full-Trinity`)** 中為 **全遮罩初稿損失 + 互補隨機遮罩損失 (`m1` 與 `~m1`)**：
  $$\mathcal{L}_{\text{diff}} = \mathcal{L}_{\text{CE}}(\text{FullMask}) + 0.5 \left(\mathcal{L}_{\text{CE}}(m_1) + \mathcal{L}_{\text{CE}}(\neg m_1)\right), \quad m_1 \sim \text{Bernoulli}(0.5)^{48 \times 12}$$

#### (6) 總目標函數與各項權重係數
$$\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{s1}} + 2.0 \cdot \mathcal{L}_{\text{humility}} + 1.0 \cdot \mathcal{L}_{\text{diff}} + 2.5 \cdot \mathcal{L}_{\text{hop}} + 0.8 \cdot \mathcal{L}_{\text{dis}}$$

---

### 3.2 關於 `85%` 與 `35%` 兩個門檻怎麼定的？有沒有用測試集調過？（對應 Issue #10 的誠實揭露）

1. **`85%` 路由門檻（`MinConf >= 0.85`）：未在 Stage 2 測試集上調參（沿用自 Stage 1 Lab 07 的先驗啟發值）**
   - `0.85` 這個門檻最早是在 **Stage 1 Lab 07（`Step 377`，2026-10-30T06:16:08Z）** 設計 `EntropyGatedDualBrainMiniMind` 時設定的先驗整數門檻（代表「12 個平行格子中，最沒把握的那一格信心也必須 $\ge 85\%$ 才算背熟」）。
   - 在那之後的 Stage 1 Lab 08、Stage 2 Lab 01 v1、Lab 01 v2、Lab 03 中，我們**從未修改過 `0.85` 這個數字**。

2. **`35%` 謙遜門檻（`F.relu(max_conf_ood - 0.35)`）：有根據 Stage 2 Lab 01 v1 的測試集輸出進行事後修正！（100% 誠實揭露）**
   - 在 **Stage 2 Lab 01 第一輪執行（[`scripts/stage2_lab01_v1_bfloat16_no_humility.py`](../scripts/stage2_lab01_v1_bfloat16_no_humility.py)，`Step 451`）** 時，我們**原本沒有寫 `loss_humility`**，且 `dual_head` 使用 `bfloat16`。
   - 當時跑完 21 題測試集後，我們從測試集 Log 觀察到：8 道未知系統題中，有 2 道題（`反物質約束環-規律2` 的 `MinConf=86.7%` 與 `星際曲速引擎-規律1` 的 `MinConf=96.5%`）因為超過了 `85%` 門檻，被誤判進入系統一快思頭，吐出了錯誤的水管句子（當時未知系統得分為 `5/8`，總分為 `16/21 = 76%`）。
   - **正因為看到了第一輪測試集上的這 2 個 False Positives**，我們才在 **第二輪（[`scripts/stage2_lab01_v2_float32_with_humility.py`](../scripts/stage2_lab01_v2_float32_with_humility.py)，`Step 458`）** 新增了 `loss_humility = F.relu(max_conf_ood - 0.35).mean()` 並將 `dual_head` 改為 `float32`，才使未知系統得分從 `5/8` 提升至 `7/8`（總分 `18/21 = 86%`）。
   - 我們將 `v1`（無謙遜閘門，`16/21`）與 `v2`（加謙遜閘門，`18/21`）兩支原始腳本**全部保留在 `scripts/` 目錄中**，供審閱者直接比對。

## 4. 稽核後新增的驗證實驗（2026-10-31 ~ 11-01）

稽核後的驗證實驗（類比 v2–v5、消融、跳躍可學性、空間 × 推理拆解、狀態追蹤前導）各有獨立腳本、固定 sha 的資料、3 個 seed 的紀錄與事先登錄的規格，索引見 [`audit_findings.md`](audit_findings.md) 的「檔案索引」。每個腳本都可用 `AUDIT_SEED=42 python scripts/<腳本>.py` 重跑（Colab L4；資料路徑會自動尋找 `/content/` 或 `benchmarks/`）。
