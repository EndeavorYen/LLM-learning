# LLM Learning: 從零到前沿的 LLM 實戰學習路徑

本專案專為 **LLM 開發新手** 設計，以 **Google Colab (單卡 GPU: T4 / L4 / A100)** 為核心訓練與實驗環境，搭配本地代碼管理與 Git 版本控制。

從最底層的隨機初始化權重（Pre-training）開始，逐步推進至現代工業級對齊、原生 RAG、長短期記憶管理，以及前沿的思維鏈強化學習（RL Reasoning）。

---

## 🗺️ 四階段實戰學習路徑 (Learning Roadmap)

```
[階段 1: 白盒拆解 (MiniMind)] 
   ├─ 自製 BPE Tokenizer
   ├─ 從隨機權重 Pre-train (26M / 108M)
   └─ 手寫 SFT Loss Masking (-100) 與 DPO 數學原理
          ↓
[階段 2: 工業級對齊 (Qwen2.5 / SmolLM)]
   ├─ 使用 Hugging Face TRL 函式庫
   ├─ SFTTrainer + Chat Template 格式化
   └─ DPOTrainer 偏好對齊 + FlashAttention-2 / SDPA 顯存優化
          ↓
[階段 3: 外掛能力 (RAG & Memory)]
   ├─ 零框架手寫 RAG (Sentence-Transformers + FAISS)
   ├─ Context 注入 Prompt 與截斷策略
   └─ 對話狀態管理器 (Sliding Window + Summary Buffer)
          ↓
[階段 4: 前沿極限 (合成數據 & RL 推理)]
   ├─ Textbook 級別合成數據微調 (極限壓榨小模型容量)
   └─ Rule-based RL (GRPO / REINFORCE) 激發小模型長思維鏈 (<think>...</think>)
```

---

## 📂 專案結構 (Repository Structure)

```plaintext
LLM-learning/
├── README.md                   # 專案總覽與學習進度
├── .gitignore                  # 忽略大權重檔與暫存資料
├── notebooks/                  # Colab 可直接一鍵開啟的實戰 Notebooks
│   ├── 01_stage1_minimind/     # 第一階段：白盒拆解 (MiniMind 26M/108M)
│   │   ├── 01_colab_pretrain.ipynb
│   │   └── 02_colab_sft_dpo.ipynb
│   ├── 02_stage2_alignment/    # 第二階段：工業級對齊 (Qwen2.5-0.5B + TRL)
│   ├── 03_stage3_rag_memory/   # 第三階段：原生 RAG 與記憶系統
│   └── 04_stage4_reasoning/    # 第四階段：合成數據與 RL 思維鏈
├── src/                        # 可重用的 Python 模組 (模型定義、資料處理、RAG 核心)
│   ├── models/
│   ├── utils/
│   └── rag/
└── requirements.txt            # 本地 / Colab 共用依賴套件
```

---

## 🎯 各階段進度追蹤 (Progress Tracking)

| 階段 | 核心目標 | 實作重點 | 狀態 |
| :--- | :--- | :--- | :---: |
| **Stage 1** | 白盒拆解 MiniMind | Pre-train 迴圈、張量維度追蹤、SFT -100 Masking | 🟡 準備中 |
| **Stage 2** | 工業級對齊 | Qwen2.5-0.5B + TRL SFT / DPO、顯存優化 | ⚪ 待開始 |
| **Stage 3** | 外掛 RAG 與記憶 | FAISS 原生檢索、Sliding Window / Summary Memory | ⚪ 待開始 |
| **Stage 4** | 合成數據與 RL 推理 | Schema 結構化輸出、Rule-based RL 思維鏈 (<think>) | ⚪ 待開始 |

---

## 🚀 Colab 環境注意事項

1. **檔案保存 (Drive Mount)**：
   Colab 離線會清空 `/content` 磁碟，訓練過程中 Checkpoint 與 Tokenizer 必須同步備份至掛載的 Google Drive (`/content/drive/MyDrive/...`)。
2. **I/O 速度優化**：
   訓練語料請先下載並解壓縮至 Colab 本地磁碟 (`/content/data`) 再讀取，避免直接在 Google Drive 上讀寫零碎小檔造成 GPU 等待 I/O。
3. **無縫同步 GitHub**：
   在 Colab 上完成訓練測試後，相關腳本與筆記本可直接同步回本 GitHub 倉庫。
