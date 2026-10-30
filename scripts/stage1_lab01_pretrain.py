# ==============================================================================
# ORIGINAL EXECUTED SCRIPT EXTRACTED VERBATIM FROM TRANSCRIPT
# Transcript Step Index : 228
# Timestamp             : 2026-10-29T03:07:32Z
# ==============================================================================

import os, sys
from colab_cli.common import state
from colab_cli.runtime import ColabRuntime
from colab_cli.contents import ContentsClient

state.sync_sessions()
s = state.get_session('stage1')
print(f"[colab] Connected to {s.endpoint} (kernel: {s.kernel_id})", flush=True)

rt = ColabRuntime(s.url, s.token, kernel_id=s.kernel_id, session_id=None)

remote_code = """
import os, sys, subprocess, torch
sys.path.insert(0, '/content/minimind')

print("=== [Step 6.1] Downloading Pre-training Dataset ===", flush=True)
os.makedirs('/content/minimind/dataset', exist_ok=True)
full_data_path = '/content/minimind/dataset/pretrain_t2t_mini.jsonl'
fast_data_path = '/content/minimind/dataset/pretrain_fast_10k.jsonl'

if not os.path.exists(full_data_path):
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="jingyaogong/minimind_dataset",
        filename="pretrain_t2t_mini.jsonl",
        repo_type="dataset",
        local_dir="/content/minimind/dataset"
    )

if not os.path.exists(fast_data_path):
    with open(full_data_path, 'r', encoding='utf-8') as fin, open(fast_data_path, 'w', encoding='utf-8') as fout:
        for i in range(10000):
            line = fin.readline()
            if not line:
                break
            fout.write(line)

with open(fast_data_path, 'r', encoding='utf-8') as f:
    sample_line = f.readline().strip()
print(f"Dataset ready! Sample 1: {sample_line[:100]}...", flush=True)

print("\\n=== [Before Pre-training] Random Untrained Model Output ===", flush=True)
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
tokenizer = AutoTokenizer.from_pretrained('/content/minimind/model')
untrained_model = MiniMindForCausalLM(MiniMindConfig(hidden_size=512, num_hidden_layers=8, use_moe=False)).half().eval().cuda()
torch.manual_seed(42)
for p in ["为什么天空是蓝色的", "解释什么是机器学习"]:
    inp = tokenizer(tokenizer.bos_token + p, return_tensors="pt").to("cuda")
    out = untrained_model.generate(inp["input_ids"], max_new_tokens=30, do_sample=True, temperature=0.85, top_p=0.95)
    ans = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True)
    print(f"Prompt: {p}\\nUntrained Output: {ans}\\n", flush=True)
del untrained_model
torch.cuda.empty_cache()

print("=== [Step 6.2] Running MiniMind Pre-training (26M Params, 10,000 samples) ===", flush=True)
cmd = [
    sys.executable, "-u", "train_pretrain.py",
    "--data_path", "../dataset/pretrain_fast_10k.jsonl",
    "--hidden_size", "512",
    "--num_hidden_layers", "8",
    "--epochs", "1",
    "--batch_size", "32",
    "--accumulation_steps", "2",
    "--learning_rate", "5e-4",
    "--log_interval", "50",
    "--save_interval", "150"
]
proc = subprocess.Popen(cmd, cwd="/content/minimind/trainer", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()

print("\\n=== [Step 8] After Pre-training: Trained Model Generation ===", flush=True)
trained_model = MiniMindForCausalLM(MiniMindConfig(hidden_size=512, num_hidden_layers=8, use_moe=False))
trained_model.load_state_dict(torch.load('/content/minimind/out/pretrain_512.pth', map_location='cuda'), strict=True)
trained_model = trained_model.half().eval().cuda()
torch.manual_seed(42)
for p in ["为什么天空是蓝色的", "解释什么是机器学习", "人工智能的发展"]:
    inp = tokenizer(tokenizer.bos_token + p, return_tensors="pt").to("cuda")
    out = trained_model.generate(inp["input_ids"], max_new_tokens=40, do_sample=True, temperature=0.85, top_p=0.95)
    ans = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True)
    print(f"Prompt: {p}\\nTrained Output: {ans}\\n", flush=True)
"""

def on_out(o):
    if o.get("output_type") == "stream":
        print(o.get("text", ""), end="", flush=True)
    elif o.get("output_type") == "error":
        print("ERROR:", o.get("ename"), o.get("evalue"), flush=True)

rt.execute_code(remote_code, output_hook=on_out, timeout=600.0)

print("\n=== [Step 7] Downloading pretrain_512.pth to Local Disk ===", flush=True)
local_ckp = r"D:\Code\LLM-learning\checkpoints\stage1_minimind\pretrain_512.pth"
os.makedirs(os.path.dirname(local_ckp), exist_ok=True)
contents = ContentsClient(s)
contents.download("/content/minimind/out/pretrain_512.pth", local_ckp)
print(f"Downloaded checkpoint to {local_ckp} ({os.path.getsize(local_ckp)/1e6:.2f} MB)", flush=True)

print("\n=== Stopping Colab L4 Session to Save Compute Units ===", flush=True)
try:
    rt.stop(shutdown_kernel=True)
except Exception:
    pass
state.client.unassign(s.endpoint)
state.store.remove('stage1')
print("Session 'stage1' terminated cleanly! Active sessions: 0", flush=True)
sys.stdout.flush()
os._exit(0)
