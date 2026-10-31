# ==============================================================================
# ORIGINAL EXECUTED SCRIPT EXTRACTED VERBATIM FROM TRANSCRIPT
# Transcript Step Index : 263
# Timestamp             : 2026-10-30T02:40:08Z
# ==============================================================================

import os, sys, uuid
from colab_cli.common import state
from colab_cli.client import Accelerator, Variant, resolve_assign_shape
from colab_cli.state import SessionState
from colab_cli.runtime import ColabRuntime
from colab_cli.contents import ContentsClient

name = "stage1_lora_dpo"
print(f"[colab] Allocating L4 GPU session '{name}'...", flush=True)
shape = resolve_assign_shape(Accelerator.L4, high_mem=False)
res = state.client.assign(uuid.uuid4(), variant=Variant.GPU, accelerator=Accelerator.L4, shape=shape)
token = res.runtime_proxy_info.token
url = res.runtime_proxy_info.url
endpoint = res.endpoint
s = SessionState(
    name=name,
    token=token,
    url=url,
    endpoint=endpoint,
    token_expires_at=res.runtime_proxy_info.expires_at(),
    variant=Variant.GPU.value,
    accelerator=Accelerator.L4.value,
    machine_shape="STANDARD"
)
state.store.add(s)
print(f"[colab] Session '{name}' READY on {endpoint}", flush=True)

rt = ColabRuntime(s.url, s.token, kernel_id=None, session_id=None)
contents = ContentsClient(s)

def on_out(o):
    if o.get("output_type") == "stream":
        print(o.get("text", ""), end="", flush=True)
    elif o.get("output_type") == "error":
        print("ERROR:", o.get("ename"), o.get("evalue"), flush=True)

try:
    rt.execute_code("""
import os, subprocess
if not os.path.exists('/content/minimind'):
    # [audit #17] pinned: 233d00f (2026-10-27) is the upstream HEAD that predates every original run
    subprocess.run(['git', 'clone', '--filter=blob:none', 'https://github.com/jingyaogong/minimind.git', '/content/minimind'], check=True)
    subprocess.run(['git', '-C', '/content/minimind', 'checkout', '-q', '233d00fa068ec2b8c0d4f07b703934d6391a5a73'], check=True)
os.makedirs('/content/minimind/out', exist_ok=True)
os.makedirs('/content/minimind/dataset', exist_ok=True)
print('MiniMind repo ready!', flush=True)
""", output_hook=on_out, timeout=120.0)

    local_sft = r"D:\Code\LLM-learning\checkpoints\stage1_minimind\full_sft_512.pth"
    print(f"[colab] Uploading {local_sft} to VM...", flush=True)
    contents.upload(local_sft, "/content/minimind/out/full_sft_512.pth")
    print("[colab] Upload complete!", flush=True)

    lora_dpo_code = """
import os, sys, json, requests, subprocess, torch
sys.path.insert(0, '/content/minimind')
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora

tokenizer = AutoTokenizer.from_pretrained('/content/minimind/model')

print("=== [Part A.1] Downloading lora_identity.jsonl (23 KB) ===", flush=True)
lora_data = '/content/minimind/dataset/lora_identity.jsonl'
r = requests.get('https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/lora_identity.jsonl')
with open(lora_data, 'wb') as f:
    f.write(r.content)
with open(lora_data, 'r', encoding='utf-8') as f:
    print("Sample LoRA Identity data:", f.readline().strip(), flush=True)

print("\\n=== [Part A.2] Running LoRA Fine-tuning (train_lora.py) ===", flush=True)
cmd_lora = [
    sys.executable, "-u", "train_lora.py",
    "--data_path", "../dataset/lora_identity.jsonl",
    "--from_weight", "full_sft",
    "--lora_name", "lora_identity",
    "--lora_rank", "8",
    "--hidden_size", "512",
    "--num_hidden_layers", "8",
    "--epochs", "20",
    "--batch_size", "16",
    "--learning_rate", "5e-4",
    "--log_interval", "20",
    "--save_interval", "100"
]
proc = subprocess.Popen(cmd_lora, cwd="/content/minimind/trainer", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()

lora_file = '/content/minimind/out/lora_identity_512.pth'
sft_file = '/content/minimind/out/full_sft_512.pth'
print(f"\\n[File Size Comparison] Base SFT weight: {os.path.getsize(sft_file)/1e6:.2f} MB vs LoRA Adapter weight: {os.path.getsize(lora_file)/1e6:.2f} MB!", flush=True)

print("\\n=== [Part A.3] Testing Model WITHOUT vs WITH LoRA Adapter Plugged In ===", flush=True)
base_model = MiniMindForCausalLM(MiniMindConfig(hidden_size=512, num_hidden_layers=8, use_moe=False))
base_model.load_state_dict(torch.load(sft_file, map_location='cuda'), strict=True)
base_model = base_model.half().eval().cuda()

prompt_q = "你好，你是谁？是谁开发的你？"
msgs = [{"role": "user", "content": prompt_q}]
prompt_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inp = tokenizer(prompt_str, return_tensors="pt").to("cuda")

torch.manual_seed(42)
out_no_lora = base_model.generate(inp["input_ids"], max_new_tokens=50, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
print(f"[Without LoRA]: {tokenizer.decode(out_no_lora[0][len(inp['input_ids'][0]):], skip_special_tokens=True)}", flush=True)

apply_lora(base_model, rank=8)
load_lora(base_model, lora_file)
base_model = base_model.half().eval().cuda()
torch.manual_seed(42)
out_with_lora = base_model.generate(inp["input_ids"], max_new_tokens=50, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
print(f"[With LoRA Plugged In]: {tokenizer.decode(out_with_lora[0][len(inp['input_ids'][0]):], skip_special_tokens=True)}", flush=True)
del base_model
torch.cuda.empty_cache()

print("\\n=== [Part B.1] Streaming 500 DPO Preference Pairs (chosen vs rejected) ===", flush=True)
dpo_fast = '/content/minimind/dataset/dpo_fast_500.jsonl'
url_dpo = 'https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/dpo.jsonl'
with requests.get(url_dpo, stream=True) as r, open(dpo_fast, 'wb') as f:
    for i, line in enumerate(r.iter_lines()):
        if i >= 500:
            break
        if line:
            f.write(line + b'\\n')

with open(dpo_fast, 'r', encoding='utf-8') as f:
    sample_dpo = json.loads(f.readline())
print(f"Sample DPO Question: {sample_dpo['chosen'][0]['content'][:80]}...", flush=True)
print(f"  👍 Chosen Answer:   {sample_dpo['chosen'][1]['content'][:80]}...", flush=True)
print(f"  👎 Rejected Answer: {sample_dpo['rejected'][1]['content'][:80]}...", flush=True)

print("\\n=== [Part B.2] Running DPO Training (train_dpo.py) ===", flush=True)
cmd_dpo = [
    sys.executable, "-u", "train_dpo.py",
    "--data_path", "../dataset/dpo_fast_500.jsonl",
    "--from_weight", "full_sft",
    "--hidden_size", "512",
    "--num_hidden_layers", "8",
    "--epochs", "1",
    "--batch_size", "8",
    "--learning_rate", "1e-6",
    "--beta", "0.15",
    "--log_interval", "15",
    "--save_interval", "50"
]
proc_dpo = subprocess.Popen(cmd_dpo, cwd="/content/minimind/trainer", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc_dpo.stdout:
    print(line, end="", flush=True)
proc_dpo.wait()
"""
    rt.execute_code(lora_dpo_code, output_hook=on_out, timeout=600.0)

    local_lora = r"D:\Code\LLM-learning\checkpoints\stage1_minimind\lora_identity_512.pth"
    local_dpo = r"D:\Code\LLM-learning\checkpoints\stage1_minimind\dpo_512.pth"
    contents.download("/content/minimind/out/lora_identity_512.pth", local_lora)
    contents.download("/content/minimind/out/dpo_512.pth", local_dpo)
    print(f"[colab] Downloaded lora_identity_512.pth ({os.path.getsize(local_lora)/1e6:.2f} MB) and dpo_512.pth ({os.path.getsize(local_dpo)/1e6:.2f} MB)!", flush=True)
finally:
    print("[colab] Stopping session 'stage1_lora_dpo'...", flush=True)
    try:
        rt.stop(shutdown_kernel=True)
    except Exception:
        pass
    state.client.unassign(s.endpoint)
    state.store.remove(name)
    print("[colab] Session 'stage1_lora_dpo' terminated cleanly! Active sessions: 0", flush=True)
    sys.stdout.flush()
    os._exit(0)
