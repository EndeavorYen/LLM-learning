# ==============================================================================
# ORIGINAL EXECUTED SCRIPT EXTRACTED VERBATIM FROM TRANSCRIPT
# Transcript Step Index : 243
# Timestamp             : 2026-10-30T02:13:58Z
# ==============================================================================

import os, sys, uuid
from colab_cli.common import state
from colab_cli.client import Accelerator, Variant, resolve_assign_shape
from colab_cli.state import SessionState
from colab_cli.runtime import ColabRuntime
from colab_cli.contents import ContentsClient

name = "stage1_sft"
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
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/jingyaogong/minimind.git', '/content/minimind'], check=True)
os.makedirs('/content/minimind/out', exist_ok=True)
os.makedirs('/content/minimind/dataset', exist_ok=True)
print('MiniMind repo ready!', flush=True)
""", output_hook=on_out, timeout=120.0)

    local_pretrain = r"D:\Code\LLM-learning\checkpoints\stage1_minimind\pretrain_512.pth"
    print(f"[colab] Uploading local {local_pretrain} to VM...", flush=True)
    contents.upload(local_pretrain, "/content/minimind/out/pretrain_512.pth")
    print("[colab] Upload complete!", flush=True)

    sft_code = """
import os, sys, requests, subprocess, torch
sys.path.insert(0, '/content/minimind')

sft_fast_path = '/content/minimind/dataset/sft_fast_5k.jsonl'
if not os.path.exists(sft_fast_path):
    print("=== [1] Streaming 5,000 SFT samples from Hugging Face ===", flush=True)
    url = 'https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/sft_t2t_mini.jsonl'
    with requests.get(url, stream=True) as r, open(sft_fast_path, 'wb') as f:
        r.raise_for_status()
        for i, line in enumerate(r.iter_lines()):
            if i >= 5000:
                break
            if line:
                f.write(line + b'\\n')
print("SFT 5k dataset ready!", flush=True)

print("\\n=== [2] Inspecting SFTDataset Token & -100 Masking ===", flush=True)
from transformers import AutoTokenizer
from dataset.lm_dataset import SFTDataset
tokenizer = AutoTokenizer.from_pretrained('/content/minimind/model')
ds = SFTDataset(sft_fast_path, tokenizer, max_length=128)
input_ids, labels = ds[0]
for idx in range(32):
    tok_id = input_ids[idx].item()
    lbl_id = labels[idx].item()
    tok_str = repr(tokenizer.decode([tok_id]))
    status = "IGNORE (-100)" if lbl_id == -100 else f"SCORE (id={lbl_id})"
    print(f"Pos {idx:2d}: Token={tok_str:14s} -> Label: {status}", flush=True)

print("\\n=== [3] Running Full SFT (train_full_sft.py) ===", flush=True)
cmd = [
    sys.executable, "-u", "train_full_sft.py",
    "--data_path", "../dataset/sft_fast_5k.jsonl",
    "--from_weight", "pretrain",
    "--hidden_size", "512",
    "--num_hidden_layers", "8",
    "--epochs", "1",
    "--batch_size", "32",
    "--accumulation_steps", "2",
    "--learning_rate", "5e-4",
    "--log_interval", "25",
    "--save_interval", "100"
]
proc = subprocess.Popen(cmd, cwd="/content/minimind/trainer", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()

print("\\n=== [4] Evaluating SFT Model (Chat Template Mode) ===", flush=True)
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
sft_model = MiniMindForCausalLM(MiniMindConfig(hidden_size=512, num_hidden_layers=8, use_moe=False))
sft_model.load_state_dict(torch.load('/content/minimind/out/full_sft_512.pth', map_location='cuda'), strict=True)
sft_model = sft_model.half().eval().cuda()
torch.manual_seed(42)
for p in ["你好，请介绍一下你自己", "为什么天空是蓝色的？", "解释什么是机器学习"]:
    msgs = [{"role": "user", "content": p}]
    prompt_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(prompt_str, return_tensors="pt").to("cuda")
    out = sft_model.generate(
        inp["input_ids"],
        max_new_tokens=60,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    ans = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True)
    print(f"User: {p}\\nAssistant: {ans}\\n", flush=True)
"""
    rt.execute_code(sft_code, output_hook=on_out, timeout=600.0)

    local_sft = r"D:\Code\LLM-learning\checkpoints\stage1_minimind\full_sft_512.pth"
    print(f"[colab] Downloading full_sft_512.pth to {local_sft}...", flush=True)
    contents.download("/content/minimind/out/full_sft_512.pth", local_sft)
    print(f"[colab] Downloaded full_sft_512.pth ({os.path.getsize(local_sft)/1e6:.2f} MB)!", flush=True)
finally:
    print("[colab] Stopping session 'stage1_sft'...", flush=True)
    try:
        rt.stop(shutdown_kernel=True)
    except Exception:
        pass
    state.client.unassign(s.endpoint)
    state.store.remove(name)
    print("[colab] Session 'stage1_sft' terminated cleanly! Active sessions: 0", flush=True)
    sys.stdout.flush()
    os._exit(0)
