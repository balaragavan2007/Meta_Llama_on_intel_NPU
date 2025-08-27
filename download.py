from huggingface_hub import snapshot_download

model_id = "llmware/llama-3.2-3b-instruct-npu-ov"
local_dir = "llama32_3b_npu"

snapshot_download(repo_id=model_id, local_dir=local_dir)
