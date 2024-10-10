copy from https://github.com/rustai-solutions/candle_demo_openchat_35
openchat在线体验 https://openchat.team/zh

This demo uses the quantized version of LLM openchat: https://huggingface.co/TheBloke/openchat_3.5-GGUF by default.

```
pip install -U "huggingface_hub[cli]"

mkdir hf_hub
HF_HUB_ENABLE_HF_TRANSFER=1 HF_ENDPOINT=https://hf-mirror.com huggingface-cli download TheBloke/openchat_3.5-GGUF openchat_3.5.Q8_0.gguf  --local-dir hf_hub
HF_HUB_ENABLE_HF_TRANSFER=1 HF_ENDPOINT=https://hf-mirror.com huggingface-cli download openchat/openchat_3.5 tokenizer.json --local-dir hf_hub
```

下载模型后hf_hub目录包含如下文件
openchat_3.5.Q8_0.gguf 7.16GB
openchat_3.5_tokenizer.json 1.71MB