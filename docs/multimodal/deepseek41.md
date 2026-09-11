# DeepSeek-V4.1-Flash vision

Convert the vision encoder and projector from the original checkpoint:

```sh
python3 convert_hf_to_gguf.py ds41 --mmproj --outtype bf16 --outfile mmproj-ds41-bf16.gguf
```

The converter writes the `deepseek41v` projector type. V4.1 uses a row-major image grid with a newline after each row and learned start/end embeddings. It does not use V4's interleaved rows or image padding. Use the V4.1 projector with the V4.1 language model.

Existing V4.1 target GGUFs produced by this branch already contain the image expert-routing biases. They do not need conversion or quantization again. Rebuild the server and RPC workers from the updated source; only the separate mmproj needs conversion.

For a target split across the local device and an RPC worker:

```sh
llama-server -m model-quant.gguf --mmproj mmproj-ds41-bf16.gguf \
    --rpc 192.168.100.11:50052 --lazy-mode on -c 8192 \
    --host 0.0.0.0 --port 8000
```

Send images through the server's usual image input or `image_url` chat content. To try a local image with the CLI:

```sh
llama-mtmd-cli -m model-quant.gguf --mmproj mmproj-ds41-bf16.gguf \
    --rpc 192.168.100.11:50052 --lazy-mode on -c 8192 \
    --image image.png -p "Describe this image."
```

The default image budget is 1024 tokens, including newline and start/end embeddings. Images use the vision routing bias and skip Engram lookups; subsequent text resumes Engram with its history cut at the image boundary. Language-model attention stays causal for image embeddings.

Implementation references: the official [vision encoder](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/vision.py), [image processor](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/image_processor.py), and [language model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py).
