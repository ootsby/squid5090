# Setting up this fork on an RTX 5070 Ti (16 GB)

This sets up the engine, the 3-bit expert files the mixes need, and the model router, on Windows 11 with WSL containers (`wslc`).
Measurements and the reasons for each choice are in [`5070TI.md`](5070TI.md).
Paths below use `F:\squid\model` for the model folder (`MODEL_DIR`) and `F:\squid-quant` as a work folder; any folders will do.

**Needs:**
- An NVIDIA driver with CUDA 13 support.
- WSL with `wslc`.
- Python 3.10+ with `numpy` and `gguf` (tested with numpy 2.2.6 and gguf 0.19.0).
- The `hf` CLI (`pip install huggingface-hub`).
- About 75 GB of disk while building: the model 21 GB, the Unsloth inputs 23 GB, one temporary GGUF at a time, and the 3-bit files 7.3 GB.

## 1. Model weights

```bash
hf download thinkingmachines/Inkling-2-Mini --local-dir F:/squid/model
```

Optional, for the `abl` presets (runtime ablation, 1.8 MB):

```bash
hf download satellitedown/Inkling-2-Mini-abliterated-ingine inkling-2-mini-abliterated.safetensors \
  --revision cea73d67cd0e424c14c8b5cd276a9a2c2a5d7cbd --local-dir F:/squid/model
# SHA-256 45686f11454c98c9e393aac4129695d0c2b86e90327e7f61421eff470bc0dcb9
```

## 2. The 3-bit expert files

Mix D reads `iq3.safetensors`: the gate/up experts of layers 0-4, 12 and 18-23 as IQ3_XXS.
Mix F (the `f8` and `f18` router modes) reads `iq3f.safetensors`: layers 0-17.
They come from Unsloth's GGUF and importance matrix, quantised with llama.cpp and then converted by `tools/mkiq3.py`.
Without them, `--mode mix` falls back to mix B, which only fits at short contexts.

Inputs (revision `8078d149286eec7193795413b9e0643da6d11fa1` of `unsloth/Inkling-2-Mini-GGUF`):

```bash
cd F:/squid-quant
hf download unsloth/Inkling-2-Mini-GGUF --revision 8078d149286eec7193795413b9e0643da6d11fa1 \
  --include "Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file" --include "imatrix_unsloth.gguf_file" --local-dir models
mv models/Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file models/Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf
# SHA-256: Q8_K_XL 2da51788d43a17e1f033bec420fa83ba9cb87ac1f0130dc9135a4d2ed2678b68
#          imatrix 6f5364dd452a2c0bf4ee17f039ae3c06667c7eb34679053347e1449d67b8c44b
```

llama.cpp release **b12676** (`llama-b12676-bin-win-cuda-*-x64.zip` from github.com/ggml-org/llama.cpp/releases), unpacked to `F:\squid-quant\llama`.
Only `llama-quantize` is used, and only on the CPU.

Then, from `F:\squid-quant` in Git Bash, with `REPO` set to this repo's folder. On a Ryzen 7 5800X the two runs took 6 and 9 minutes of quantising, plus about a minute each for `mkiq3.py`.

```bash
q() {  # $1: layers to make IQ3_XXS, $2: output name
  ./llama/llama-quantize.exe --allow-requantize --imatrix models/imatrix_unsloth.gguf_file \
    --tensor-type "blk\.($1)\.ffn_gate_up_exps=iq3_xxs" --tensor-type 'blk\.([0-9]|1[0-9])\.ffn_(gate_up|down)_exps=nvfp4' \
    --tensor-type 'weight=bf16' --output-tensor-type bf16 --token-embedding-type bf16 \
    models/Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf $2.gguf Q8_0 12
}
q '0|1|2|3|4|12|18|19|20|21|22|23' D && python $REPO/tools/mkiq3.py D.gguf F:/squid/model iq3.safetensors && rm D.gguf
q '[0-9]|1[0-7]' F && python $REPO/tools/mkiq3.py F.gguf F:/squid/model iq3f.safetensors && rm F.gguf
```

`mkiq3.py` takes only the IQ3_XXS gate/up tensors from the GGUF; the other flags only affect tensors it ignores.
It checks two experts per layer against the official weights: relative errors of about 0.18-0.19 are normal for 3 bits, and it stops above 0.5.
Expected results (rebuilt and compared byte for byte on 2026-11-26):

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `iq3.safetensors` | 3,121,359,064 | d9ea44d94aa35d82f126d4bd346d67a7822c5479583b498084e80e69b287f0f5 |
| `iq3f.safetensors` | 4,682,038,600 | 22413035f0500bd1451a4bc6a722b35b48a0407c8cf6b0eaa6c9ff46b0f8c831 |

The work folder can be deleted afterwards.

## 3. Engine images

From the repo root. The prefill batch is fixed at build time: 512 for most presets, 1024 for the 128K preset.

```bash
wslc build -f deploy/Dockerfile --build-arg PREFILL_BATCH=512 -t squid5090:f512 .
wslc build -f deploy/Dockerfile --build-arg PREFILL_BATCH=1024 -t squid5090:f1024 .
```

## 4. Faster loading (optional)

The engine loads its weights at about 0.4 GB/s from a Windows folder and about 4 GB/s from a `wslc` volume.
A volume is stored inside WSL's disk image on C:, so it needs about 28 GB free there.

```bash
wslc volume create squid-model
wslc run --rm -v 'F:\squid\model:/src:ro' -v squid-model:/dst --entrypoint sh squid5090:f512 -c 'cp -r /src/. /dst/ && sync'
```

Re-run the copy after changing anything in the model folder.

## 5. Router

Edit `router.json`:
- **`model_dir`:** your model folder.
- **`volume`:** the volume name, or remove the line to load from `model_dir`.

Then run `router.bat`, or `python router.py`. It listens on `http://127.0.0.1:8000/v1`, loads a preset on its first request, and unloads it after `idle_unload_seconds`.

| Preset | What it is |
|---|---|
| `inkling-2-mini` (default) | Mix D, 32K context |
| `inkling-2-mini-64k` | Mix D, 64K |
| `inkling-2-mini-quality` | Mix F8: closer to the reference, 32K |
| `inkling-2-mini-128k` | Mix F18, 128K, prefill 1024 |
| `inkling-2-mini-vision` | Mix D with image input (+0.6 GiB) |
| `inkling-2-mini-abliterated`, `-64k`, `-vision` | The same with runtime ablation (needs the file from step 1) |

Options can be added to any preset name, for example `inkling-2-mini:ctx=49152,mtp=0` (see the top of `router.py`).
The VRAM limit (`max_arena_gib`, 14.5) suits a 5070 Ti with an idle desktop. Lower it if other programs use the GPU.

Check that it works:

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"Say hello."}]}'
```

The first request takes about a minute while the model loads.

`squid.bat [port] [mix or nvfp4] [abl]` runs one engine directly, without the router, with mix D at 32K by default.
