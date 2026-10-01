# Release provenance

## Source tree

The NInfer Windows V100 derivative starts from
[`geoffwatts/ninfer-v100`](https://github.com/geoffwatts/ninfer-v100) revision
[`b37d0dd3e1163b9d802d8bccfa89918bf68d793e`](https://github.com/geoffwatts/ninfer-v100/commit/b37d0dd3e1163b9d802d8bccfa89918bf68d793e).
The Windows/V100 port and browser UI changes are attributed to taotuotu, 2026.
The source tree keeps its root `LICENSE` and the per-component notices listed
in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

The Windows binary is built for Windows x64 and the Volta `sm_70` target. The
release bundle includes `cudart64_12.dll` from CUDA Toolkit 12.9. It does not
include the NVIDIA driver, MSVC runtime, CUDA Toolkit, or model weights. The
host must supply the driver and Visual C++ Runtime. The installed CUDA 12.9
EULA text shipped with the bundle is [`licenses/cuda-12.9-EULA.txt`](../licenses/cuda-12.9-EULA.txt).

## Qwen3.8-27B NVFP4 artifact

The Windows server uses the upstream NInfer artifact repository
[`neroued/Qwen3.8-27B-nvfp4-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer),
pinned to release revision
[`52907138a5d23a8f7f868ba7b773e721fd275405`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer/commit/52907138a5d23a8f7f868ba7b773e721fd275405).
Use the canonical file `qwen3_8_27b_nvfp4.ninfer` from that revision and verify:

| Field | Value |
|---|---|
| Container version | 2 |
| Size | 23,719,496,192 bytes |
| SHA-256 | `552c374c685dce302603b95fbe940fb04243c0cd44c083efc644ad3d980d462c` |
| NInfer model ID | `qwen3.8-27b` |
| NInfer weights ID | `nvfp4` |

The local server installation may name the verified file
`qwen3_8_27b_nvfp4_v2.ninfer`; the bytes must still match the hash above. The
model file is not included in the source archive or Windows binary package.
This project points to the fixed upstream artifact and does not create or
publish new model weights.

The v2 artifact manifest records these source identities:

| Content | Repository and revision in the v2 manifest | Published license metadata |
|---|---|---|
| Base model, tokenizer and frontend resources | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0), revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | Apache-2.0 |
| Quantized text source | [`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4), recorded revision `60e813d4dbbdc5d64cf3f5a8caf2897bedf03679` | Apache-2.0 |
| DFlash2 companion weights | [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4), revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` | Apache-2.0 |

The pinned Qwen and DFlash2 pages expose these revisions and their Apache-2.0
metadata. The recorded Unsloth revision did not resolve from the current
Hugging Face endpoint during this provenance check. Its source revision and
file availability remain to be confirmed before claiming that the conversion
inputs are fully reproducible.

The v2 manifest records optimized proposal-head input
`tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64` with SHA-256
`c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5`. The
associated manifest names `tools/freq_corpus/fixtures/corpus/train.jsonl`,
which is not present in this tree, and does not identify its source license.
The corpus lineage therefore remains unclosed; this release does not regenerate
or publish weights from that input.

The upstream artifact repository's current `main` has advanced to container
version 3 and a different SHA-256. Download instructions for this Windows
release must keep the v2 revision and hash above rather than following `main`.
