# Third-party notices

This inventory covers third-party code and runtime components used by the
Windows V100 build. The full license texts copied from the checked-out sources
and installed CUDA 12.9 toolkit are under [`licenses/`](licenses/). The
project's own license remains [`LICENSE`](LICENSE).

| Component | Version / source | License text |
|---|---|---|
| llama.cpp Volta flash-attention and MMA fragments | Upstream commit `62bf73d25`; see [`third_party/llama_cpp_fattn/README.md`](third_party/llama_cpp_fattn/README.md) and `src/ops/common/volta_mma.cuh` | [MIT](licenses/llama.cpp-MIT.txt) |
| cpp-httplib | 0.54.1, vendored single header | [MIT](licenses/cpp-httplib-MIT.txt) |
| nlohmann/json | 3.12.0, vendored single header | [MIT](licenses/nlohmann-json-MIT.txt) |
| spdlog | 1.17.0, commit `79524ddd08a4ec981b7fea76afd08ee05f83755d` | [MIT](licenses/spdlog-MIT.txt) |
| fmt, bundled by spdlog | 12.1.0; vendored sources under `third_party/spdlog/include/spdlog/fmt/bundled/` | [MIT and embedded-code exception](licenses/fmt-MIT-exception.txt) |
| utf8proc | 2.11.3, vendored source; includes Unicode-derived data | [MIT and Unicode data terms](licenses/utf8proc-and-Unicode.txt) |
| NVIDIA CUTLASS | v4.4.2, source commit `da5e086dab31d63815acafdac9a9c5893b1c69e2`; C++ template headers used by the Volta kernels | [BSD-3-Clause](licenses/CUTLASS-BSD-3-Clause.txt) |
| NVIDIA CUDA Runtime | CUDA Toolkit 12.9 `cudart64_12.dll` for the Windows binary bundle | [Installed CUDA 12.9 EULA](licenses/cuda-12.9-EULA.txt) |

The Windows bundle includes the CUDA Runtime DLL only. It does not include the
NVIDIA display driver, the CUDA Toolkit, the Microsoft Visual C++ Runtime, or
model weights. The host provides the NVIDIA driver and Visual C++ Runtime.

The `eval/corpora/perplexity-1m/` inputs have separate source licenses and
attribution in [`eval/corpora/perplexity-1m/THIRD_PARTY_NOTICES.md`](eval/corpora/perplexity-1m/THIRD_PARTY_NOTICES.md).
