# 🪵 Architectural Decision Log (ADR)

| ID | Decision | Context | Outcome |
|----|----------|---------|---------|
| **ADR-001** | **Abandon 4-bit Quantization** | MI300X has 192GB VRAM. 4-bit `bitsandbytes` kernels crashed on ROCm 6.4 due to blocksize assertions. | **Switched to Native Bfloat16.** Gained speed, stability, and utilized available VRAM. |
| **ADR-002** | **Map-Style vs. Iterable** | `IterableDataset` caused `RuntimeError: shape [0,4,-1]` because `trl` trainer incorrectly sliced 1D visual tokens. | **Switched to Map-Style (`load_from_disk`).** Solved tensor shaping issues; enabled efficient disk-offloading. |
| **ADR-003** | **Batch Size = 1** | Qwen2.5-VL handles variable image sizes. Batching >1 caused broadcasting errors (`tensor a(2) != b(1600)`). | **Enforced BS=1, GradAccum=8.** Maintains effective batch size of 8 while ensuring tensor stability. |
| **ADR-004** | **Resolution Clamp (504px)** | Removing resolution limits caused the processor to collapse videos to 1x1 pixels (empty tensor crash). | **Clamped to 504x504.** 504 is a perfect multiple of the 14px patch size ($14 \times 36$), ensuring valid grid formation. |
