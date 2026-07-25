---
name: Benchmark report
about: Ran rapidAI on your own Apple Silicon Mac? Share the numbers — I want to map where the wall sits across real hardware.
title: "[benchmark] <chip> <RAM> — <model>"
labels: benchmark
assignees: ''
---

Thanks for running it! The whole thesis is about the relationship between RAM, model size, and speed, and I've got exactly one hardware data point. Yours helps.

## Machine

- **Chip:** (e.g. M3 Pro, M1 Max, M4)
- **RAM:** (e.g. 18 GB)
- **macOS version:** (e.g. 15.5 / Darwin 25.5)
- **Disk:** (internal NVMe / external SSD — and roughly how fast, if you know)

## Run

- **Model:** (e.g. Qwen3-30B-A3B 3-bit, GPT-OSS-120B-MXFP4-Q4)
- **Expert-cache size:** (e.g. 6 GB)
- **tok/s:**
- **Cache hit rate:**
- **bytes/token:** (if reported)
- **Decode tokens measured:** (e.g. 256)

## Notes

Anything odd? Thermal throttling, page-cache effects, OOM, coherent-vs-garbage output, prompt used, whatever stood out. Raw JSON from `docs/experiments/data/` is welcome as an attachment.
</content>
