# Baseline status v0.6.0

## CPU에서 검증된 부분

- Episode/chain data generation and validation
- Transaction rendering and policy logic
- Coherence metrics and shard merge
- Mock model end-to-end smoke
- Transformer suffix-reprefill cache-crop unit test
- Transaction slot encoder and loss unit tests

## 대상 서버에서 확인해야 하는 부분

- `fla-hub/rwkv7-2.9B-g1`과 동일 source PTH(`20250519`)의 logits/ranking 교차 검증

- FLA/HF RWKV-7 model loading on CUDA 13/Blackwell
- Recurrent cache clone/crop semantics for the pinned model revision
- Differentiable `inputs_embeds` forward through the RWKV recurrent cache
- Custom CUDA extension compilation for the actual compute capability
- 2.9B runtime throughput and memory

## 워크샵 이후 확장 범위

- Learned transaction writer
- Graph database or retrieval training
- Full backbone fine-tuning
- RL, branch/merge, online deployment
