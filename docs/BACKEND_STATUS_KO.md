# RWKV backend 상태와 주의점

## 과학적 main path

H3/H4는 transaction slots에서 gradient가 frozen RWKV native forward를 지나 encoder로 돌아와야 한다. 따라서 `inputs_embeds`와 recurrent cache를 동시에 지원하는 differentiable backend가 필요하다.

현재 baseline은 다음 두 경로를 제공한다.

1. `rwkv_pip`: 공식 PTH 계열 inference-only. H1/H2 text policy와 결과 교차 검증에 사용한다.
2. `hf_stateful`: FLA/HF-format RWKV 후보. H3/H4의 differentiable slot forward에 사용한다.

## 서버에서 반드시 확인할 gate

- `AutoModelForCausalLM` 로드 성공
- `past_key_values` 또는 recurrent cache 반환
- `inputs_embeds` + cache forward 성공
- slot embedding에서 encoder parameter까지 finite gradient
- token ID/embedding path candidate ranking parity
- chunked/full prefill parity
- clone된 cache가 원본을 mutate하지 않음

이 gate가 통과하기 전에는 H3/H4 코드가 존재한다는 사실을 실제 RWKV 학습이 동작한다고 표현하지 않는다.

## 모델 고정 원칙

메인 weight는 `fla-hub/rwkv7-2.9B-g1`으로 고정한다. 해당 FLA model card가 가리키는 원본은 `rwkv7-g1-2.9b-20250519-ctx4096.pth`다. `rwkv_pip` 교차 검증에서도 `CATENA_RWKV_PTH`가 이 파일을 가리키도록 한다.

Pilot에서 동작한 정확한 model repository revision과 FLA commit을 pin한다. 최신 PTH와 FLA converted model이 다른 weight라면 한 논문의 H1-H4 main result를 섞지 않는다. 가능하면 동일 source checkpoint를 변환한 backend를 사용하고, 불가능하면 backend/model 차이를 명시한다.

## 실패 시 경로

- FLA 2.9B만 실패: 0.4B debug로 adapter 수정 후 재시도
- custom kernel 실패: pure PyTorch/reference mode로 correctness gate, 이후 kernel 패치
- differentiable RWKV가 마감 전에 불안정: H1/H2 diagnostic + Transformer boundary를 4쪽 paper로 제한
