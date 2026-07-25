# Knowledge Insulation 구현 계획

> 대상: `openpi` 저장소 (`/home/mibe/git/openpi`)
> 기준 논문: _Knowledge Insulating Vision-Language-Action Models_ (PI, 2025)
> 참고 논문: _π0.5: a Vision-Language-Action Model with Open-World Generalization_ (PI, 2025)
> 분석 문서: `docs/ki_analysis.md`

---

## 개념 정리

### KI + Subtask는 상호 보완적

KI(Knowledge Insulation)와 π0.5의 Subtask(HL) 학습은 동일한 수식 구조 `L = L_CE(text) + α·L_FM(action)` 를 공유한다. 차이는 **예측 대상 텍스트의 종류**일 뿐이며, 두 기능은 아키텍처 충돌 없이 공존한다:

```
                    VLM Backbone (2B) — NTP(Next-Token Prediction)로 학습
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   Subtask/Caption   FAST Action Token   Action Expert (300M)
   CE Loss (π0.5)    CE Loss (KI)        Flow Matching Loss
         │               │               │
         └───────────────┘          Stop Gradient
              │                    (backbone 보호)
              ▼
         Backbone 의미 지식 강화
```

**주의**: 이 코드베이스의 Attention은 모든 expert의 Q/K/V를 단일 텐서로 concat하여 fused softmax를 수행한다 (`gemma.py:196`). "Stop Gradient는 `.detach()` 한 줄"이라는 설명은 expert가 backbone의 feature 텐서를 별도로 소비하는 아키텍처에만 해당하며, 이 코드베이스에서는 **Attention 모듈 내부를 약 50라인 수준으로 수정**해야 한다 (Phase 1.1 참조).

### Dual Loss + Stop Gradient 관계

- **Dual Loss는 필수** — CE(Backbone) + FM(Action Expert) 중 하나라도 제거 시 성능 저하
- **Stop Gradient 단독 사용은 오히려 성능 저하** — backbone이 로봇 신호를 전혀 받지 못해 표현 학습이 불가능 (논문 "frozen backbone → 0%" 결과 참조)
- 따라서 `stop_gradient_actions=True`는 반드시 `joint_fast_training=True` 또는 `cotrain_subtask_data=True`와 함께 사용
- VLM Co-training은 Backbone이 NTP 모드에 머물기 때문에 자연스럽게 가능

---

## Phase 0: 사전 작업

### 0.1 Feature Flag 선언

- [ ] `src/openpi/models/pi0_config.py`에 신규 flag 4개 추가

```python
# 신규 필드 (기본값 False — 기존 코드 완전 호환)
stop_gradient_actions: bool = False   # Phase 1
joint_fast_training: bool = False     # Phase 2A
cotrain_subtask_data: bool = False    # Phase 2B
cotrain_vlm_data: bool = False        # Phase 3
```

### 0.2 Guard 및 토큰 버짓

- [ ] `Pi0Config.__post_init__`에 validation 추가:
  - `stop_gradient_actions=True` + `joint_fast_training=False` + `cotrain_subtask_data=False` → 오류 ("stop-gradient requires at least one CE source")
  - `joint_fast_training=True` 또는 `cotrain_subtask_data=True`일 때 `max_token_len` 최소값 검증 (FAST 토큰 + subtask 토큰 합산 → 250~400 필요)
- [ ] `max_token_len` 기본값 재계산:
  - Prompt + State: ~80 tokens
  - FAST actions: ~50-100 tokens
  - Subtask text: ~30-50 tokens
  - 총합: 약 200-300. 기존 `pi05`의 200은 부족할 수 있음. Flag 활성화 시 자동 조정 또는 경고

### 0.3 `get_freeze_filter` 회귀 방지

- [ ] Action-expert params가 `.*llm.*_1.*` 패턴으로 식별됨 (`pi0_config.py:93`). Stop-gradient 구현이 expert 인덱싱이나 naming을 변경하지 않음을 확인하는 테스트 추가

### 0.4 `ModelTransformFactory` dispatch

- [ ] `config.py:114-160`의 `ModelTransformFactory`는 `model_type`(`PI0`/`PI05`/`PI0_FAST`) 기반으로 transform을 분기. 새 flag는 `model_type`을 변경하지 않으므로:
  - `joint_fast_training=True` → `TokenizeJointInputs`가 적용되도록 dispatch 로직 수정
  - `cotrain_subtask_data=True` → HL/WD 데이터 transform이 적용되도록 수정

**검증**: 기존 config(`pi05=True`, `pi05=False`)가 그대로 로드되고 동작하는지 확인.

---

## Phase 1: Stop Gradient (`stop_gradient_actions`)

> **목표**: Action Expert의 그래디언트가 VLM Backbone으로 역전파되지 않도록 차단
> **핵심**: 논문 식 (5)-(6). 이 코드베이스는 fused-QKV attention 구조이므로 단순 `.detach()` 이상의 수정이 필요.

### 1.1 설계 결정: Two-Pass vs Split-Einsum

Attention (`gemma.py:196`)은 모든 expert의 Q/K/V를 단일 텐서로 concat → fused softmax → 단일 einsum. Stop-gradient를 위해 두 가지 설계 선택지:

| 방식 | 구현 | 메모리 | FLOPs |
|---|---|---|---|
| **Two-Pass** | Backbone K/V를 `sg()`한 복사본으로 attention 한 번 더 수행, E_a/E_b 분리 계산 | K/V 복사본 1개 추가 | ~1.3x |
| **Split-Einsum** | fused softmax 유지하되, `sg(V_b)`와 `V_b`로 value einsum 분리 | 거의 무변화 | ~1.0x |

- [ ] JAX: **Split-Einsum** 권장 (value projection만 `sg` + split, softmax는 공유). `stop_gradient_actions=False`일 때 기존 fused path 유지.
- [ ] PyTorch (`gemma_pytorch.py:compute_layer_complete`): 동일 접근. `V_b.detach()`로 value split.

### 1.2 Flag 전달 방식 (remat/scan 안전성)

- [ ] `stop_gradient_actions`는 `Block`/`Attention`의 **생성자 인자(module attribute)** 로 전달. `__call__`의 positional argument로 추가하면 `nn.remat(static_argnums=(5,))`와 `nn.scan(in_axes=...)`가 깨짐 (`gemma.py:345-372`).
- [ ] `pi0.py`: `PaliGemma.llm()`의 `Module` 생성자에 `stop_gradient_actions` 전달. call-signature 변경 없음.

```
# Block 생성 시 (gemma.py:Block.__init__)
def __init__(self, configs, dropout=0.0, dropout_bdims=(), stop_gradient_actions=False):
    self.stop_gradient_actions = stop_gradient_actions
    ...

# Attention.__call__ 내부
if self.stop_gradient_actions:
    V_b_detached = jax.lax.stop_gradient(V_b)  # for E_a computation
```

### 1.3 JAX: `gemma.py:Attention.__call__()`

- [ ] Backbone의 Value (`V_b`)를 Action Expert의 output 계산(`E_a`)에 사용할 때만 `jax.lax.stop_gradient()` 적용
- [ ] Backbone → Backbone attention (`E_b = P_bb @ V_b`)은 정상 gradient flow
- [ ] 기존 out-projection slice (`encoded[:, start:end]`)의 contiguous 가정 유지

### 1.4 PyTorch: `gemma_pytorch.py`

- [ ] `compute_layer_complete()`에서 동일한 로직을 `.detach()`로 구현
- [ ] forward에 `stop_gradient_actions: bool` 파라미터 추가

**변경 파일**:
- `src/openpi/models/gemma.py` — `Attention`, `Block`, `Module`
- `src/openpi/models/pi0.py` — flag를 `Module` 생성자로 전달
- `src/openpi/models_pytorch/gemma_pytorch.py`
- `src/openpi/models_pytorch/pi0_pytorch.py`

### 1.5 검증

- [ ] `stop_gradient_actions=True`일 때 Action Expert → Backbone 그래디언트가 0인지 단위 테스트
- [ ] 기존 `pi05=True`/`pi05=False` 성능 회귀 없음 확인
- [ ] **JAX↔PyTorch parity test**: 동일 config + 동일 입력에서 두 구현체의 loss/grad 차이가 machine epsilon 이내인지 확인. `gemma_pytorch.py`는 hand-rolled fused attention 루프를 사용하므로 drift 위험 높음.
- [ ] Inference 시 (`sample_actions`): `no_grad` context에서 stop-gradient가 no-op임을 확인 (구현 시 sampling path에 불필요한 `sg` 추가 방지)

---

## Phase 2.0: Gemma 텍스트 출력 헤드 확장 (Phase 2A/2B 사전 요건)

> **차단 이슈**: `gemma.py:Module.__call__`은 hidden states만 반환. CE loss를 위한 logits/prelogits 경로가 없음.
> `gemma_fast.py`에는 `return_prelogits`/`pre_logits`/`decode`가 구현되어 있으므로 이를 포팅.

- [ ] `gemma.py:Module.__call__`에 `return_prelogits: bool` 및 `pre_logits` 파라미터 추가 (`gemma_fast.py:303-341` 참조)
- [ ] `embedder.decode(x)` 호출로 logits 출력 경로 추가
- [ ] `pi0.py:PaliGemma.llm()` 호출 시 새 파라미터 전달 인터페이스 확장

**변경 파일**:
- `src/openpi/models/gemma.py` — `Module.__call__`
- `src/openpi/models/pi0.py` — 호출부 수정

---

## Phase 2A: Joint FAST + Flow Matching (`joint_fast_training`)

> **목표**: 하나의 모델에서 FAST 이산 토큰 CE loss + Action Expert Flow Matching loss 동시 학습
> **수식**: `L = L_CE(FAST tokens) + α·L_FM(continuous actions)`

### 2A.1 모델 아키텍처 확장

- [ ] `Pi0` 클래스에 FAST 토큰 처리 경로 추가
  - `embed_prefix()`: FAST 이산 액션 토큰 임베딩 추가
  - Attention Mask: FAST 토큰 ↔ Action Expert 연속 토큰 간 attention 차단
  - `compute_loss()`: CE loss + Flow Matching loss 동시 계산

**변경 파일**: `src/openpi/models/pi0.py`
**PyTorch 대응**: `src/openpi/models_pytorch/pi0_pytorch.py`

### 2A.2 Attention Mask 수정

- [ ] FAST 토큰: prefix에 포함, 자기회귀적 causal attention
- [ ] Action Expert 연속 토큰: suffix에 포함, prefix attend 가능
- [ ] FAST 토큰 ↔ Action Expert 토큰: **상호 attend 불가**

```
Token Attention 규칙:
  ┌─────────────────────┬──────────┬──────────────┬──────────────┬───────────────┐
  │ Token →             │ Image/   │ FAST Action  │ Subtask Text │ Action Expert │
  │                     │ Prompt   │              │              │               │
  ├─────────────────────┼──────────┼──────────────┼──────────────┼───────────────┤
  │ Image/Prompt/State  │ Bidir    │ X            │ Bidir        │ X             │
  │ FAST Action         │ ✓(causal)│ Causal       │ Causal       │ X             │
  │ Subtask Text        │ ✓(causal)│ Causal       │ Causal       │ X             │
  │ Action Expert       │ ✓        │ X            │ X            │ Bidir         │
  └─────────────────────┴──────────┴──────────────┴──────────────┴───────────────┘
```

### 2A.3 손실 함수 및 Reduction Contract

- [ ] `BaseModel.compute_loss`의 return type: `at.Float[at.Array, "*b ah"]` → CE loss는 `(b,)`, FM loss는 `(b, ah)`. **Shape 불일치 해결**:
  - **Option A (권장)**: FM loss를 `axis=-1`로 mean → `(b,)`. CE도 `(b,)`. `L_total = L_CE + α * L_FM` → `(b,)`.
  - **Option B**: CE loss를 `(b, 1)`로 unsqueeze → broadcast → `(b, ah)`. α scale이 `action_horizon`(50)으로 implicit scaling됨에 주의.
  - `train.py:151`의 `jnp.mean(chunked_loss)`를 고려하여 per-sample mean이 동등하게 기여하도록 설계.
- [ ] `α`는 configurable (기본값 1.0, stop gradient 시 자연스러운 균형)
- [ ] `joint_fast_training=False`일 때 기존 FM-only 경로 유지

### 2A.4 추론 경로

- [ ] Inference 시 FAST 토큰 경로는 사용하지 않고 Action Expert만으로 연속 액션 생성
- [ ] 기존 `sample_actions()` 로직 그대로 사용

### 2A.5 데이터 변환 및 Dispatch

- [ ] `transforms.py`에 `TokenizeJointInputs` transform 추가
  - FAST 토큰(훈련 시 사용) + 연속 액션 레이블 + 텍스트 입력을 하나의 Observation으로 변환
- [ ] `ModelTransformFactory` (`config.py`)에 `joint_fast_training=True` 분기 추가
  - `pi05=True, joint_fast_training=True` config가 `TokenizeJointInputs`를 받도록 보장

### 2A.6 검증

- [ ] `joint_fast_training=True` + `stop_gradient_actions=False` 동작 확인
- [ ] `joint_fast_training=True` + `stop_gradient_actions=True` 동작 확인
- [ ] `joint_fast_training=False`일 때 FAST 관련 연산이 0 cost인지 확인
- [ ] **JAX↔PyTorch parity test**: 동일 config/입력에서 dual-loss 값 일치 확인

---

## Phase 2B: Subtask / HL 텍스트 예측 (`cotrain_subtask_data`)

> **목표**: 로봇 액션 데이터 없이도 subtask 문자열을 텍스트 CE loss로 학습 (π0.5 호환)
> **참고**: π0.5 논문 Section IV-C (HL), 식 (1). KI와 충돌 없음.
> **Phase 3과의 경계**: WD(Web Data)는 Phase 3에서 처리. Phase 2B는 HL(High-Level subtask)만 담당.

### 2B.1 텍스트 출력 헤드 활성화

- [ ] Phase 2.0에서 추가된 `gemma.py:Module`의 logits 경로를 활용하여 CE loss 계산
- [ ] Observation에 `text_target_tokens`, `text_target_mask` 필드 추가
- [ ] `compute_loss()`에 텍스트 CE loss 경로 추가 (기존 FM loss와 병렬)

**Loss 구성**:
```
L_CE(text) = -Σ mask_j · log p(target_token_j | prefix)   → (b,)
L_FM(action) = mean(‖v_t - u_t‖², axis=-1)                 → (b, ah) → reduce to (b,)
L_total = L_CE(text) + α * L_FM(action)                    → (b,)
```

### 2B.2 Subtask 데이터 포맷

- [ ] π0.5 포맷: `"Task: clean the bedroom, State: ...; Subtask: pick up pillow|"`
- [ ] HL 데이터: observation + high-level prompt → subtask text target
- [ ] `M_act = 0` — Flow Matching loss 미적용
- [ ] `ModelTransformFactory`에 `cotrain_subtask_data=True` 분기 추가

### 2B.3 손실 마스킹 (`M_act`)

- [ ] 데이터 타입별 loss 적용 규칙:

| 데이터 타입 | Text CE | FAST CE | Flow Matching (`M_act`) |
|---|---|---|---|
| Robot Action Data | 1 | 1 (`joint_fast_training=True`일 때) | 1 |
| Subtask/HL Data | 1 | 0 | 0 |

### 2B.4 2단계 추론 (Inference)

- [ ] Step 1: Autoregressive decode → subtask 문자열 ℓ̂ 예측
- [ ] Step 2: ℓ̂를 prefix에 추가 → Action Expert로 low-level action 생성
- [ ] 고주파 low-level inference 사이에 저주파로(예: 5Hz) subtask 재예측

### 2B.5 검증

- [ ] 액션 없는 HL 데이터 배치에서 Flow Matching loss = 0 확인
- [ ] Subtask 예측 → 조건부 action 생성 파이프라인 end-to-end 동작 확인
- [ ] Attention mask가 Subtask ↔ Action Expert 간 차단하는지 확인

---

## Phase 3: VLM Data Co-training (`cotrain_vlm_data`)

> **목표**: 웹 데이터(이미지 캡셔닝, VQA, 객체 탐지)를 Backbone NTP 모드로 co-training
> **핵심**: Backbone이 NTP 모드에 있으므로 자연스럽게 VLM 데이터 혼합 가능
> **Phase 2B와의 경계**: Phase 2B = HL/subtask only, Phase 3 = WD (caption/VQA/bbox) only.

### 3.1 데이터셋 확장

- [ ] VLM 데이터셋 포맷 정의 (이미지 + 텍스트 프롬프트 + 텍스트 타겟, 액션 없음)
- [ ] `DataConfig`에 VLM 데이터 경로/소스 추가
- [ ] 데이터 로더에서 VLM 데이터 + 액션 데이터 혼합 샘플링
- [ ] WD 데이터: `M_act = 0`, Text CE만 계산 (Subtask CE와 동일한 손실 경로 사용)

**변경 파일**: `src/openpi/training/data_loader.py`, `src/openpi/training/config.py`

### 3.2 검증

- [ ] VLM 데이터만 있는 배치에서 Flow Matching loss가 0인지 확인
- [ ] 액션 데이터 + VLM 데이터 혼합 배치에서 손실 분리 확인

---

## Phase 4: 통합 및 검증

### 4.1 Full KI Config

- [ ] `training/config.py`에 `pi05_ki_*` config preset 추가

```python
TrainConfig(
    name="pi05_ki_aloha",
    model=pi0_config.Pi0Config(
        pi05=True,
        stop_gradient_actions=True,
        joint_fast_training=True,
        cotrain_subtask_data=True,
        cotrain_vlm_data=True,
    ),
    ...
)
```

### 4.2 End-to-end 테스트

- [ ] Full KI config로 training loop 시작/1스텝 실행 성공
- [ ] Full KI config로 `sample_actions()` 추론 성공
- [ ] 기존 π0, π0.5, π0-FAST config 회귀 없음

### 4.3 성능 벤치마크

- [ ] `joint_fast_training=True` + `stop_gradient_actions=True` vs `joint_fast_training=True` only: 학습 속도 비교 (동일 데이터 기준 상대적 개선 측정)
- [ ] `cotrain_subtask_data` on/off Language following rate 측정
- [ ] `cotrain_vlm_data` on/off OOD 일반화 성능 비교
- [ ] **참고**: KI 논문은 full recipe + 다중 embodiment 혼합 기준 ~7.5x faster convergence 보고. 단일 embodiment에서는 상대적 추세만 검증.

---

## Phase 5: 문서화 (사후)

- [ ] `docs/ki_usage.md` — KI flag 사용 가이드
- [ ] CHANGELOG 업데이트

---

## 변경 파일 총괄

| 파일 | Phase | 변경 유형 |
|---|---|---|
| `src/openpi/models/pi0_config.py` | 0 | Flag 4개 + `__post_init__` guard 추가 |
| `src/openpi/models/gemma.py` | 1, 2.0 | Stop Gradient (JAX) + logits/prelogits 경로 |
| `src/openpi/models/pi0.py` | 1, 2.0, 2A, 2B | Stop Gradient 전달, logits 호출, Joint Training, Subtask CE, 2단계 추론 |
| `src/openpi/models/model.py` | 2A, 2B | Observation 필드 확장, `compute_loss` shape contract |
| `src/openpi/models/tokenizer.py` | 2B | Subtask 텍스트 토크나이즈 지원 |
| `src/openpi/transforms.py` | 2A, 2B, 3 | Joint/Subtask/VLM transform |
| `src/openpi/training/config.py` | 0, 2A, 2B, 3, 4 | `ModelTransformFactory` dispatch + Config preset |
| `src/openpi/models_pytorch/gemma_pytorch.py` | 1 | Stop Gradient (PyTorch) |
| `src/openpi/models_pytorch/pi0_pytorch.py` | 1, 2A, 2B | Stop Gradient 전달, Joint Training, Subtask CE |
| `src/openpi/training/data_loader.py` | 3 | VLM + Subtask 데이터셋 지원 |

---

## 구현 원칙

1. **기존 코드 불변**: 모든 신규 기능은 flag가 `False`일 때 기존 코드와 동일한 제어 흐름을 따른다.
2. **JAX / PyTorch 동시 변경**: 두 구현체에 동일한 기능을 적용하고 parity test로 drift 방지.
3. **단위 테스트 우선**: 각 Phase 완료 후 해당 flag에 대한 단위 테스트를 작성한다.
4. **작은 PR**: 각 Phase를 독립적인 PR로 제출 가능하도록 설계한다.
5. **Stop Gradient + CE Source 필수**: `stop_gradient_actions=True`는 반드시 `joint_fast_training=True` 또는 `cotrain_subtask_data=True`와 페어링. 단독 사용은 backbone이 로봇 신호를 받지 못해 성능 collapse. Phase 0.2의 `__post_init__` guard로 강제.

## 주의사항

- `docs/plan.md`의 체크리스트 항목을 구현 완료 시 반드시 갱신할 것
- 각 Phase 시작 전 `docs/ki_analysis.md`의 해당 섹션을 재확인할 것
- PyTorch/JAX 간 구현 차이가 없도록 parity test로 확인할 것
- 모델 아키텍처 변경 시 `pi0_test.py` 및 `model_test.py` 테스트 갱신할 것
- Inference 시 FAST 토큰/Subtask 토큰은 Action Expert와 attend 불가 (Attention Mask 검증 필수)
- Inference `sample_actions()`는 `no_grad` context — stop-gradient 구현이 sampling path와 무관함을 확인

---

## Flag 조합 시나리오

| Config | Train Fast | Run Fast | Generalize | Subtask | 비고 |
|---|---|---|---|---|---|
| π0 (전부 False) | ❌ 느림 | ✅ | ❌ | ❌ | Baseline |
| + stop_gradient_actions only | ❌ collapse | ✅ | ❌ | ❌ | **금지** — CE source 없음 |
| + joint_fast_training | ✅ | ✅ | ✅ | ❌ | KI Core |
| + joint_fast_training + stop_gradient_actions | ✅✅ | ✅ | ✅✅ | ❌ | KI Full (no subtask) |
| + cotrain_subtask_data | ✅ | ✅ | ✅ | ✅ | π0.5 스타일 HL |
| **π0.5 + KI (Full)** | ✅✅ | ✅ | ✅✅ | ✅ | 모든 flag 활성화 |