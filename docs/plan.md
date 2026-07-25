# Knowledge Insulation 구현 계획

> 대상: `openpi` 저장소 (`/home/mibe/git/openpi`)
> 기준 논문: _Knowledge Insulating Vision-Language-Action Models_ (PI, 2025)
> 참고 논문: _π0.5: a Vision-Language-Action Model with Open-World Generalization_ (PI, 2025)
> 분석 문서: `docs/ki_analysis.md`
> 외부 참고: https://www.engineeringmaxxing.com/veanors/papers/pi05-ki.html

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

**핵심 통찰** (from engineeringmaxxing):
- Stop Gradient는 PyTorch 기준 `.detach()` 한 줄로 구현 가능
- Dual Loss는 필수 — **둘 중 하나라도 제거하면 성능 저하**
- VLM Co-training은 Backbone이 NTP 모드에 머물기 때문에 자연스럽게 가능
- 결과: **Train Fast × Run Fast × Generalize Better** — 삼중 승리

---

## Phase 0: 사전 작업

### 0.1 Feature Flag 선언

- [ ] `src/openpi/models/pi0_config.py`에 신규 flag 추가

```python
# 신규 필드 (기본값 False — 기존 코드 완전 호환)
stop_gradient_actions: bool = False   # Phase 1
joint_fast_training: bool = False     # Phase 2A
cotrain_vlm_data: bool = False        # Phase 3
cotrain_subtask_data: bool = False    # Phase 2B
```

**검증**: 기존 config(`pi05=True`, `pi05=False`)가 그대로 로드되고 동작하는지 확인.

---

## Phase 1: Stop Gradient (`stop_gradient_actions`)

> **목표**: Action Expert의 그래디언트가 VLM Backbone으로 역전파되지 않도록 차단
> **핵심**: 논문 식 (5)-(6). PyTorch는 `.detach()` 한 줄, JAX는 `jax.lax.stop_gradient()`

### 1.1 JAX: `gemma.py:Attention.__call__()`

- [ ] `stop_gradient_actions` flag를 `Attention`에 전달할 수 있도록 인터페이스 추가
- [ ] Action Expert가 Backbone K/V에 attend할 때 `jax.lax.stop_gradient()` 적용

```
# 논문 식 (5)-(6) — Stop Gradient Attention
P_ab = softmax(Q_a @ sg(K_b)^T)     # Action attends Backbone → stop grad on K
E_a  = P_ab @ sg(V_b) + P_aa @ V_a  # Action value → stop grad on V
E_b  = P_bb @ V_b                    # Backbone value → normal flow
```

**변경 파일**:
- `src/openpi/models/gemma.py` — `Attention.__call__()`, `Block.__call__()`
- `src/openpi/models/pi0.py` — flag를 `PaliGemma.llm()` 호출 시 전달

### 1.2 PyTorch: `gemma_pytorch.py`

- [ ] `compute_layer_complete()`에서 동일한 로직을 `.detach()`로 구현
- [ ] forward에 `stop_gradient_actions: bool` 파라미터 추가

**변경 파일**:
- `src/openpi/models_pytorch/gemma_pytorch.py`
- `src/openpi/models_pytorch/pi0_pytorch.py`

### 1.3 검증

- [ ] `stop_gradient_actions=True`일 때 Action Expert → Backbone 그래디언트가 0인지 단위 테스트
- [ ] 기존 `pi05=True`/`pi05=False` 성능 회귀 없음 확인

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

### 2A.3 손실 함수

- [ ] `compute_loss()` 확장: `L_total = L_CE(FAST tokens) + α * L_FlowMatching`
  - `joint_fast_training=False`일 때 기존 단일 손실 경로 유지
  - `α`는 configurable (기본값 1.0, stop gradient 시 자연스러운 균형)

### 2A.4 추론 경로

- [ ] Inference 시 FAST 토큰 경로는 사용하지 않고 Action Expert만으로 연속 액션 생성
- [ ] 기존 `sample_actions()` 로직 그대로 사용

### 2A.5 데이터 변환

- [ ] `transforms.py`에 `TokenizeJointInputs` transform 추가
  - FAST 토큰(훈련 시 사용) + 연속 액션 레이블 + 텍스트 입력을 하나의 Observation으로 변환

### 2A.6 FIAG 검증

- [ ] `joint_fast_training=True` + `stop_gradient_actions=False` 동작 확인
- [ ] `joint_fast_training=True` + `stop_gradient_actions=True` 동작 확인
- [ ] `joint_fast_training=False`일 때 FAST 관련 연산이 0 cost인지 확인

---

## Phase 2B: Subtask / HL 텍스트 예측 (`cotrain_subtask_data`)

> **목표**: 로봇 액션 데이터 없이도 subtask 문자열 + 웹 데이터(VQA/캡셔닝/탐지)를 텍스트 CE loss로 학습
> **참고**: π0.5 논문 Section IV-C (HL, WD), 식 (1)

### 2B.1 텍스트 출력 헤드 활성화

- [ ] PaliGemma의 `embedder.decode()` (logits → vocab)를 텍스트 CE loss 계산에 연결
- [ ] Observation에 `text_target_tokens`, `text_target_mask` 필드 추가
- [ ] `compute_loss()`에 텍스트 CE loss 경로 추가

**Loss 구성**:
```
L_CE(text) = -Σ mask_j · log p(target_token_j | prefix)
  where mask_j = 1 for text target positions, 0 otherwise
```

### 2B.2 Subtask 데이터 포맷

- [ ] π0.5 포맷: `"Task: clean the bedroom, State: ...; Subtask: pick up pillow|"`
- [ ] HL 데이터: observation + high-level prompt → subtask text target
- [ ] WD 데이터: observation + task prompt → text answer (caption, VQA answer, bounding box)
- [ ] `M_act = 0` — Flow Matching loss 미적용

### 2B.3 손실 마스킹 (`M_act`)

- [ ] 데이터 타입별 loss 적용 규칙:

| 데이터 타입 | Text CE (`M_ℓ`) | FAST CE | Flow Matching (`M_act`) |
|---|---|---|---|
| Robot Action Data | 1 | 1 | 1 |
| Subtask/HL Data | 1 | 0 | 0 |
| Web Data (WD) | 1 | 0 | 0 |

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

### 3.1 데이터셋 확장

- [ ] VLM 데이터셋 포맷 정의 (이미지 + 텍스트, 액션 없음)
- [ ] `DataConfig`에 VLM 데이터 경로/소스 추가
- [ ] 데이터 로더에서 VLM 데이터 + 액션 데이터 혼합 샘플링

**변경 파일**: `src/openpi/training/data_loader.py`, `src/openpi/training/config.py`

### 3.2 검증

- [ ] VLM 데이터만 있는 배치에서 Flow Matching loss가 0인지 확인
- [ ] 액션 데이터만 있는 배치에서 기존과 동일한 손실인지 확인

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
        cotrain_vlm_data=True,
        cotrain_subtask_data=True,
    ),
    ...
)
```

### 4.2 End-to-end 테스트

- [ ] Full KI config로 training loop 시작/1스텝 실행 성공
- [ ] Full KI config로 `sample_actions()` 추론 성공
- [ ] 기존 π0, π0.5, π0-FAST config 회귀 없음

### 4.3 성능 벤치마크

- [ ] `stop_gradient_actions` on/off 학습 속도 비교 (목표: ~5x faster convergence)
- [ ] `joint_fast_training` on/off 수렴 속도 비교
- [ ] `cotrain_subtask_data` on/off Language following rate 측정
- [ ] `cotrain_vlm_data` on/off OOD 일반화 성능 비교

---

## Phase 5: 문서화 (사후)

- [ ] `docs/ki_usage.md` — KI flag 사용 가이드
- [ ] CHANGELOG 업데이트

---

## 변경 파일 총괄

| 파일 | Phase | 변경 유형 |
|---|---|---|
| `src/openpi/models/pi0_config.py` | 0 | Flag 5개 추가 |
| `src/openpi/models/gemma.py` | 1 | Stop Gradient (JAX) |
| `src/openpi/models/pi0.py` | 1, 2A, 2B | Stop Gradient 전달, Joint Training, Subtask CE, 2단계 추론 |
| `src/openpi/models/model.py` | 2A, 2B | Observation 필드 확장 (`text_target_tokens`, `text_target_mask`) |
| `src/openpi/models/tokenizer.py` | 2B | Subtask 텍스트 토크나이즈 지원 |
| `src/openpi/transforms.py` | 2A, 2B, 3 | Joint/Subtask/VLM transform |
| `src/openpi/models_pytorch/gemma_pytorch.py` | 1 | Stop Gradient (PyTorch) |
| `src/openpi/models_pytorch/pi0_pytorch.py` | 1, 2A, 2B | Stop Gradient 전달, Joint Training, Subtask CE |
| `src/openpi/training/config.py` | 0, 3, 4 | Config preset |
| `src/openpi/training/data_loader.py` | 3 | VLM + Subtask 데이터셋 지원 |

---

## 구현 원칙

1. **기존 코드 불변**: 모든 신규 기능은 flag가 `False`일 때 기존 코드와 동일한 제어 흐름을 따른다.
2. **JAX / PyTorch 동시 변경**: 두 구현체에 동일한 기능을 적용한다.
3. **단위 테스트 우선**: 각 Phase 완료 후 해당 flag에 대한 단위 테스트를 작성한다.
4. **작은 PR**: 각 Phase를 독립적인 PR로 제출 가능하도록 설계한다.
5. **Dual Loss 필수**: CE loss(Backbone) + Flow Matching loss(Action Expert)는 둘 중 하나라도 제거하면 성능이 저하된다. `joint_fast_training=False`일 때도 Phase 2B의 텍스트 CE는 유지되어야 Backbone이 NTP 신호를 받을 수 있다.

## 주의사항

- `docs/plan.md`의 체크리스트 항목을 구현 완료 시 반드시 갱신할 것
- 각 Phase 시작 전 `docs/ki_analysis.md`의 해당 섹션을 재확인할 것
- PyTorch/JAX 간 구현 차이가 없도록 주의할 것
- 모델 아키텍처 변경 시 `pi0_test.py` 및 `model_test.py` 테스트 갱신할 것
- Inference 시 FAST 토큰/Subtask 토큰은 Action Expert와 attend 불가 (Attention Mask 검증 필수)

---

## Flag 조합 시나리오

| Config | Train Fast | Run Fast | Generalize | Subtask |
|---|---|---|---|---|
| π0 (전부 False) | ❌ 느림 | ✅ | ❌ | ❌ |
| + stop_gradient_actions | ✅ 5x↑ | ✅ | ✅ | ❌ |
| + joint_fast_training | ✅ | ✅ | ✅ | ❌ |
| + cotrain_subtask_data | ✅ | ✅ | ✅ | ✅ |
| + cotrain_vlm_data | ✅ | ✅ | ✅✅ | ✅ |
| **π0.5 + KI (Full)** | ✅✅ | ✅ | ✅✅ | ✅ |