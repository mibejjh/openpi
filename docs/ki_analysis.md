# Knowledge Insulation — 코드베이스 분석

> 논문: _Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better_
> Physical Intelligence, 2025.05.28
> 대상 저장소: `openpi` (https://github.com/Physical-Intelligence/openpi)

---

## 1. 프로젝트 구조

```
src/openpi/
├── models/
│   ├── model.py            # BaseModel, Observation, Actions
│   ├── pi0.py              # π0: Flow Matching Action Expert
│   ├── pi0_config.py       # Pi0Config (+ pi05 flag)
│   ├── pi0_fast.py         # π0-FAST: Autoregressive Discrete Tokens
│   ├── gemma.py            # JAX: Dual-expert Mixture-of-Transformers
│   ├── gemma_fast.py       # JAX: FAST 전용 Gemma
│   ├── siglip.py           # Vision encoder (SigLIP)
│   ├── tokenizer.py        # PaligemmaTokenizer, FASTTokenizer
│   ├── lora.py             # LoRA fine-tuning
│   └── utils/fsq_tokenizer.py
├── models_pytorch/
│   ├── pi0_pytorch.py      # PyTorch: π0 동일 아키텍처
│   ├── gemma_pytorch.py    # PyTorch: PaliGemmaWithExpertModel
│   └── transformers_replace/  # HuggingFace transformers 패치
├── training/
│   ├── config.py           # TrainConfig (+ pi05 사전 설정)
│   ├── data_loader.py
│   └── ...
├── transforms.py           # 데이터 변환 파이프라인
├── policies/               # 환경별 정책 래퍼
└── serving/                # 추론 서버
```

---

## 2. 이미 구현된 항목 (π0.5 베이스라인)

| # | 기능 | 파일 | 비고 |
|---|---|---|---|
| 1 | **Dual-expert architecture** (PaliGemma 2B + Action Expert 300M) | `gemma.py:Attention` | Mixture-of-Transformers 방식으로 QKV concatenate |
| 2 | **π05 flag** | `pi0_config.py:36` | adaRMSNorm + 이산 상태 입력(text token) 활성화 |
| 3 | **Flow Matching Action Expert** | `pi0.py:compute_loss()`, `sample_actions()` | `Beta(1.5, 1)` 시간 샘플링, Euler 적분 추론 |
| 4 | **FAST Autoregressive Model** | `pi0_fast.py` | DCT 기반 토크나이저 + CE loss, 별도 모델 클래스 |
| 5 | **FAST Tokenizer** | `tokenizer.py`, `utils/fsq_tokenizer.py` | DCT + 양자화 + BPE |
| 6 | **PyTorch 버전** | `pi0_pytorch.py`, `gemma_pytorch.py` | JAX와 1:1 대응 |
| 7 | **Gradient Checkpointing** | `pi0_pytorch.py` | 메모리 최적화 |
| 8 | **LoRA fine-tuning** | `lora.py` | Attention / FFN low-rank adaptation |
| 9 | **Gradient Checkpointing** | `pi0_pytorch.py` | Layer-level checkpointing |

---

## 3. 미구현 항목 (KI 3대 요소)

### 3.1 Stop Gradient: Action Expert → VLM Backbone

**현재**: `gemma.py:Attention.__call__()`에서 모든 expert의 Q,K,V를 concatenate하여 joint attention.
그래디언트가 자유롭게 역전파됨.

```python
# gemma.py (현재)
q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))
```

**필요 (논문 식 5-6)**:
```
P_ab = softmax(Q_a @ sg(K_b)^T)    # Action attends to Backbone → stop grad
E_a  = P_ab @ sg(V_b) + P_aa @ V_a # Backbone Value → stop grad
```

### 3.2 FAST 토큰 + Flow Matching Joint Training

**현재**: `Pi0`(flow matching)과 `Pi0FAST`(cross-entropy)가 완전히 분리된 모델 클래스.

| | π0 | π0-FAST |
|---|---|---|
| Loss | Flow Matching MSE | Cross-Entropy |
| Model class | `Pi0` | `Pi0FAST` |
| Data transform | `TokenizePrompt` | `TokenizeFASTInputs` |

**필요**: 하나의 모델에서 `L = L_CE(FAST tokens) + α · L_FlowMatching` 동시 계산.
Attention Mask에서 FAST 토큰과 연속 액션 토큰이 서로 attend 불가하도록 수정.

### 3.3 VLM 데이터 Co-training

**현재**: 로봇 액션 데이터만 사용.

**필요**: 이미지 캡셔닝, VQA, 객체 탐지, 로봇 계획 데이터 추가.
비-액션 데이터는 CE loss만, 액션 데이터는 CE + Flow Matching.

---

## 4. Feature Flag 설계

```python
@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    # 기존 필드 (변경 없음)
    pi05: bool = False

    # 신규: Knowledge Insulation flags (기본값 False → 기존 코드 호환)
    stop_gradient_actions: bool = False   # Action Expert → VLM Backbone 그래디언트 차단
    joint_fast_training: bool = False     # FAST 토큰 + 연속 액션 공동 학습
    cotrain_vlm_data: bool = False        # VLM 데이터 co-training 활성화
```

### Flag 조합 시나리오

| Config | 동작 |
|---|---|
| 전부 기본값 (π0) | 기존 π0 완전 호환 |
| `pi05=True` (π0.5) | 기존 π0.5 완전 호환 |
| + `stop_gradient_actions=True` | π0.5 + Stop Gradient |
| + `joint_fast_training=True` | π0.5 + Joint Training |
| + `cotrain_vlm_data=True` | π0.5 + VLM Co-train |
| 모두 True | π0.5 + KI (Full) |

---

## 5. 변경 영향도 (모듈별)

| 모듈 | 영향도 | 변경 내용 |
|---|---|---|
| `pi0_config.py` | 낮음 | Flag 3개 추가 |
| `gemma.py:Attention` | 중간 | Stop Gradient 분기, Key/Value detach |
| `gemma_pytorch.py:compute_layer_complete` | 중간 | 동일 (PyTorch) |
| `pi0.py:Pi0` | 중-상 | Joint Training loss, FAST 토큰 처리, config 전달 |
| `pi0_pytorch.py:PI0Pytorch` | 중-상 | 동일 (PyTorch) |
| `transforms.py` | 낮음 | VLM co-training용 transform 추가 |
| `training/config.py` | 낮음 | 신규 config preset |
| `model.py:Observation` | 낮음 | FAST 토큰 필드 추가 (이미 pi0_fast에서 사용 중) |

---

## 6. 구현 우선순위

| 순위 | 기능 | 난이도 | 핵심 변경 파일 |
|---|---|---|---|
| **P1** | `stop_gradient_actions` | 중 | `gemma.py`, `gemma_pytorch.py` |
| **P2** | `joint_fast_training` | 상 | `pi0.py`, `pi0_pytorch.py`, `gemma.py`, `transforms.py` |
| **P3** | `cotrain_vlm_data` | 중 | `data_loader.py`, `transforms.py`, 손실 마스킹 |

---

_분석일: 2026-07-25_