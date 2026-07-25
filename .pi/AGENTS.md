# openpi — 작업 지침

## Knowledge Insulation 구현

[Knowledge Insulation 논문](https://www.pi.website/research/knowledge_insulation)의 내용을 openpi 코드베이스에 구현.
π0.5의 Subtask(HL) 학습도 KI와 호환되므로 함께 구현.

### 참조 문서

| 문서 | 용도 |
|---|---|
| `docs/ki_analysis.md` | 기구현/미구현 분석, Feature Flag 설계, 모듈 영향도 |
| `docs/plan.md` | Phase별 체크리스트, 구현 상세, 파일 목록, 검증 항목 |
| https://www.engineeringmaxxing.com/veanors/papers/pi05-ki.html | KI 개념 참고 (Dual Loss 필수 등) |
| https://arxiv.org/pdf/2504.16054 | π0.5 Subtask/HL 학습 명세 |

### Workflow

```
Phase 0 ──→ Phase 1 ──→ Phase 2.0 ──→ Phase 2A ──→ Phase 2B ──→ Phase 3 ──→ Phase 4
(Flags)    (StopGrad)  (Logits)     (Joint)     (Subtask)   (VLM Co)   (통합)
```

**주의**: Phase 2.0은 2A/2B의 blocking prerequisite. `gemma.py`에 logits/prelogits 경로가 없으면 CE loss 자체가 불가능.
**중요**: `stop_gradient_actions=True`는 반드시 `joint_fast_training=True` 또는 `cotrain_subtask_data=True`와 페어링. 단독 사용 금지 (`__post_init__` guard로 강제).

각 Phase 진입 전:
1. `docs/plan.md`의 해당 Phase 체크리스트 확인
2. `docs/ki_analysis.md`의 관련 모듈 영향도 확인
3. 구현 완료 후 `docs/plan.md` 체크박스를 `- [x]`로 갱신

### 원칙

- Flag 기본값 `False` → 기존 코드 경로 절대 변경 금지
- JAX + PyTorch 동시 변경 + parity test
- Phase마다 단위 테스트 작성
- `stop_gradient_actions` 단독 사용 금지 (CE source 필요)

### Feature Flags (4개)

`src/openpi/models/pi0_config.py` — `Pi0Config`:

```python
stop_gradient_actions: bool = False   # Phase 1
joint_fast_training: bool = False     # Phase 2A
cotrain_subtask_data: bool = False    # Phase 2B (HL only, WD는 Phase 3)
cotrain_vlm_data: bool = False        # Phase 3 (WD: caption/VQA/bbox)
```