# openpi — 작업 지침

## Knowledge Insulation 구현

[Knowledge Insulation 논문](https://www.pi.website/research/knowledge_insulation)의 내용을 openpi 코드베이스에 구현.
π0.5의 Subtask(HL) 학습도 KI와 호환되므로 함께 구현.

### 참조 문서

| 문서 | 용도 |
|---|---|
| `docs/ki_analysis.md` | 기구현/미구현 분석, Feature Flag 설계, 모듈 영향도 |
| `docs/plan.md` | Phase별 체크리스트, 구현 상세, 파일 목록, 검증 항목 |
| https://www.engineeringmaxxing.com/veanors/papers/pi05-ki.html | KI 인사이트 (.detach() 한 줄, Dual Loss 필수) |
| https://arxiv.org/pdf/2504.16054 | π0.5 Subtask/HL 학습 명세 |

### Workflow

```
Phase 1 ──→ Phase 2A ──→ Phase 2B ──→ Phase 3 ──→ Phase 4
(StopGrad)   (Joint)     (Subtask)   (VLM Co)   (통합)
```

각 Phase 진입 전:
1. `docs/plan.md`의 해당 Phase 체크리스트 확인
2. `docs/ki_analysis.md`의 관련 모듈 영향도 확인
3. 구현 완료 후 `docs/plan.md` 체크박스를 `- [x]`로 갱신

### 원칙

- Flag 기본값 `False` → 기존 코드 경로 절대 변경 금지
- JAX + PyTorch 동시 변경
- Phase마다 단위 테스트 작성
- Dual Loss(CE + FM) 보존 — 어느 한쪽만 제거하면 성능 저하

### Feature Flags

`src/openpi/models/pi0_config.py` — `Pi0Config`:

```python
stop_gradient_actions: bool = False   # Phase 1
joint_fast_training: bool = False     # Phase 2A
cotrain_subtask_data: bool = False    # Phase 2B
cotrain_vlm_data: bool = False        # Phase 3
```