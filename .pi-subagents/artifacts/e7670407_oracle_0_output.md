I have verified all my findings against the code. Here is my review.

## Review: `docs/plan.md`

The plan is **well-structured and directionally correct** — flag design, phase decomposition, and the KI/Subtask compatibility argument are sound. However, verification against the code found **3 blockers** and **several gaps** that will stall implementation if not addressed first.

---

### Blockers (must fix in the plan before coding)

#### B1. `pi0.py` has no text output head — Phase 2A/2B are underscoped

The plan's Phase 2B says "PaliGemma의 `embedder.decode()`를 텍스트 CE loss 계산에 연결" as if this is a wiring task. It is not.

- `Pi0` uses `models/gemma.py`, whose `Module.__call__` returns **hidden states only** (`gemma.py:390-395`). There is no `decode=`, `pre_logits=`, `return_prelogits=`, or logits path.
- `models/gemma_fast.py` (used by `Pi0FAST`) *does* have this — `__call__` accepts `pre_logits`/`return_prelogits` and calls `embedder.decode(x)` (`gemma_fast.py:303-341`).
- `grep decode|logits src/openpi/models/pi0.py` returns **nothing**.

So Phase 2A (FAST CE loss) *and* Phase 2B (text CE loss) both require adding a prelogits/logits return path to `gemma.py:Module.__call__`, mirroring `gemma_fast.py`. This is a change to the shared transformer module, not a per-phase detail — and it's a prerequisite for **both** 2A and 2B, yet appears in neither phase's file list.

**Fix:** add a `Phase 2.0` — "extend `gemma.py:Module` with `return_prelogits`/`pre_logits` logits path (port from `gemma_fast.py`)" — and add `gemma.py` to the Phase 2A/2B changed-file table (currently only listed under Phase 1).

#### B2. Loss shape contract breaks the type annotation

`BaseModel.compute_loss` is annotated `-> at.Float[at.Array, "*b ah"]` (`model.py:273-280`), and `pi0.py` is decorated with `@at.typecheck` at module level.

- `pi0.compute_loss` returns `(b, action_horizon)`
- `pi0_fast.compute_loss` returns `(b,)`

The plan's `L_total = L_CE + α·L_FM` combines these two incompatible shapes and never says how. Broadcasting `(b,)` against `(b, ah)` silently replicates the CE loss `ah` times, which **rescales the CE term by `action_horizon` (50×)** relative to the intended `α`. That would silently invalidate every `α` tuning experiment.

**Fix:** the plan must specify the reduction contract explicitly — e.g. reduce both to `(b,)` and relax the base annotation, or keep `(b, ah)` and divide the CE term by `ah`. Also note `train.py:151` does `jnp.mean(chunked_loss)`, so per-sample weighting differs between the two choices when `M_act` masking is active (a masked-out sample must not dilute the mean).

#### B3. Stop-gradient must be inside `Attention`, not at the `Block`/`Module` boundary

The plan's Phase 1 pseudocode is correct in spirit, but the listed change surface (`Block.__call__`, "flag를 `PaliGemma.llm()` 호출 시 전달") understates the difficulty:

- `Attention.__call__` **concatenates all experts' Q/K/V into single tensors** (`gemma.py:196`) and computes one fused softmax. There is no separate `P_ab` to intercept — the paper's block-decomposed form does not exist in this code.
- Implementing `sg(K_b)`/`sg(V_b)` requires building **two K/V variants** (grad and no-grad) and either (a) running attention twice, or (b) splitting the fused einsum into per-block products. Both change FLOPs/memory, not just gradient routing.
- The out-projection loop slices `encoded[:, start:end]` assuming contiguous per-expert blocks (`gemma.py:233-246`) — any restructuring must preserve that layout.
- `Block` is wrapped in `nn.remat(..., static_argnums=(5,))` and `nn.scan(..., in_axes=(0, broadcast, broadcast, broadcast, broadcast))` (`gemma.py:345-372`). Adding a positional arg to `Block.__call__` **breaks both** unless the flag is passed as a constructor attribute (like `dropout`) or added to `static_argnums`/`in_axes`. This is the single most likely thing to blow up on first attempt.

**Fix:** state explicitly that the flag is a **module attribute** on `Block`/`Attention` (not a call argument), and add a sub-item for the two-pass-vs-split-einsum decision with its memory cost.

---

### Gaps

| # | Gap | Detail |
|---|---|---|
| G1 | **Token budget not addressed** | `pi05` sets `max_token_len=200` (`pi0_config.py:39`); `Pi0FASTConfig` uses `250` and real configs use `180` *just for prompt+state+FAST actions*. Phase 2A adds FAST tokens **and** Phase 2B adds subtask targets into the same 200-token prefix. This likely needs `max_token_len` ≈ 300–400, which changes memory and invalidates the "기존 config 회귀 없음" claim if the default is touched. No phase mentions this. |
| G2 | **`get_freeze_filter` uses `.*llm.*_1.*`** | Action-expert params are identified by the `_1` name suffix (`pi0_config.py:93`). If stop-gradient is implemented by *any* renaming or by adding a third expert, the LoRA freeze filter breaks silently. Add a regression check. |
| G3 | **`ModelTransformFactory` dispatches on `model_type`** | `config.py:114-160` matches `PI0`/`PI05`/`PI0_FAST`. The new flags don't change `model_type`, so a `pi05=True, joint_fast_training=True` config still gets the plain `TokenizePrompt` transform and will **never receive FAST tokens** — training would silently run FM-only. The plan's `TokenizeJointInputs` (2A.5) is listed but the *dispatch* change is not. This is a silent-failure path, worth calling out. |
| G4 | **Phase 3 vs 2B overlap is unresolved** | Phase 2B.2 already assigns WD (caption/VQA/bbox) data to `cotrain_subtask_data`, and Phase 3 assigns the *same* WD data to `cotrain_vlm_data`. Two flags own one data category. Either split cleanly (2B = HL/subtask only, 3 = web only) or merge the flags. As written, `cotrain_vlm_data` has no distinct behavior once 2B ships. |
| G5 | **Stop-gradient + KV-cache at inference** | `sample_actions` prefills a KV cache then runs the expert against it (`pi0.py:243`). Under `no_grad` inference stop-gradient is a no-op, so this is safe — but the plan should state it so nobody "implements" sg in the sampling path and diverges JAX/PyTorch. |
| G6 | **No PyTorch numerical-parity test** | Principle #2 says "JAX/PyTorch 동시 변경" but validation never checks the two produce *matching* losses/grads. Given `gemma_pytorch.py` has its own hand-rolled fused-attention loop, drift is very likely. Add a parity test to Phase 1.3 and 2A.6. |

---

### Assumption challenges

**"Stop Gradient는 `.detach()` 한 줄로 구현 가능"** — this quote is from a third-party explainer and is **misleading for this codebase**. It is true for architectures where the expert consumes a discrete feature tensor from the backbone. Here, experts are fused inside one attention op (B3). Keeping this line in the plan will set wrong expectations about Phase 1's size. Recommend replacing it with a note that the fused-QKV design makes it a ~50-line surgical change to `Attention`, not one line.

**"목표: ~5x faster convergence"** — the KI paper reports **7.5×** fewer steps vs π0 on bussing; the "5x" figure comes from the explainer page. Also, this is only reproducible with the *full* recipe (FAST CE + VLM co-training) on a large multi-embodiment mixture. Setting it as a Phase 4.3 target for a single-embodiment run invites a false negative. Recommend reframing as "relative improvement vs. same-data π0 baseline," no absolute multiplier.

**Phase 4.3's `stop_gradient_actions` on/off comparison is not measurable in isolation** — the paper is explicit that gradient-blocking *alone* is insufficient (backbone gets no robot signal and performance collapses; cf. the "frozen backbone → 0%" result). So the Flag-조합 table row "`+ stop_gradient_actions` → ✅ 5x↑, Generalize ✅" is wrong: that configuration should be **worse** than baseline, not better. The first row that can legitimately claim the triple win is `stop_gradient_actions + joint_fast_training` together.

**Principle #5 is self-contradictory** — it says "`joint_fast_training=False`일 때도 Phase 2B의 텍스트 CE는 유지되어야" but `cotrain_subtask_data` defaults to `False`, and Principle #1 mandates flags-off ⇒ legacy path. So the default config has no CE at all. If the real invariant is "never enable `stop_gradient_actions` without a CE source," encode it as a `__post_init__` validation error, not a prose principle.

---

### Recommended plan edits (priority order)

1. Insert **Phase 2.0**: add logits/prelogits path to `gemma.py:Module` (prereq for 2A + 2B).
2. Add to **Phase 0**: `max_token_len` budget re-derivation; `__post_init__` guard rejecting `stop_gradient_actions=True` with no CE source.
3. Rewrite **Phase 1.1** to specify flag-as-module-attribute (remat/scan safety) and the two-pass vs split-einsum decision.
4. Add **loss reduction contract** subsection to Phase 2A.3 (shape + `α` normalization + `M_act`-aware mean).
5. Add **`ModelTransformFactory` dispatch** item to Phase 2A.5 / 2B.2.
6. Resolve **G4** (2B vs 3 data ownership).
7. Add **JAX↔PyTorch parity test** to 1.3 and 2A.6.
8. Correct the **Flag 조합 시나리오** table (stop-grad-alone row) and soften the 5× claim.

Overall: the architecture reasoning is solid and the KI+Subtask compatibility argument holds up — the shared `L_CE(text) + α·L_FM(action)` framing is correct and the attention-mask table is right. The weakness is that the plan describes the *target design* accurately but underestimates the *fused-attention* and *missing-logits-head* realities of this specific codebase, and has one likely-wrong benchmark expectation.