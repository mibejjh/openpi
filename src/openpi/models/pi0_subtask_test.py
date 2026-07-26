import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.pi0_config as _pi0_config
import openpi.transforms as _transforms
from openpi.models import tokenizer as _tokenizer


def test_config_validation():
    """cotrain_subtask_data requires pi05=True."""
    with _raises(ValueError, match="cotrain_subtask_data requires pi05=True"):
        _pi0_config.Pi0Config(cotrain_subtask_data=True, pi05=False)

    # Should not raise
    _pi0_config.Pi0Config(cotrain_subtask_data=True, pi05=True)
    _pi0_config.Pi0Config(cotrain_subtask_data=False, pi05=False)


def test_tokenize_subtask_transform():
    """TokenizeSubtask produces correct masks."""
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=200)
    transform = _transforms.TokenizeSubtask(tokenizer=tokenizer)

    data = {
        "prompt": np.array(["Task: clean the bedroom"]),
        "subtask": np.array(["pick up pillow"]),
    }
    result = transform(data)

    assert "tokenized_prompt" in result
    assert "tokenized_prompt_mask" in result
    assert "token_ar_mask" in result
    assert "token_loss_mask" in result

    prompt_tokens = result["tokenized_prompt"]
    ar_mask = result["token_ar_mask"]
    loss_mask = result["token_loss_mask"]

    assert prompt_tokens.ndim == 2  # (b, l)
    assert ar_mask.shape == prompt_tokens.shape
    assert loss_mask.shape == prompt_tokens.shape

    # Find split point: ar_mask 0→1 boundary
    context_len = int(jnp.argmax(ar_mask[0]))
    assert context_len > 0, "Expected some context tokens"
    assert ar_mask[0, context_len:].all(), "Target tokens should be causal (1)"
    assert not ar_mask[0, :context_len].any(), "Context tokens should be bidir (0)"
    assert not loss_mask[0, :context_len].any(), "Context tokens should have no loss"
    assert loss_mask[0, context_len:].all(), "Target tokens should have loss"


def test_cotrain_fake_data_smoke():
    """Smoke test: loss computation shape with cotrain_subtask_data."""
    config = _pi0_config.Pi0Config(pi05=True, cotrain_subtask_data=True)
    assert config.cotrain_subtask_data
    assert config.ce_loss_weight == 1.0
    assert config.max_token_len == 200  # pi05 default


def test_ce_loss_weight_default():
    """ce_loss_weight defaults to 1.0."""
    config = _pi0_config.Pi0Config(pi05=True)
    assert config.ce_loss_weight == 1.0

    config_custom = _pi0_config.Pi0Config(pi05=True, ce_loss_weight=0.5)
    assert config_custom.ce_loss_weight == 0.5


def _raises(exc_class, match=None):
    """Context manager for asserting exception."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        try:
            yield
        except exc_class as e:
            if match and match not in str(e):
                raise AssertionError(
                    f"Expected exception to match {match!r}, got: {e}"
                ) from e
        else:
            raise AssertionError(f"Expected {exc_class.__name__} was not raised")

    return _ctx()