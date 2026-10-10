from afllm.train import cosine_lr

LR = 3e-4
MIN_LR = 3e-5
WARMUP = 50
MAX_STEPS = 500


def test_warmup_reaches_lr_at_last_warmup_step():
    # Linear ramp: at step = warmup_steps - 1 (the last warmup step), the
    # schedule should have reached exactly `lr`.
    assert cosine_lr(WARMUP - 1, WARMUP, MAX_STEPS, LR, MIN_LR) == LR


def test_warmup_first_step_is_lr_over_warmup_steps():
    assert cosine_lr(0, WARMUP, MAX_STEPS, LR, MIN_LR) == LR / WARMUP


def test_warmup_to_cosine_handoff_is_continuous():
    # No discontinuity at the warmup/cosine boundary: both sides agree at lr.
    last_warmup = cosine_lr(WARMUP - 1, WARMUP, MAX_STEPS, LR, MIN_LR)
    first_cosine = cosine_lr(WARMUP, WARMUP, MAX_STEPS, LR, MIN_LR)
    assert last_warmup == LR
    assert first_cosine == LR
    assert last_warmup == first_cosine


def test_cosine_decays_toward_min_lr_by_last_step():
    last_step_lr = cosine_lr(MAX_STEPS - 1, WARMUP, MAX_STEPS, LR, MIN_LR)
    assert MIN_LR <= last_step_lr < LR
    # Should be very close to min_lr (cosine has nearly bottomed out).
    assert abs(last_step_lr - MIN_LR) < 1e-3 * LR


def test_lr_clamps_to_min_lr_at_and_past_max_steps():
    assert cosine_lr(MAX_STEPS, WARMUP, MAX_STEPS, LR, MIN_LR) == MIN_LR
    assert cosine_lr(MAX_STEPS + 100, WARMUP, MAX_STEPS, LR, MIN_LR) == MIN_LR


def test_lr_is_monotonic_non_increasing_after_warmup():
    prev = cosine_lr(WARMUP, WARMUP, MAX_STEPS, LR, MIN_LR)
    for step in range(WARMUP + 1, MAX_STEPS):
        current = cosine_lr(step, WARMUP, MAX_STEPS, LR, MIN_LR)
        assert current <= prev + 1e-12, f"lr increased at step {step}"
        prev = current


def test_cosine_lr_is_a_pure_function_of_step():
    # This is what makes resume preserve the intended LR schedule: the
    # same step must always produce the same lr, independent of call order
    # or any external state.
    for step in [0, 1, WARMUP - 1, WARMUP, WARMUP + 1, MAX_STEPS // 2, MAX_STEPS - 1]:
        a = cosine_lr(step, WARMUP, MAX_STEPS, LR, MIN_LR)
        b = cosine_lr(step, WARMUP, MAX_STEPS, LR, MIN_LR)
        assert a == b
