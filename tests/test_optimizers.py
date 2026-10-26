"""Independent optimizer recurrences, derivatives, state and experiment boundaries."""

import copy
import hashlib
import json
import math

import numpy as np
import pytest

from test_numerics import central_difference, reference_functions


def functions():
    return reference_functions("momentum-and-adam")


def assert_nested_equal(left, right):
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def scalar_transition(parameters, gradients, state, config, rate):
    """Python scalar recurrence; no displayed transition, norm or correction helper."""
    result, next_state = copy.deepcopy(parameters), copy.deepcopy(state)
    next_state["t"] += 1
    t = next_state["t"]
    norm = math.hypot(*(float(v) for key in sorted(gradients) for v in gradients[key].ravel()))
    factor = 1. if config["clip"] is None or norm == 0 else min(1., config["clip"] / norm)
    for key in parameters:
        for index in np.ndindex(parameters[key].shape):
            g = float(gradients[key][index]) * factor
            old = float(parameters[key][index])
            if config["kind"] == "sgd":
                direction = g
            elif config["kind"] == "momentum":
                direction = config["momentum"] * float(state["first"][key][index]) + g
                next_state["first"][key][index] = direction
            else:
                first = config["beta1"] * float(state["first"][key][index]) + (1 - config["beta1"]) * g
                second = config["beta2"] * float(state["second"][key][index]) + (1 - config["beta2"]) * g * g
                next_state["first"][key][index], next_state["second"][key][index] = first, second
                corrected_first = first / (1 - config["beta1"] ** t)
                corrected_second = second / (1 - config["beta2"] ** t)
                direction = corrected_first / (math.sqrt(corrected_second) + config["epsilon"])
            result[key][index] = old - rate * direction
            if config["kind"] == "adamw" and key in config["decay_keys"]:
                result[key][index] -= rate * config["decay"] * old
    return result, next_state


@pytest.mark.parametrize("kind", ["sgd", "momentum", "adam", "adamw"])
@pytest.mark.parametrize("clip", [None, .3, 10.])
def test_complete_state_transitions_against_independent_scalar_recurrence(kind, clip):
    f = functions()
    p = {"W": np.array([[.7, -.2], [.3, .8]]), "b": np.array([-.1, .4])}
    config = f["opt_config"](kind=kind, rate=.05, momentum=.8, beta1=.6, beta2=.7,
                             epsilon=.03, decay=.2 if kind == "adamw" else 0.,
                             decay_keys=["W"], clip=clip, budget=8, warmup=2)
    state = f["opt_new_state"](p, config)
    for j, rate in enumerate((.05, 0., .02)):
        gradients = {"W": np.array([[.2, -.5], [.8, .3]]) * (j + 1), "b": np.array([-.4, .7]) / (j + 1)}
        before = copy.deepcopy((p, gradients, state, config))
        expected_p, expected_state = scalar_transition(p, gradients, state, config, rate)
        actual_p, actual_state, diagnostics = f["opt_step"](p, gradients, state, config, rate)
        for key in p:
            np.testing.assert_allclose(actual_p[key], expected_p[key], atol=3e-15)
        for buffer in ("first", "second"):
            assert actual_state[buffer].keys() == expected_state[buffer].keys()
            for key in actual_state[buffer]:
                np.testing.assert_allclose(actual_state[buffer][key], expected_state[buffer][key], atol=3e-15)
        assert actual_state["t"] == expected_state["t"]
        norm = math.hypot(*(float(v) for value in gradients.values() for v in value.ravel()))
        assert diagnostics["original_gradient_norm"] == pytest.approx(norm)
        assert diagnostics["applied_gradient_norm"] == pytest.approx(norm if clip is None else min(norm, clip))
        assert diagnostics["update_norm"] == pytest.approx(math.hypot(*(float(v) for k in p for v in (actual_p[k] - p[k]).ravel())))
        for actual, saved in zip((p, gradients, state, config), before):
            assert_nested_equal(actual, saved)
        p, state = actual_p, actual_state


def test_hand_momentum_rate_convention_and_raw_adam_moments():
    f = functions(); p = {"w": np.array([1., -1.])}
    config = f["opt_config"](kind="momentum", rate=.1, budget=4, warmup=0)
    state = f["opt_new_state"](p, config)
    for g, rate in [([2., -4.], .1), ([-1., 2.], .025)]:
        p, state, _ = f["opt_step"](p, {"w": np.array(g)}, state, config, rate)
    np.testing.assert_allclose(state["first"]["w"], [.8, -1.6], atol=1e-15)
    np.testing.assert_allclose(p["w"], [.78, -.56], atol=1e-15)
    displacement_velocity = .9 * np.array([-.2, .4]) - .025 * np.array([-1., 2.])
    np.testing.assert_allclose(np.array([.8, -.6]) + displacement_velocity, [.645, -.29])
    assert not np.allclose(p["w"], [.645, -.29])
    config = f["opt_config"](kind="adam", rate=.1, epsilon=.01, budget=4, warmup=0)
    p = {"w": np.array([1., -1.])}; state = f["opt_new_state"](p, config)
    p, state, _ = f["opt_step"](p, {"w": np.array([2., -4.])}, state, config, .1)
    np.testing.assert_allclose(p["w"], [1 - .2 / 2.01, -1 + .4 / 4.01])
    p, state, _ = f["opt_step"](p, {"w": np.array([-1., 2.])}, state, config, .1)
    np.testing.assert_allclose(state["first"]["w"], [.08, -.16], atol=1e-16)
    np.testing.assert_allclose(state["second"]["w"], [.0496, .1984], atol=1e-16)


@pytest.mark.parametrize("kind", ["momentum", "adam", "adamw"])
def test_zero_gradient_and_zero_rate_are_not_skipped_transitions(kind):
    f = functions(); p = {"w": np.array([1.])}
    config = f["opt_config"](kind=kind, rate=.1, budget=5, warmup=0)
    empty = f["opt_new_state"](p, config)
    same_p, history, stats = f["opt_step"](p, {"w": np.array([2.])}, empty, config, 0.)
    np.testing.assert_array_equal(same_p["w"], p["w"])
    assert history["t"] == 1 and stats["update_norm"] == 0.
    moved, advanced, _ = f["opt_step"](same_p, {"w": np.zeros(1)}, history, config, .1)
    stationary, skipped, _ = f["opt_step"](p, {"w": np.zeros(1)}, empty, config, .1)
    assert moved["w"][0] < 1. and stationary["w"][0] == 1.
    assert advanced["t"] == 2 and skipped["t"] == 1


@pytest.mark.parametrize("beta", [0., .9, np.nextafter(1., 0.)])
def test_first_adam_step_correction_and_epsilon_placement(beta):
    f = functions(); p = {"w": np.zeros(2)}
    config = f["opt_config"](rate=.1, beta1=beta, beta2=beta, epsilon=.04, budget=3, warmup=0)
    actual, _, _ = f["opt_step"](p, {"w": np.array([.01, -.02])}, f["opt_new_state"](p, config), config, .1)
    np.testing.assert_allclose(actual["w"], [-.02, .1 * .02 / .06], atol=3e-16)
    wrong_inside_root = -.1 * .01 / math.sqrt(.01**2 + .04)
    assert not np.isclose(actual["w"][0], wrong_inside_root)


@pytest.mark.parametrize("scale", [1e-4, 2., 1e4])
def test_positive_gradient_scaling_requires_corresponding_epsilon_scaling(scale):
    f = functions(); p = {"w": np.zeros(2)}
    base = f["opt_config"](rate=.1, epsilon=.02, budget=5, warmup=0)
    scaled = f["opt_config"](rate=.1, epsilon=.02 * scale, budget=5, warmup=0)
    a, b = copy.deepcopy(p), copy.deepcopy(p)
    sa, sb = f["opt_new_state"](a, base), f["opt_new_state"](b, scaled)
    for g in ([.3, -.7], [-.2, .4], [0., .1]):
        a, sa, _ = f["opt_step"](a, {"w": np.array(g)}, sa, base, .1)
        b, sb, _ = f["opt_step"](b, {"w": scale * np.array(g)}, sb, scaled, .1)
        np.testing.assert_allclose(a["w"], b["w"], atol=2e-15)
    original_epsilon, _, _ = f["opt_step"](p, {"w": scale * np.array([.3, -.7])}, f["opt_new_state"](p, base), base, .1)
    first, _, _ = f["opt_step"](p, {"w": np.array([.3, -.7])}, f["opt_new_state"](p, base), base, .1)
    assert not np.allclose(first["w"], original_epsilon["w"])


@pytest.mark.parametrize("change", [
    {"kind": "unknown"}, {"rate": 0.}, {"rate": -1.}, {"rate": np.nan},
    {"epsilon": 0.}, {"epsilon": np.inf}, {"beta1": 1.}, {"beta2": -.1},
    {"momentum": 1.}, {"decay": -.1}, {"decay": .1}, {"clip": 0.},
    {"budget": True}, {"budget": 1001}, {"warmup": 600}, {"batch": 0},
    {"decay_keys": ["w", "w"]}, {"decay_keys": "w"},
    {"kind": "adamw", "rate": 1., "decay": 1.},
])
def test_invalid_optimizer_contracts_are_rejected(change):
    with pytest.raises(ValueError):
        functions()["opt_config"](**change)


@pytest.mark.parametrize("bad", [
    {}, {"w": []}, {"w": 1.}, {"w": [[[]]]}, {"": [1.]},
    {"w": [np.inf]}, {"w": [True]}, {"w": [1j]}, {"w": ["1"]},
    {"w": np.zeros(8193)},
])
def test_invalid_arrays_are_rejected(bad):
    with pytest.raises(ValueError):
        functions()["opt_arrays"](bad)


@pytest.mark.parametrize("case", ["missing", "extra", "broadcast", "nonfinite", "negative_second", "zero_counter_buffer", "bad_counter", "rate", "decay_name"])
def test_invalid_named_gradients_and_state_do_not_mutate_inputs(case):
    f = functions(); p = {"w": np.array([1., 2.]), "b": np.zeros(1)}
    config = f["opt_config"](rate=.1, budget=4, warmup=0)
    state = f["opt_new_state"](p, config)
    gradient = {"w": np.ones(2), "b": np.ones(1)}; rate = .1
    if case == "missing":
        del gradient["b"]
    elif case == "extra":
        gradient["other"] = np.ones(1)
    elif case == "broadcast":
        gradient["w"] = np.ones((1, 2))
    elif case == "nonfinite":
        gradient["w"][0] = np.nan
    elif case == "negative_second":
        state["second"]["w"][0] = -1.
    elif case == "zero_counter_buffer":
        state["first"]["w"][0] = 1.
    elif case == "bad_counter":
        state["t"] = 1.0
    elif case == "rate":
        rate = .2
    elif case == "decay_name":
        config["decay_keys"] = ["missing"]
    before = copy.deepcopy((p, state, config))
    with pytest.raises(ValueError):
        f["opt_step"](p, gradient, state, config, rate)
    for actual, expected in zip((p, state, config), before):
        assert_nested_equal(actual, expected)


def test_global_norm_scaling_clipping_before_square_and_failure_atomicity():
    f = functions(); p = {"w": np.zeros(2)}
    assert f["opt_norm"]({"w": np.array([1e200, -1e200])}) == pytest.approx(math.sqrt(2) * 1e200)
    config = f["opt_config"](rate=.1, clip=1., budget=4, warmup=0)
    before = f["opt_new_state"](p, config)
    updated, state, diagnostics = f["opt_step"](p, {"w": np.array([1e200, -1e200])}, before, config, .1)
    assert np.isfinite(updated["w"]).all() and diagnostics["applied_gradient_norm"] == pytest.approx(1.)
    assert np.isfinite(state["second"]["w"]).all()
    config["clip"] = None
    with pytest.raises(FloatingPointError):
        f["opt_step"](p, {"w": np.array([1e200, -1e200])}, before, config, .1)
    np.testing.assert_array_equal(p["w"], [0., 0.])
    assert before["t"] == 0 and not before["second"]["w"].any()
    with pytest.raises(FloatingPointError):
        f["opt_norm"]({"w": np.array([1.5e308, -1.5e308])})


def test_coupled_l2_and_decoupled_adamw_hand_values_and_mask():
    f = functions(); p = {"w": np.array([1., 2.]), "b": np.array([3.])}
    coupled = f["opt_config"](kind="adam", rate=.1, budget=4, warmup=0)
    decay = f["opt_config"](kind="adamw", rate=.1, decay=.1, decay_keys=["w"], budget=4, warmup=0)
    l2_grad = {"w": .1 * p["w"], "b": np.zeros(1)}
    zero_grad = {key: np.zeros_like(value) for key, value in p.items()}
    a, a_state, _ = f["opt_step"](p, l2_grad, f["opt_new_state"](p, coupled), coupled, .1)
    b, b_state, _ = f["opt_step"](p, zero_grad, f["opt_new_state"](p, decay), decay, .1)
    np.testing.assert_allclose(a["w"], [1 - .01 / (.1 + 1e-8), 2 - .02 / (.2 + 1e-8)])
    np.testing.assert_allclose(b["w"], [.99, 1.98])
    np.testing.assert_array_equal(a["b"], [3.]); np.testing.assert_array_equal(b["b"], [3.])
    assert a_state["second"]["w"].sum() > 0 and b_state["second"]["w"].sum() == 0
    sgd = f["opt_config"](kind="sgd", rate=.1, budget=4, warmup=0)
    direct, _, _ = f["opt_step"](p, l2_grad, f["opt_new_state"](p, sgd), sgd, .1)
    np.testing.assert_array_equal(direct["w"], b["w"])
    expected = p["w"].copy()
    state = f["opt_new_state"](p, decay)
    for rate in (.1, .03, 0.):
        p, state, _ = f["opt_step"](p, zero_grad, state, decay, rate)
        expected *= 1 - rate * .1
    np.testing.assert_allclose(p["w"], expected, atol=3e-16)


@pytest.mark.parametrize("budget,warmup", [(2, 0), (2, 1), (100, 10), (100, 99)])
def test_schedule_one_based_boundaries_and_fixed_budget(budget, warmup):
    f = functions(); config = f["opt_config"](rate=.2, budget=budget, warmup=warmup)
    actual = np.array([f["opt_rate"](config, t) for t in range(1, budget + 1)])
    expected = []
    for t in range(1, budget + 1):
        if warmup and t <= warmup:
            expected.append(.2 * t / warmup)
        else:
            progress = (t - warmup) / (budget - warmup) if warmup else (t - 1) / (budget - 1)
            expected.append(.1 * (1 + math.cos(math.pi * progress)))
    np.testing.assert_array_equal(actual, expected)
    assert actual[-1] == 0. and actual[0] > 0 and actual.max() == .2
    assert np.all(np.diff(actual[max(0, warmup - 1):]) <= 0)
    for bad in (0, budget + 1, True, 1.0):
        with pytest.raises(ValueError):
            f["opt_rate"](config, bad)


@pytest.mark.parametrize("masses", [[1., 3.], [1e308, 3e307], [0., 3.]])
def test_weighted_accumulation_against_scalar_union(masses):
    f = functions()
    gradients = [{"w": np.array([[2., -.5]]), "b": np.array([.2])}, {"w": np.array([[-2., .8]]), "b": np.array([-.3])}]
    result = f["opt_accumulate"](gradients, masses)
    a, b = (v / max(masses) for v in masses)
    for key in result:
        expected = (a * gradients[0][key] + b * gradients[1][key]) / (a + b)
        np.testing.assert_allclose(result[key], expected, atol=2e-16)
    np.testing.assert_allclose(f["opt_accumulate"]([{"w": np.array([2.])}, {"w": np.array([-2.])}], [1, 3])["w"], [-1.])


@pytest.mark.parametrize("masses", [[0, 0], [-1, 3], [1, np.nan], [1], [True, True]])
def test_invalid_accumulation_masses(masses):
    with pytest.raises(ValueError):
        functions()["opt_accumulate"]([{"w": np.ones(1)}, {"w": np.ones(1)}], masses)


def test_relative_mass_underflow_is_visible_and_adaptation_does_not_commute():
    f = functions()
    with pytest.raises(FloatingPointError):
        f["opt_accumulate"]([{"w": np.ones(1)}, {"w": np.ones(1)}], [1e308, 1e-308])
    with pytest.raises(FloatingPointError):
        f["opt_accumulate"]([{"w": np.ones(1)}] * 3, [1., 1., np.nextafter(0., 1.)])
    p = {"w": np.zeros(1)}; config = f["opt_config"](rate=.1, budget=4, warmup=0)
    state = f["opt_new_state"](p, config)
    union, union_state, _ = f["opt_step"](p, {"w": np.array([-1.])}, state, config, .1)
    sequential, sequential_state, _ = f["opt_step"](p, {"w": np.array([.5])}, state, config, .1)
    sequential, sequential_state, _ = f["opt_step"](sequential, {"w": np.array([-1.5])}, sequential_state, config, .1)
    assert union["w"][0] > 0 and sequential["w"][0] < 0
    assert union_state["t"] == 1 and sequential_state["t"] == 2
    clip = lambda vector: vector / max(1., math.hypot(*vector))
    np.testing.assert_allclose(clip(np.array([5., .5])), [10 / math.sqrt(101), 1 / math.sqrt(101)])
    np.testing.assert_allclose((clip(np.array([10., 0.])) + clip(np.array([0., 1.]))) / 2, [.5, .5])


def scalar_network_objective(parameters, X, y, l2):
    value = 0.
    for row, target in zip(X, y):
        hidden = []
        for j in range(len(parameters["b1"])):
            activation = sum(float(row[k]) * float(parameters["W1"][k, j]) for k in range(len(row)))
            hidden.append(math.tanh(activation + float(parameters["b1"][j])))
        prediction = sum(a * float(parameters["W2"][j, 0]) for j, a in enumerate(hidden)) + float(parameters["b2"][0])
        value += (prediction - float(target)) ** 2 / (2 * len(X))
    for key in ("W1", "W2"):
        for scalar in parameters[key].ravel():
            value += l2 * float(scalar) ** 2 / 2
    return value


@pytest.mark.parametrize("n", [1, 5])
@pytest.mark.parametrize("l2", [0., .03])
@pytest.mark.parametrize("hidden", [1, 3])
def test_manual_network_every_parameter_against_independent_scalar_difference(n, l2, hidden):
    f = functions(); X = np.linspace(-.6, .8, 2 * n).reshape(n, 2); y = np.linspace(.2, -.4, n)
    parameters = f["opt_initial"](2, hidden, 600)
    original = copy.deepcopy(parameters)
    loss, gradients, prediction = f["opt_mlp"](parameters, X, y, l2)
    assert loss == pytest.approx(scalar_network_objective(parameters, X, y, l2), abs=3e-15)
    assert f["opt_scalar_objective"](parameters, X, y, l2) == pytest.approx(loss, abs=3e-15)
    assert prediction.shape == (n,)
    for key in parameters:
        def objective(candidate):
            trial = copy.deepcopy(parameters)
            trial[key] = candidate.reshape(parameters[key].shape)
            return scalar_network_objective(trial, X, y, l2)
        expected = central_difference(objective, parameters[key].ravel())
        np.testing.assert_allclose(gradients[key].ravel(), expected, atol=2e-9, rtol=2e-6)
    assert gradients["b1"].shape == (hidden,) and gradients["b2"].shape == (1,)
    assert_nested_equal(parameters, original)
    report = f["opt_audit"](parameters, X, y, l2)
    assert report["loss_difference"] < 1e-12 and max(report["maximum_absolute_errors"].values()) < 1e-7


def test_network_row_replication_weight_only_penalty_and_visible_overflow():
    f = functions(); X = np.array([[-.4, .3], [.8, -.2]]); y = np.array([.7, -.4])
    p = f["opt_initial"](2, 3, 14)
    loss, gradient, _ = f["opt_mlp"](p, X, y, .04)
    repeated, repeated_gradient, _ = f["opt_mlp"](p, np.repeat(X, 3, axis=0), np.repeat(y, 3), .04)
    assert repeated == pytest.approx(loss)
    _, unpenalized, _ = f["opt_mlp"](p, X, y, 0.)
    for key in p:
        np.testing.assert_allclose(repeated_gradient[key], gradient[key], atol=2e-15)
        np.testing.assert_allclose(gradient[key] - unpenalized[key], .04 * p[key] if key.startswith("W") else 0., atol=2e-15)
    p = {"W1": np.array([[1e200]]), "b1": np.zeros(1), "W2": np.zeros((1, 1)), "b2": np.zeros(1)}
    assert f["opt_mlp"](p, [[0.]], [0.], 0.)[0] == 0.
    with pytest.raises(FloatingPointError):
        f["opt_mlp"](p, [[0.]], [0.], .1)
    p["W1"][0, 0] = 1e308
    with pytest.raises(FloatingPointError):
        f["opt_mlp"](p, [[2.]], [0.], 0.)


@pytest.mark.parametrize("X,y", [([], []), ([[1.]], [[1.]]), ([[np.inf]], [1.]), ([[1.]], [np.nan]), ([[1j]], [1.])])
def test_training_array_contracts(X, y):
    with pytest.raises(ValueError):
        functions()["opt_training_arrays"](X, y)


def fixture_run(kind="adam", budget=40):
    f = functions(); X = np.array([[-.7, .2], [.3, -.5], [.8, .4]]); y = np.array([.4, -.2, .7])
    config = f["opt_config"](kind=kind, rate=.03, budget=budget, warmup=min(4, budget - 1), batch=5,
                             decay=.1 if kind == "adamw" else 0., decay_keys=["W1", "W2"])
    run = f["opt_make_run"](X, y, config, initial_seed=14, batch_seed=21, hidden=3, l2=0. if kind == "adamw" else .02)
    return X, y, run


@pytest.mark.parametrize("kind", ["sgd", "momentum", "adam", "adamw"])
def test_real_json_checkpoint_exact_resume_and_input_preservation(kind, tmp_path):
    f = functions(); X, y, initial = fixture_run(kind)
    before = copy.deepcopy(initial)
    full = f["opt_continue"](initial, X, y, 40)
    part = f["opt_continue"](initial, X, y, 13)
    checkpoint = tmp_path / "state.json"
    checkpoint.write_bytes(f["opt_dump"](part, X, y))
    restored = f["opt_load"](checkpoint.read_bytes(), X, y)
    assert_nested_equal(restored, part)
    continued = f["opt_continue"](restored, X, y, 27)
    assert_nested_equal(continued, full)
    assert checkpoint.stat().st_size < 2 * 1024**2
    assert_nested_equal(initial, before)
    assert len(full["trace"]) == 40 and full["trace"][-1]["rate"] == 0.
    np.testing.assert_array_equal(part["trace"][-1]["t"], 13)
    assert [row["t"] for row in full["trace"] if row["full_objective"] is not None] == [20, 40]
    assert full["trace"][-1]["update_norm"] == 0.
    with pytest.raises(ValueError):
        f["opt_continue"](full, X, y, 1)


@pytest.mark.parametrize("rate,stable", [(1., True), (3.7, False)])
def test_momentum_quadratic_mode_matches_independent_second_order_recurrence(rate, stable):
    f = functions(); p = {"w": np.ones(1)}
    config = f["opt_config"](kind="momentum", rate=rate, momentum=.8, budget=200, warmup=0)
    state = f["opt_new_state"](p, config)
    actual = [1.]
    for _ in range(200):
        p, state, _ = f["opt_step"](p, {"w": p["w"].copy()}, state, config, rate)
        actual.append(float(p["w"][0]))
    expected = [1., 1 - rate]
    for _ in range(199):
        expected.append((1.8 - rate) * expected[-1] - .8 * expected[-2])
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-12)
    roots = np.roots([1., -(1.8 - rate), .8])
    assert (max(abs(roots)) < 1) == stable
    assert abs(actual[-1]) < 1e-8 if stable else abs(actual[-1]) > 1e5


def test_synthetic_data_and_training_only_scaler_against_recreated_streams():
    f = functions(); data = f["opt_data"]()
    raw = []
    for child, n in zip(np.random.SeedSequence(40500).spawn(3), [192, 96, 192]):
        rng = np.random.default_rng(child)
        a = rng.uniform(-2, 2, n); b = .9 * a + .3 * rng.normal(size=n)
        target = np.sin(1.7 * a) + .35 * b * b + .2 * a * b + .12 * rng.normal(size=n)
        raw.append((np.column_stack((a, b)), target))
    mean = raw[0][0].mean(axis=0); scale = np.sqrt(np.mean((raw[0][0] - mean)**2, axis=0))
    np.testing.assert_array_equal(data["mean"], mean)
    np.testing.assert_array_equal(data["scale"], scale)
    for key, (X, y) in zip(["train", "validation", "final"], raw):
        np.testing.assert_allclose(data[key][0], (X - mean) / scale, atol=2e-15)
        np.testing.assert_allclose(data[key][1], y, atol=2e-15)
    np.testing.assert_allclose(data["train"][0].mean(axis=0), 0., atol=2e-15)
    assert np.max(abs(data["validation"][0].mean(axis=0))) > .01


def test_comparison_complete_candidates_paired_streams_and_group_selection():
    f = functions(); data = f["opt_data"]()
    panel = f["opt_compare"](data, budget=12)
    assert len(panel["records"]) == 27 and len(panel["groups"]) == 9
    expected_order = [(kind, rate, seed) for kind, rates in [("sgd", [.01, .04, .12]), ("momentum", [.002, .008, .024]), ("adam", [.003, .01, .03])]
                      for rate in rates for seed in (40501, 40502, 40503)]
    assert [(r["kind"], r["rate"], r["seed"]) for r in panel["records"]] == expected_order
    assert all(r["status"] == "fixed_budget" and r["updates"] == 12 for r in panel["records"])
    assert sum(r["updates"] for r in panel["records"]) == 324
    for seed in panel["seeds"]:
        paired = [r for r in panel["records"] if r["seed"] == seed]
        assert len({r["run"]["initial_objective"] for r in paired}) == 1
        assert all(r["run"]["random"] == paired[0]["run"]["random"] for r in paired)
        assert all(r["batch_seed"] == seed + 100 and r["run"]["config"]["clip"] is None for r in paired)
    for group in panel["groups"]:
        expected = sum(panel["records"][i]["validation_half_mse"] for i in group["indices"]) / 3
        assert group["mean_validation_half_mse"] == pytest.approx(expected)
    for kind, selected in panel["selected"].items():
        candidates = [i for i, g in enumerate(panel["groups"]) if g["kind"] == kind]
        assert selected == min(candidates, key=lambda i: panel["groups"][i]["mean_validation_half_mse"])
        assert panel["groups"][candidates[0]]["rate_grid_boundary"] == "lower"
        assert panel["groups"][candidates[1]]["rate_grid_boundary"] == "interior"
        assert panel["groups"][candidates[2]]["rate_grid_boundary"] == "upper"
    assert panel["best_kind"] == min(panel["selected"], key=lambda kind: panel["groups"][panel["selected"][kind]]["mean_validation_half_mse"])


def test_validation_changes_selection_scores_but_never_training_and_final_is_unread():
    f = functions(); data = f["opt_data"](); changed = copy.deepcopy(data)
    changed["validation"] = (changed["validation"][0] * 10., changed["validation"][1] + 100.)
    changed["final"] = (object(), object())
    original = f["opt_compare"](data, budget=12)
    new = f["opt_compare"](changed, budget=12)
    assert any(a["validation_half_mse"] != b["validation_half_mse"] for a, b in zip(original["records"], new["records"]))
    for left, right in zip(original["records"], new["records"]):
        assert_nested_equal(left["run"], right["run"])
    only_final_changed = copy.deepcopy(data); only_final_changed["final"] = (object(), object())
    unchanged = f["opt_compare"](only_final_changed, budget=12)
    assert unchanged["selected"] == original["selected"]


def test_final_panel_caches_nine_predictions_and_consumes_guard_before_failure(monkeypatch):
    f = functions(); data = f["opt_data"](); comparison = f["opt_compare"](data, budget=12)
    original = f["opt_forward"]; calls = 0
    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)
    monkeypatch.setitem(f, "opt_forward", counted)
    result = f["opt_final_panel"](comparison, *data["final"])
    assert calls == 9 and len(result["rows"]) == 3
    for row in result["rows"]:
        expected = np.mean((row["predictions"] - data["final"][1][None, :]) ** 2, axis=1) / 2
        np.testing.assert_array_equal(row["half_mse"], expected)
        assert row["seeds"] == [40501, 40502, 40503]
        assert row["mean"] == pytest.approx(sum(expected) / 3)
        assert row["minimum"] == min(expected) and row["maximum"] == max(expected)
    expected_baseline = float(np.mean((data["train"][1].mean() - data["final"][1]) ** 2) / 2)
    assert result["baseline_half_mse"] == expected_baseline
    with pytest.raises(RuntimeError):
        f["opt_final_panel"](comparison, *data["final"])
    assert calls == 9
    fresh = copy.deepcopy(comparison); fresh["final_consumed"] = False
    def injected(*args, **kwargs):
        raise FloatingPointError("final inference failed")
    monkeypatch.setitem(f, "opt_forward", injected)
    with pytest.raises(FloatingPointError):
        f["opt_final_panel"](fresh, *data["final"])
    assert fresh["final_consumed"] is True
    with pytest.raises(RuntimeError):
        f["opt_final_panel"](fresh, *data["final"])


def test_failed_attempts_remain_visible_and_ineligible(monkeypatch):
    f = functions(); original = f["opt_continue"]
    def fail_one_setting(run, *args, **kwargs):
        if run["config"]["kind"] == "sgd" and run["config"]["rate"] == .01:
            raise FloatingPointError("injected nonfinite attempt")
        return original(run, *args, **kwargs)
    monkeypatch.setitem(f, "opt_continue", fail_one_setting)
    comparison = f["opt_compare"](f["opt_data"](), budget=4)
    assert len(comparison["records"]) == 27
    failures = [r for r in comparison["records"] if r["status"] == "nonfinite"]
    assert len(failures) == 3
    assert all(r["updates"] is None and r["run"] is None and "injected" in r["error"] for r in failures)
    assert comparison["groups"][0]["status"] == "failed" and comparison["groups"][0]["mean_validation_half_mse"] is None
    assert comparison["selected"]["sgd"] != 0


def test_batch_sampler_clock_and_batch_objective_at_old_snapshot():
    f = functions(); X, y, initial = fixture_run("momentum")
    run = f["opt_continue"](initial, X, y, 3)
    rng = np.random.default_rng(21)
    p, state = copy.deepcopy(initial["parameters"]), copy.deepcopy(initial["optimizer"])
    for t in (1, 2, 3):
        rows = rng.integers(0, len(X), size=5)
        expected_loss = scalar_network_objective(p, X[rows], y[rows], .02)
        _, gradient, _ = f["opt_mlp"](p, X[rows], y[rows], .02)
        rate = .03 * t / 4
        p, state = scalar_transition(p, gradient, state, initial["config"], rate)
        assert run["trace"][t - 1]["batch_objective"] == pytest.approx(expected_loss, abs=3e-15)
        assert run["trace"][t - 1]["rate"] == rate
    for key in p:
        np.testing.assert_allclose(run["parameters"][key], p[key], atol=3e-15)
    assert run["random"] == rng.bit_generator.state


def test_incomplete_restore_interventions_change_continuation():
    f = functions(); X, y, initial = fixture_run()
    partial = f["opt_continue"](initial, X, y, 13)
    complete = f["opt_continue"](partial, X, y, 27)
    bad_buffers, bad_rng = copy.deepcopy(partial), copy.deepcopy(partial)
    for name in ("first", "second"):
        bad_buffers["optimizer"][name] = {k: np.zeros_like(v) for k, v in partial["optimizer"][name].items()}
    bad_rng["random"] = np.random.default_rng(21).bit_generator.state
    for changed in (bad_buffers, bad_rng):
        result = f["opt_continue"](changed, X, y, 27)
        assert result["optimizer"]["t"] == complete["optimizer"]["t"] == 40
        assert max(np.max(np.abs(result["parameters"][k] - complete["parameters"][k])) for k in result["parameters"]) > 1e-6


def checkpoint_with_payload(payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return json.dumps(dict(format="mlfs-optimizer-json-1", payload=payload, sha256=hashlib.sha256(canonical).hexdigest())).encode()


@pytest.mark.parametrize("case", ["version", "trace_order", "trace_rate", "trace_length", "negative_second", "counter", "rng_state", "rng_flag", "config", "shape"])
def test_checkpoint_payload_validation_after_checksum(case):
    f = functions(); X, y, run = fixture_run()
    run = f["opt_continue"](run, X, y, 3)
    payload = json.loads(f["opt_dump"](run, X, y))["payload"]
    if case == "version":
        payload["version"] = True
    elif case == "trace_order":
        payload["trace"][0]["t"] = 1.0
    elif case == "trace_rate":
        payload["trace"][0]["rate"] = .02
    elif case == "trace_length":
        payload["trace"].pop()
    elif case == "negative_second":
        payload["optimizer"]["second"]["b1"][0] = -.1
    elif case == "counter":
        payload["optimizer"]["t"] = True
    elif case == "rng_state":
        payload["random"]["state"]["state"] = -1
    elif case == "rng_flag":
        payload["random"]["has_uint32"] = 2
    elif case == "config":
        payload["config"]["beta1"] = 1.
    elif case == "shape":
        payload["parameters"]["b1"] = [[0., 0., 0.]]
    with pytest.raises(ValueError):
        f["opt_load"](checkpoint_with_payload(payload), X, y)


def test_checkpoint_checksum_size_nonfinite_data_and_schema_failures():
    f = functions(); X, y, initial = fixture_run()
    wire = f["opt_dump"](initial, X, y)
    envelope = json.loads(wire); envelope["payload"]["l2"] = .3
    with pytest.raises(ValueError, match="checksum"):
        f["opt_load"](json.dumps(envelope).encode(), X, y)
    for bad in (b"", b"{}", b"NaN", b"[]", b" " * (2 * 1024**2 + 1)):
        with pytest.raises(ValueError):
            f["opt_load"](bad, X, y)
    for changed_x, changed_y in ((X[::-1], y[::-1]), (X + .1, y), (X, y + .1)):
        with pytest.raises(ValueError, match="changed"):
            f["opt_load"](wire, changed_x, changed_y)


def test_training_failure_does_not_partially_mutate_the_input_run(monkeypatch):
    f = functions(); X, y, initial = fixture_run()
    before = copy.deepcopy(initial); original = f["opt_step"]; calls = 0
    def fail_third(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise FloatingPointError("injected transition failure")
        return original(*args, **kwargs)
    monkeypatch.setitem(f, "opt_step", fail_third)
    with pytest.raises(FloatingPointError):
        f["opt_continue"](initial, X, y, 5)
    assert calls == 3
    assert_nested_equal(initial, before)
