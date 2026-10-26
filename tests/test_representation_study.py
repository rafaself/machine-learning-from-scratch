"""Independent mathematical and protocol checks for the Stage 03 capstone."""

import copy
from fractions import Fraction
import json
import math

import numpy as np
import pytest

from test_numerics import central_difference, reference_functions


def functions():
    return reference_functions("representation-study")


def small_study():
    f = functions()
    x = np.random.default_rng(9124).normal(size=(64, 4)) * [1., 3., .4, 2.]
    x[-1] = x[0]
    data = dict(name="tiny", X=x, groups=f["study_groups"](x), audit_labels=np.arange(64) % 3)
    parts = f["study_partition"](data["groups"])
    bundle = f["study_select"](x[parts[0]], x[parts[1]])
    return data, parts, bundle


def rewrite_checkpoint(path, mutate):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(str(arrays["metadata"]))
    mutate(arrays, metadata)
    arrays["metadata"] = np.array(json.dumps(metadata, allow_nan=False))
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def scalar_common_loss(row, reconstructed, scale):
    return sum(((float(a) - float(b)) / float(c)) ** 2 for a, b, c in zip(row, reconstructed, scale)) / len(row)


def scalar_ari(a, b):
    n = len(a)
    if n < 2:
        return 1.
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    left = sum(a[i] == a[j] for i, j in pairs)
    right = sum(b[i] == b[j] for i, j in pairs)
    both = sum(a[i] == a[j] and b[i] == b[j] for i, j in pairs)
    expected = Fraction(left * right, len(pairs))
    denominator = Fraction(left + right, 2) - expected
    return 1. if denominator == 0 else float((both - expected) / denominator)


@pytest.mark.parametrize("name,n,groups", [("mixture", 360, 360), ("gaussian_null", 360, 360), ("iris", 150, 147)])
def test_tasks_and_label_free_complete_group_partitions(name, n, groups):
    f = functions(); first = f["study_datasets"]()[name]; second = f["study_datasets"]()[name]
    assert first["X"].shape == (n, 4)
    assert len(np.unique(first["groups"])) == groups
    for key in ("X", "groups"):
        np.testing.assert_array_equal(first[key], second[key])
    if name == "gaussian_null":
        assert first["audit_labels"] is None
    else:
        np.testing.assert_array_equal(first["audit_labels"], second["audit_labels"])
    parts = f["study_partition"](first["groups"])
    np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(n))
    for i, part in enumerate(parts):
        for other in parts[i + 1:]:
            assert not np.intersect1d(first["groups"][part], first["groups"][other]).size
    before = tuple(part.copy() for part in parts)
    first["audit_labels"] = np.random.default_rng(12).integers(0, 7, size=n)
    for a, b in zip(before, f["study_partition"](first["groups"])):
        np.testing.assert_array_equal(a, b)
    if name != "iris":
        assert [len(part) for part in parts] == [180, 72, 54, 54]
        np.testing.assert_array_equal(first["groups"], np.arange(n))


def test_first_occurrence_groups_avoid_feature_rank_dependent_fixed_splits():
    f = functions()
    np.testing.assert_array_equal(f["study_groups"]([[9.], [2.], [3.], [9.]]), [0, 1, 2, 0])
    rng = np.random.default_rng(982)
    x = rng.normal(size=(32, 4))
    first = f["study_partition"](f["study_groups"](x))
    changed = x[:, ::-1] * [-1., 10., .1, 3.]
    for a, b in zip(first, f["study_partition"](f["study_groups"](changed))):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("groups", [[], [0] * 12, [0., 1.] * 8, np.arange(7), np.arange(2001), [[0, 1]]])
def test_group_partition_invalid_workloads(groups):
    with pytest.raises(ValueError):
        functions()["study_partition"](groups)


def test_training_coordinate_variance_divisor_constant_and_held_out_isolation():
    f = functions(); x = np.array([[-2., 5.], [2., 5.]])
    mean, scale = f["study_fit_coordinates"](x)
    np.testing.assert_array_equal(mean, [0., 5.]); np.testing.assert_array_equal(scale, [2., 1.])
    model = f["study_fit_representation"](x, dict(name="check", mode="standardized", components=None))
    before = copy.deepcopy(model)
    np.testing.assert_array_equal(f["study_coordinates"](model, [[10., 7.]]), [[5., 2.]])
    for key in ("mean", "scale", "reference_scale", "axes"):
        np.testing.assert_array_equal(model[key], before[key])
    x[:] = -100
    np.testing.assert_array_equal(model["mean"], [0., 5.])
    with pytest.raises(FloatingPointError):
        f["study_fit_coordinates"]([[0.], [1e-300]])
    with pytest.raises(FloatingPointError):
        f["study_fit_coordinates"]([[-1e308], [1e308]])


@pytest.mark.parametrize("mode", ["raw", "standardized"])
@pytest.mark.parametrize("retained", [1, 2, None])
def test_pca_projector_spectrum_and_direct_common_reconstruction(mode, retained):
    f = functions()
    x = np.array([[-2., -1.], [-2., 1.], [2., -1.], [2., 1.]])
    model = f["study_fit_representation"](x, dict(name="audit", mode=mode, components=retained))
    scale = [1., 1.] if mode == "raw" else [2., 1.]
    covariance = np.diag([4 / scale[0] ** 2, 1 / scale[1] ** 2])
    values, vectors = np.linalg.eigh(covariance)
    np.testing.assert_allclose(model["spectrum"], values[::-1], atol=1e-14)
    np.testing.assert_allclose(model["axes"].T @ model["axes"], np.eye(model["axes"].shape[1]), atol=1e-14)
    if retained == 1 and mode == "raw":
        np.testing.assert_allclose(model["axes"] @ model["axes"].T, [[1., 0.], [0., 0.]])
        assert model["retained_ratio"] == pytest.approx(.8)
    else:
        # Tied standardized one-component eigenvectors are nonunique.
        assert model["retained_ratio"] == pytest.approx(.5 if retained == 1 else 1.)
    r = model["axes"].shape[1]
    model["centers"] = np.zeros((1, r))
    output = f["study_predict"](model, x)
    expected = [scalar_common_loss(row, reconstruction, [2., 1.]) for row, reconstruction in zip(x, output["reconstruction"])]
    np.testing.assert_allclose(output["losses"], expected)
    assert np.mean(expected) == pytest.approx(1.)


def test_metric_cross_term_and_hard_vs_marginal_mixture_hand_case():
    residual = np.array([1., 1.]); quantization = np.array([1., -1.]); metric = np.array([2., 1.])
    assert residual @ quantization == 0
    assert (residual * metric) @ (quantization * metric) == 3
    assert np.sum(((residual + quantization) * metric) ** 2) == 16
    assert np.sum((residual * metric) ** 2) + np.sum((quantization * metric) ** 2) == 10
    marginal_density = .5 * math.exp(-.5) / math.sqrt(2 * math.pi) * 2
    marginal_nll = -math.log(marginal_density)
    hard_nll = -math.log(marginal_density / 2)
    assert marginal_nll == pytest.approx(1.4189385332046727)
    assert hard_nll == pytest.approx(2.112085713764618)
    assert hard_nll - marginal_nll == pytest.approx(math.log(2))


@pytest.mark.parametrize("n", [1, 7])
def test_all_center_gradient_coordinates_against_independent_sum_objective(n):
    f = functions(); z = np.linspace(-1., 1., 2 * n).reshape(n, 2)
    centers = np.array([[.2, -.3], [.7, -.6], [1.2, -.5]])
    labels = np.arange(n) % 2  # Third center is deliberately empty.

    def objective(candidate):
        c = candidate.reshape(3, 2)
        return sum(sum((float(z[i, j]) - float(c[labels[i], j])) ** 2 for j in range(2)) for i in range(n))

    loss, gradient = f["study_centroid_objective"](z, centers, labels)
    assert loss == pytest.approx(objective(centers.ravel()))
    np.testing.assert_allclose(gradient.ravel(), central_difference(objective, centers.ravel()), rtol=2e-8, atol=2e-8)
    repeated = f["study_centroid_objective"](np.repeat(z, 3, axis=0), centers, np.repeat(labels, 3))
    assert repeated[0] == pytest.approx(3 * loss)
    np.testing.assert_allclose(repeated[1], 3 * gradient)
    np.testing.assert_array_equal(gradient[2], [0., 0.])


def test_lloyd_hand_optimum_against_exhaustive_two_group_objective_and_budget_status():
    f = functions(); x = np.array([[0.], [2.], [8.], [10.]])
    run = f["study_kmeans"](x, k=2, initial_indices=[0, 2])
    np.testing.assert_allclose(run["centers"], [[1.], [9.]])
    assert run["trace"][0] == 8 and run["training_sse"] == 4
    assert run["status"] == "stable_labels" and run["gradient_norm"] == 0
    values = []
    for mask in range(1, 2 ** 3):
        labels = [0] + [(mask >> j) & 1 for j in range(3)]
        if len(set(labels)) < 2:
            continue
        centers = [sum(float(x[i, 0]) for i in range(4) if labels[i] == k) / labels.count(k) for k in (0, 1)]
        values.append(sum((float(x[i, 0]) - centers[labels[i]]) ** 2 for i in range(4)))
    assert min(values) == 4
    assert 4 / (4 * 17) == pytest.approx(1 / 17)
    unfinished = f["study_kmeans"](x, k=2, initial_indices=[0, 2], max_steps=0)
    assert unfinished["status"] == "budget_exhausted" and unfinished["iterations"] == 0
    assert unfinished["training_sse"] == 8 and unfinished["gradient_norm"] > 0


@pytest.mark.parametrize("initialization", ["random", "d_squared"])
@pytest.mark.parametrize("seed", [0, 13, 29])
def test_restart_trace_centers_and_diagnostics_from_independent_assignment(initialization, seed):
    f = functions(); x = np.random.default_rng(82).normal(size=(24, 3))
    run = f["study_kmeans"](x, initialization=initialization, seed=seed)
    assert len(set(run["initial_indices"])) == 3
    assert len(run["trace"]) == run["iterations"] + 1
    assert np.all(np.diff(run["trace"]) <= 2e-12)
    center = run["centers"]
    labels = np.array([min(range(3), key=lambda k: sum((float(row[j]) - float(center[k, j])) ** 2 for j in range(3))) for row in x])
    objective = sum(sum((float(row[j]) - float(center[label, j])) ** 2 for j in range(3)) for row, label in zip(x, labels))
    assert run["training_sse"] == pytest.approx(objective, abs=1e-12)
    assert run["empty_centers"] == sum(not np.any(labels == k) for k in range(3))
    if run["status"] == "stable_labels":
        for k in range(3):
            if np.any(labels == k):
                np.testing.assert_allclose(center[k], x[labels == k].mean(axis=0), atol=1e-14)


def test_zero_potential_ties_and_empty_keep_policy_are_visible():
    f = functions(); x = np.zeros((8, 2))
    for method in ("random", "d_squared"):
        run = f["study_kmeans"](x, initialization=method)
        assert len(set(run["initial_indices"])) == 3 and run["empty_centers"] == 2
        assert run["status"] == "stable_labels" and run["training_sse"] == 0
    distances = f["study_squared_distances"]([[0., 0.]], [[1., 0.], [-1., 0.]])
    assert int(distances.argmin(axis=1)[0]) == 0


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"k": 9}, {"k": True}, {"max_steps": -1}, {"max_steps": 201}, {"max_steps": True}, {"initialization": "rescue"}, {"initial_indices": [0, 0, 2]}])
def test_centroid_budget_and_initialization_contracts(kwargs):
    with pytest.raises(ValueError):
        functions()["study_kmeans"](np.zeros((12, 2)), **kwargs)


def test_distance_scalar_large_offset_empty_queries_and_arithmetic_failure():
    f = functions(); query = np.array([[1e12 + 1., 1e12 + 3.]])
    centers = np.array([[1e12, 1e12], [1e12 + 2., 1e12 + 1.]])
    for block in (1, 64, 256):
        np.testing.assert_array_equal(f["study_squared_distances"](query, centers, block), [[10., 5.]])
    assert f["study_squared_distances"](np.empty((0, 2)), centers).shape == (0, 2)
    with pytest.raises(FloatingPointError):
        f["study_squared_distances"]([[-1e308]], [[1e308]])
    with pytest.raises(ValueError):
        f["study_squared_distances"](query, centers, 257)


def test_selection_complete_start_budgets_scalar_common_losses_and_first_tie():
    f = functions(); data, parts, bundle = small_study()
    assert len(bundle["models"]) == len(bundle["records"]) == 7
    assert bundle["records"][0]["starts"] == 0
    for i, model in enumerate(bundle["models"]):
        output = f["study_predict"](model, data["X"][parts[1]])
        expected = [scalar_common_loss(row, rec, model["reference_scale"]) for row, rec in zip(data["X"][parts[1]], output["reconstruction"])]
        np.testing.assert_allclose(bundle["validation_losses"][i], expected, atol=2e-14)
        assert bundle["records"][i]["validation_mean"] == pytest.approx(sum(expected) / len(expected))
        np.testing.assert_array_equal(model["reference_scale"], bundle["models"][0]["reference_scale"])
        assert model["fit_n"] == len(parts[0])
        if i:
            assert len(model["runs"]) == 8
            assert [r["initialization"] for r in model["runs"]] == ["random"] * 4 + ["d_squared"] * 4
            assert model["chosen_run"] == int(np.argmin([r["training_sse"] for r in model["runs"]]))
    assert bundle["chosen"] == int(np.argmin(bundle["validation_losses"].mean(axis=1)))
    constant = f["study_select"](np.ones((12, 4)), np.ones((5, 4)), max_steps=0)
    assert constant["chosen"] == 0
    assert all(r["status"] == "budget_exhausted" for m in constant["models"][1:] for r in m["runs"])


def test_validation_changes_do_not_change_fitted_arrays_or_training_start_choices():
    f = functions(); data, parts, first = small_study()
    second = f["study_select"](data["X"][parts[0]], data["X"][parts[1]] + 100)
    for a, b in zip(first["models"], second["models"]):
        for field in f["study_model_array_fields"]():
            np.testing.assert_array_equal(a[field], b[field])
        assert a["runs"] == b["runs"] and a["chosen_run"] == b["chosen_run"]


@pytest.mark.parametrize("a,b", [([0, 0, 1, 1], [8, 8, 3, 3]), ([0, 0, 1, 1], [0, 1, 0, 1]), ([0, 0, 0], [0, 1, 2]), ([0, 0], [7, 7]), ([0, 1], [3, 4]), ([0], [9])])
def test_ari_against_independent_pair_counts(a, b):
    assert functions()["study_ari"](a, b) == pytest.approx(scalar_ari(a, b))


def test_initialization_stability_retains_all_twenty_eight_pairs_and_no_fake_baseline():
    f = functions(); data, parts, bundle = small_study()
    baseline = f["study_restart_stability"](bundle["models"][0], data["X"][parts[1]])
    assert baseline == dict(pairs=[], mean=None, minimum=None, maximum=None)
    model = bundle["models"][1]
    output = f["study_restart_stability"](model, data["X"][parts[1]])
    assert len(output["pairs"]) == 28
    assignments = [f["study_predict"](dict(model, centers=np.array(run["centers"])), data["X"][parts[1]])["clusters"] for run in model["runs"]]
    expected = [scalar_ari(assignments[i], assignments[j]) for i in range(8) for j in range(i + 1, 8)]
    np.testing.assert_allclose([r["ari"] for r in output["pairs"]], expected)
    assert output["mean"] == pytest.approx(sum(expected) / 28)


@pytest.mark.parametrize("scores", [[0., 1., 2., 3., 4.], [0., 0., 1., 1., 2.], [1.] * 5])
@pytest.mark.parametrize("alpha", ["0.1", "0.2", "0.35", "0.9"])
def test_exact_exchangeable_rank_enumeration_and_conservative_ties(scores, alpha):
    rejected = 0; level = Fraction(alpha)
    for i, query in enumerate(scores):
        calibration = scores[:i] + scores[i + 1:]
        numerator = 1 + sum(x >= query for x in calibration)
        result = functions()["study_rank_decisions"](calibration, [query], alpha)
        assert result["numerators"][0] == numerator
        assert result["flags"][0] == (Fraction(numerator, len(scores)) <= level)
        assert result["flags"][0] == (query > result["threshold"])
        rejected += int(result["flags"][0])
    assert Fraction(rejected, len(scores)) <= level


def test_rank_resolution_and_decimal_boundary_without_quantile_interpolation():
    f = functions()
    hand = f["study_rank_decisions"](np.arange(1., 10.), [7., 7.1, 9., 10.], "0.35")
    assert hand["threshold"] == 7.
    np.testing.assert_array_equal(hand["flags"], [False, True, True, True])
    below = f["study_rank_decisions"]([0., 1.], [1000.], "0.1")
    assert math.isinf(below["threshold"]) and not below["flags"][0]


@pytest.mark.parametrize("alpha", [True, 0, 1, np.nan, np.inf, "1/0"])
def test_invalid_rank_levels(alpha):
    with pytest.raises(ValueError):
        functions()["study_rank_decisions"]([0., 1.], [2.], alpha)


def test_beta_flag_posterior_against_rational_moments_and_prior_prediction():
    f = functions()
    posterior = f["study_beta_flags"](3, 10)
    assert (posterior["a"], posterior["b"]) == (4., 8.)
    assert posterior["posterior_mean"] == pytest.approx(float(Fraction(1, 3)))
    assert posterior["posterior_variance"] == pytest.approx(float(Fraction(2, 117)))
    assert posterior["one_future_flag_probability"] == posterior["posterior_mean"]
    prior = f["study_beta_flags"](0, 0)
    assert prior["posterior_mean"] == .5 and prior["posterior_variance"] == pytest.approx(1 / 12)


@pytest.mark.parametrize("flagged,n", [(-1, 3), (4, 3), (1, 2001), (True, 3), (1.5, 3)])
def test_beta_count_contract(flagged, n):
    with pytest.raises(ValueError):
        functions()["study_beta_flags"](flagged, n)


def test_paired_resampling_uses_complete_groups_and_row_weighted_denominator():
    f = functions(); differences = np.array([-2., 1., 3., 4., 5., -1.]); groups = np.array([0, 0, 1, 2, 2, 2])
    result = f["study_paired_group_sensitivity"](differences, groups, repeats=80, seed=11)
    rng = np.random.default_rng(11); sums = {0: -1., 1: 3., 2: 8.}; counts = {0: 2, 1: 1, 2: 3}
    expected = []
    for _ in range(80):
        sampled = rng.choice([0, 1, 2], 3, replace=True)
        expected.append(sum(sums[g] for g in sampled) / sum(counts[g] for g in sampled))
    np.testing.assert_allclose(result["draws"], expected, atol=2e-15)
    assert result["observed_mean"] == pytest.approx(10 / 6)
    np.testing.assert_allclose([result["lower"], result["upper"]], np.quantile(expected, [.025, .975]))


def test_frozen_checkpoint_round_trip_no_pickle_no_refitting_and_no_overwrite(tmp_path, monkeypatch):
    f = functions(); data, parts, bundle = small_study(); path = tmp_path / "study.npz"
    f["study_save_checkpoint"](path, bundle, data, parts)
    with np.load(path, allow_pickle=False) as archive:
        assert all(not archive[key].dtype.hasobject for key in archive.files)
    monkeypatch.setitem(f, "study_fit_model", lambda *a, **kw: pytest.fail("resumption refitted a model"))
    resumed, saved_parts, metadata, result = f["study_resume_checkpoint"](path, data)
    assert metadata["phase"] == "frozen" and result is None
    for a, b in zip(parts, saved_parts):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(bundle["models"], resumed["models"]):
        assert a["runs"] == b["runs"]
        for key in f["study_model_array_fields"]():
            np.testing.assert_array_equal(a[key], b[key])
    with pytest.raises(FileExistsError):
        f["study_save_checkpoint"](path, bundle, data, parts)


@pytest.mark.parametrize("damage", ["coverage", "group_overlap", "selection", "mean_shape", "axes_nan", "centers_disagree", "phase_evidence", "loss_shape"])
def test_checkpoint_rejects_inconsistent_internal_evidence(tmp_path, damage):
    f = functions(); data, parts, bundle = small_study(); path = tmp_path / "study.npz"
    f["study_save_checkpoint"](path, bundle, data, parts)

    def mutate(arrays, metadata):
        if damage == "coverage":
            arrays["final_indices"][0] = arrays["train_indices"][0]
        elif damage == "group_overlap":
            arrays["groups"][arrays["final_indices"][0]] = arrays["groups"][arrays["train_indices"][0]]
        elif damage == "selection":
            metadata["chosen"] = (metadata["chosen"] + 1) % 7
        elif damage == "mean_shape":
            arrays["model_1_mean"] = arrays["model_1_mean"][:, None]
        elif damage == "axes_nan":
            arrays["model_1_axes"][0, 0] = np.nan
        elif damage == "centers_disagree":
            arrays["model_1_centers"][0, 0] += 100
        elif damage == "phase_evidence":
            metadata["phase"] = "complete"
        else:
            arrays["validation_losses"] = arrays["validation_losses"][:, :-1]

    rewrite_checkpoint(path, mutate)
    with pytest.raises(ValueError):
        f["study_load_checkpoint"](path)


def test_checkpoint_identity_rejects_changed_feature_and_audit_label_bytes(tmp_path):
    f = functions(); data, parts, bundle = small_study(); path = tmp_path / "study.npz"
    f["study_save_checkpoint"](path, bundle, data, parts)
    for key in ("X", "audit_labels"):
        changed = copy.deepcopy(data); changed[key].flat[1] += 1
        with pytest.raises(ValueError, match="identity"):
            f["study_resume_checkpoint"](path, changed)


def test_final_claim_persists_across_injected_failure_and_second_call_is_refused(tmp_path):
    f = functions(); data, parts, bundle = small_study(); path = tmp_path / "study.npz"
    f["study_save_checkpoint"](path, bundle, data, parts)

    def interrupt(*args):
        assert f["study_load_checkpoint"](path)[2]["phase"] == "claimed"
        raise RuntimeError("injected inference interruption")

    with pytest.raises(RuntimeError, match="injected"):
        f["study_finalize_checkpoint"](path, data, evaluate=interrupt)
    resumed, _, metadata, result = f["study_resume_checkpoint"](path, data)
    assert resumed["phase"] == metadata["phase"] == "claimed" and result is None
    with pytest.raises(RuntimeError, match="already claimed"):
        f["study_finalize_checkpoint"](path, data)


def test_final_panel_independent_losses_ranks_external_audit_and_saved_complete_phase(tmp_path, monkeypatch):
    f = functions(); data, parts, bundle = small_study(); path = tmp_path / "study.npz"
    f["study_save_checkpoint"](path, bundle, data, parts)
    original_predict = f["study_predict"]; calls = []

    def count(model, rows):
        calls.append(len(rows))
        return original_predict(model, rows)

    monkeypatch.setitem(f, "study_predict", count)
    result = f["study_finalize_checkpoint"](path, data)
    assert calls == [len(parts[3])] * 7 + [len(parts[2])] * 2
    resumed, saved_parts, metadata, saved = f["study_resume_checkpoint"](path, data)
    assert resumed["phase"] == metadata["phase"] == "complete"
    for key in result["arrays"]:
        np.testing.assert_array_equal(result["arrays"][key], saved["arrays"][key])
    array = saved["arrays"]
    for i, model in enumerate(resumed["models"]):
        expected = [scalar_common_loss(row, rec, model["reference_scale"]) for row, rec in zip(data["X"][parts[3]], array["reconstructions"][i])]
        np.testing.assert_allclose(array["losses"][i], expected, atol=2e-14)
        assert saved["metadata"]["external_ari"][i] == pytest.approx(scalar_ari(data["audit_labels"][parts[3]], array["clusters"][i]))
    for row, index in enumerate([bundle["chosen"], 0]):
        calibration = array["calibration_scores"][row]
        expected_numerators = [1 + sum(float(c) >= float(q) for c in calibration) for q in array["losses"][index]]
        expected_flags = [Fraction(numerator, len(calibration) + 1) <= Fraction(1, 10) for numerator in expected_numerators]
        np.testing.assert_array_equal(array["rank_numerators"][row], expected_numerators)
        np.testing.assert_array_equal(array["flags"][row], expected_flags)
    before = len(calls)
    with pytest.raises(RuntimeError):
        f["study_finalize_checkpoint"](path, data)
    assert len(calls) == before
    rewrite_checkpoint(path, lambda arrays, meta: arrays.update(result_losses=arrays["result_losses"][:, :-1]))
    with pytest.raises(ValueError, match="final array"):
        f["study_load_checkpoint"](path)


def test_atomic_checkpoint_write_failure_preserves_previous_artifact(tmp_path, monkeypatch):
    f = functions(); data, parts, bundle = small_study(); path = tmp_path / "study.npz"
    f["study_save_checkpoint"](path, bundle, data, parts)
    before = path.read_bytes()

    def broken(*args, **kwargs):
        raise OSError("injected write failure")

    monkeypatch.setattr(np, "savez_compressed", broken)
    with pytest.raises(OSError, match="injected"):
        f["study_save_checkpoint"](path, bundle, data, parts, allow_update=True)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_units_and_bootstrap_sensitivity_keep_original_metric_selection_and_all_trials():
    f = functions(); data, parts, bundle = small_study(); before = copy.deepcopy(bundle)
    train, validation = data["X"][parts[0]], data["X"][parts[1]]
    result = f["study_sensitivity"](train, validation, bundle, repeats=2)
    assert len(result["unit_records"]) == 7 and len(result["bootstrap_records"]) == 2
    assert result["original_selected"] == bundle["chosen"] == before["chosen"]
    for row in result["unit_records"]:
        if row["name"].startswith("standardized") or row["name"] == "mean":
            assert row["ari"] == pytest.approx(1.) and row["max_loss_difference"] < 2e-10
    reference = bundle["models"][bundle["chosen"]]
    for record in result["bootstrap_records"]:
        assert record["indices"].shape == (len(train),)
        fitted = f["study_fit_model"](train[record["indices"]], reference["spec"], bundle["seed"], bundle["max_steps"])
        reconstruction = f["study_predict"](fitted, validation)["reconstruction"]
        expected = [scalar_common_loss(row, rec, reference["reference_scale"]) for row, rec in zip(validation, reconstruction)]
        assert record["validation_mean"] == pytest.approx(sum(expected) / len(expected))
    for a, b in zip(bundle["models"], before["models"]):
        for key in f["study_model_array_fields"]():
            np.testing.assert_array_equal(a[key], b[key])
