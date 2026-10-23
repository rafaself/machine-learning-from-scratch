"""Independent small oracles for displayed clustering references, never answers."""

import copy
from fractions import Fraction
from itertools import combinations, product
import math

import numpy as np
import pytest

from test_numerics import central_difference, reference_functions


def functions():
    return reference_functions("clustering")


def scalar_squared_distances(X, centers):
    return np.array([
        [sum((float(x[j]) - float(center[j])) ** 2 for j in range(len(x)))
         for center in centers]
        for x in X
    ])


@pytest.mark.parametrize("block", [1, 2, 64])
def test_direct_distances_large_offsets_ties_and_block_invariance(block):
    X = np.array([[1e12 + 1, -1e12 + 2], [1e12 + 4, -1e12 - 1]])
    centers = np.array([[1e12, -1e12], [1e12 + 2, -1e12 + 4]])
    expected = scalar_squared_distances(X, centers)
    before = X.copy(), centers.copy()
    actual = functions()["cluster_squared_distances"](X, centers, block)
    np.testing.assert_array_equal(actual, expected)
    labels, nearest = functions()["assign_clusters"](X, centers, block)
    np.testing.assert_array_equal(labels, [0, 0])
    np.testing.assert_array_equal(nearest, expected[:, 0])
    np.testing.assert_array_equal(X, before[0])
    np.testing.assert_array_equal(centers, before[1])
    empty_labels, empty_distances = functions()["assign_clusters"](np.empty((0, 2)), centers, block)
    assert empty_labels.shape == empty_distances.shape == (0,)


@pytest.mark.parametrize("block", [0, -1, 1.5, True])
def test_distance_block_contract(block):
    with pytest.raises(ValueError):
        functions()["cluster_squared_distances"]([[1.]], [[0.]], block)


@pytest.mark.parametrize("X,centers", [
    ([], [[0.]]), ([[1.]], []), ([[1., 2.]], [[0.]]),
    ([[np.inf]], [[0.]]), ([[1.]], [[np.nan]]), (np.empty((2, 0)), [[0.]]),
])
def test_distance_shapes_and_finiteness(X, centers):
    with pytest.raises(ValueError):
        functions()["cluster_squared_distances"](X, centers)


def test_distance_overflow_and_distinct_value_underflow_are_visible():
    with pytest.raises(FloatingPointError):
        functions()["cluster_squared_distances"]([[1e308]], [[-1e308]])
    with pytest.raises(FloatingPointError, match="zero representable"):
        functions()["seed_kmeans"]([[0.], [1e-300]], 2)


@pytest.mark.parametrize("method", ["random", "plus_plus"])
@pytest.mark.parametrize("seed", [0, 250, 913])
def test_seeding_matches_independent_record_weighted_rng_replay(method, seed):
    X = np.array([[0., 0.], [0., 0.], [2., 1.], [10., -1.], [13., 2.]])
    rng = np.random.default_rng(seed)
    if method == "random":
        expected = rng.choice(len(X), 3, replace=False).tolist()
    else:
        expected = [int(rng.integers(len(X)))]
        for _ in range(2):
            weights = scalar_squared_distances(X, X[expected]).min(axis=1)
            weights /= max(weights)
            expected.append(int(rng.choice(len(X), p=weights / sum(weights))))
    centers, indices = functions()["seed_kmeans"](X, 3, method, seed)
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(centers, X[expected])
    centers[0] = 999
    assert not np.any(X == 999)


@pytest.mark.parametrize("k", [0, 4, 1.5, True])
def test_seed_and_fit_reject_invalid_distinct_center_counts(k):
    X = [[0.], [0.], [1.], [2.]]
    for name in ("seed_kmeans", "fit_kmeans"):
        with pytest.raises(ValueError):
            functions()[name](X, k)


@pytest.mark.parametrize("rows", [1, 4])
def test_centroid_gradient_and_hessian_against_independent_scalar_sse(rows):
    X = np.linspace(-1.1, 2.3, rows * 3).reshape(rows, 3)
    center = np.array([.4, -.3, .9])

    def objective(candidate):
        return sum((float(x[j]) - float(candidate[j])) ** 2
                   for x in X for j in range(3))

    loss, gradient, hessian = functions()["centroid_sse_derivatives"](X, center)
    assert loss == pytest.approx(objective(center))
    np.testing.assert_allclose(gradient, central_difference(objective, center), rtol=1e-8)
    # Differentiate independent scalar finite differences, not the reference gradient.
    numeric_hessian = np.column_stack([
        (central_difference(objective, center + np.eye(3)[j] * 1e-3)
         - central_difference(objective, center - np.eye(3)[j] * 1e-3)) / 2e-3
        for j in range(3)
    ])
    np.testing.assert_allclose(hessian, numeric_hessian, atol=2e-6)
    mean = np.array([sum(float(x[j]) for x in X) / rows for j in range(3)])
    centered = sum((float(x[j]) - mean[j]) ** 2 for x in X for j in range(3))
    ordered_pairs = sum(sum((float(a[j]) - float(b[j])) ** 2 for j in range(3))
                        for a in X for b in X)
    assert centered == pytest.approx(ordered_pairs / (2 * rows), abs=1e-12)
    assert objective(center) == pytest.approx(centered + rows * sum((center - mean) ** 2))


def test_lloyd_hand_states_short_budget_and_output_consistency():
    fit = functions()["fit_kmeans"]
    model = fit([[0.], [2.], [8.], [10.]], 2, init=[[0.], [8.]])
    np.testing.assert_array_equal(model["inertia_history"], [8., 4.])
    np.testing.assert_array_equal(model["centers"], [[1.], [9.]])
    np.testing.assert_array_equal(model["labels"], [0, 0, 1, 1])
    assert model["converged"] and model["iterations"] == 1
    X = np.array([[0.], [2.], [3.], [10.]])
    limited = fit(X, 2, init=[[0.], [2.]], max_steps=1)
    assert not limited["converged"] and limited["iterations"] == 1
    squared = scalar_squared_distances(X, limited["centers"])
    np.testing.assert_array_equal(limited["labels"], squared.argmin(axis=1))
    assert limited["inertia"] == pytest.approx(sum(squared.min(axis=1)))
    assert np.all(np.diff(limited["inertia_history"]) <= 1e-12)


@pytest.mark.parametrize("policy,occupied,cost", [("keep", 1, 2.), ("farthest", 2, 0.)])
def test_empty_cluster_policy_is_explicit(policy, occupied, cost):
    model = functions()["fit_kmeans"]([[-1.], [1.]], 2, init=[[0.], [0.]], empty=policy)
    assert model["converged"] and model["effective_clusters"] == occupied
    assert model["inertia"] == pytest.approx(cost)
    assert model["empty_events"][0] == dict(iteration=1, clusters=[1], policy=policy,
                                          rows=[0] if policy == "farthest" else [])


def test_restarts_retain_all_seeds_statuses_and_first_minimum():
    X = np.linspace(-2, 2, 30).reshape(-1, 1)
    result, records, selected = functions()["kmeans_restarts"](X, 3, starts=6, seed=22, max_steps=1)
    seeds = [int(child.generate_state(1)[0]) for child in np.random.SeedSequence(22).spawn(6)]
    expected = [functions()["fit_kmeans"](X, 3, seed=seed, max_steps=1) for seed in seeds]
    assert len(records) == 6 and [record["seed"] for record in records] == seeds
    assert any(not record["converged"] for record in records)
    for record, model in zip(records, expected):
        assert record["inertia"] == model["inertia"]
        assert record["converged"] == model["converged"]
        assert record["effective_clusters"] == model["effective_clusters"]
    assert selected == min(range(6), key=lambda i: expected[i]["inertia"])
    np.testing.assert_array_equal(result["centers"], expected[selected]["centers"])
    # Identical k=1 costs must select the first run, not a last-wins replacement.
    assert functions()["kmeans_restarts"](X, 1, starts=4)[2] == 0


@pytest.mark.parametrize("kwargs", [
    {"starts": 0}, {"starts": 33}, {"starts": True}, {"max_steps": 0},
    {"max_steps": 501}, {"max_steps": True}, {"method": "unknown"}, {"empty": "unknown"},
])
def test_fit_search_work_budgets_and_policies(kwargs):
    with pytest.raises(ValueError):
        functions()["kmeans_restarts"]([[0.], [1.]], 2, **kwargs)


@pytest.mark.parametrize("method,last,first", [
    ("single", 6., 2.), ("complete", 10., 2.), ("average", 8., 2.), ("ward", 64., 2.),
])
def test_hand_hierarchies_have_declared_heights_and_tie_order(method, last, first):
    model = functions()["agglomerative_linkage"]([[0.], [2.], [8.], [10.]], method)
    np.testing.assert_array_equal(model["linkage"][:, :2], [[0, 1], [2, 3], [4, 5]])
    np.testing.assert_allclose(model["linkage"][:, 2], [first, first, last])
    np.testing.assert_array_equal(model["linkage"][:, 3], [2, 2, 4])
    np.testing.assert_array_equal(functions()["cut_linkage"](model, 2), [0, 0, 1, 1])


def member_sse(X, indices):
    mean = [sum(float(X[i, j]) for i in indices) / len(indices) for j in range(X.shape[1])]
    return sum((float(X[i, j]) - mean[j]) ** 2 for i in indices for j in range(X.shape[1]))


def independent_merge_cost(X, A, B, method):
    if method == "ward":
        return member_sse(X, A + B) - member_sse(X, A) - member_sse(X, B)
    distances = [math.sqrt(sum((float(X[a, j]) - float(X[b, j])) ** 2
                               for j in range(X.shape[1]))) for a in A for b in B]
    return min(distances) if method == "single" else (
        max(distances) if method == "complete" else sum(distances) / len(distances)
    )


@pytest.mark.parametrize("method", ["single", "complete", "average", "ward"])
def test_every_merge_against_independent_member_level_objective_and_nested_cuts(method):
    X = np.array([[0., 0.], [0., 0.], [1., 2.], [3., -1.], [8., 1.], [11., 2.]])
    model = functions()["agglomerative_linkage"](X, method)
    active = {i: [i] for i in range(len(X))}
    for t, (a, b, height, count) in enumerate(model["linkage"]):
        candidates = [independent_merge_cost(X, active[i], active[j], method)
                      for i, j in combinations(active, 2)]
        A, B = active.pop(int(a)), active.pop(int(b))
        assert height == pytest.approx(independent_merge_cost(X, A, B, method), abs=1e-11)
        assert height == pytest.approx(min(candidates), abs=1e-11)
        assert count == len(A + B)
        active[len(X) + t] = A + B
    if method == "ward":
        assert sum(model["linkage"][:, 2]) == pytest.approx(member_sse(X, list(range(len(X)))))
    for k in range(1, len(X) + 1):
        labels = functions()["cut_linkage"](model, k)
        assert len(np.unique(labels)) == k
        if k > 1:
            coarser = functions()["cut_linkage"](model, k - 1)
            assert all(len(np.unique(coarser[labels == c])) == 1 for c in np.unique(labels))


def test_average_cross_pairs_and_ward_unequal_size_weights():
    X = [[0.], [1.], [2.], [9.], [20.]]
    average = functions()["agglomerative_linkage"](X, "average")
    ward = functions()["agglomerative_linkage"](X, "ward")
    np.testing.assert_allclose(average["linkage"][:, 2], [1., 1.5, 8., 17.])
    np.testing.assert_allclose(ward["linkage"][:, 2], [.5, 1.5, 48., 231.2])
    assert sum(ward["linkage"][:, 2]) == pytest.approx(281.2)


@pytest.mark.parametrize("column,value", [(0, .5), (0, 999), (1, 0), (3, 3)])
def test_cut_rejects_invalid_consumed_merge_prefix(column, value):
    model = functions()["agglomerative_linkage"]([[0.], [2.], [8.], [10.]])
    broken = copy.deepcopy(model)
    broken["linkage"][0, column] = value
    with pytest.raises(ValueError):
        functions()["cut_linkage"](broken, 2)


def test_linkage_and_silhouette_enforce_small_dense_limits():
    with pytest.raises(ValueError):
        functions()["agglomerative_linkage"](np.arange(81).reshape(-1, 1))
    with pytest.raises(ValueError):
        functions()["silhouette_samples"](np.arange(401).reshape(-1, 1), np.arange(401) % 2)


def test_silhouette_hand_singletons_and_zero_distances():
    score = functions()["silhouette_samples"]
    values = score([[0.], [2.], [8.], [10.]], [0, 0, 1, 1])
    np.testing.assert_allclose(values, [7 / 9, 5 / 7, 5 / 7, 7 / 9])
    assert values.mean() == pytest.approx(47 / 63)
    np.testing.assert_array_equal(values, score([[0.], [2.], [8.], [10.]], [8, 8, -3, -3]))
    assert score([[0.], [2.], [8.]], [0, 0, 1])[2] == 0.
    np.testing.assert_array_equal(score(np.zeros((4, 1)), [0, 0, 1, 1]), np.zeros(4))


@pytest.mark.parametrize("partition", [[0, 0, 0], [0, 1, 2], [0., 0., 1.], [0, 1]])
def test_silhouette_partition_contract(partition):
    with pytest.raises(ValueError):
        functions()["silhouette_samples"]([[0.], [2.], [8.]], partition)


def pair_count_ari(left, right):
    pairs = list(combinations(range(len(left)), 2))
    A = sum(left[i] == left[j] for i, j in pairs)
    B = sum(right[i] == right[j] for i, j in pairs)
    same = sum(left[i] == left[j] and right[i] == right[j] for i, j in pairs)
    if not pairs:
        return Fraction(1)
    expected = Fraction(A * B, len(pairs))
    denominator = Fraction(A + B, 2) - expected
    return (Fraction(same) - expected) / denominator if denominator else Fraction(1)


def test_ari_against_exact_pair_oracle_all_small_binary_partitions():
    partitions = list(product((0, 1), repeat=4))
    for left, right in product(partitions, repeat=2):
        actual = functions()["adjusted_rand_index"](left, right)
        assert actual == pytest.approx(float(pair_count_ari(left, right)), abs=1e-15)
    assert functions()["adjusted_rand_index"]([0, 0, 1, 1], [0, 1, 0, 1]) == -.5
    for left, right in [([0], [99]), ([0, 0, 0], [9, 9, 9]), ([0, 1, 2], [9, 8, 7])]:
        assert functions()["adjusted_rand_index"](left, right) == 1.


def test_exact_partition_oracle_and_seed_expectation_with_rational_row_weights():
    data = [0, 0, 2, 10]
    costs = []
    for tail in product((0, 1), repeat=3):
        labels = (0,) + tail
        if 1 not in labels:
            continue
        cost = Fraction(0)
        for c in (0, 1):
            members = [x for x, label in zip(data, labels) if label == c]
            mean = Fraction(sum(members), len(members))
            cost += sum((Fraction(x) - mean) ** 2 for x in members)
        costs.append(cost)
    assert min(costs) == Fraction(8, 3)
    expected = Fraction(0)
    for first in range(4):
        weights = [(x - data[first]) ** 2 for x in data]
        for second in range(4):
            probability = Fraction(weights[second], 4 * sum(weights))
            potential = sum(min((x - data[first]) ** 2, (x - data[second]) ** 2) for x in data)
            expected += probability * potential
    assert expected == Fraction(10234, 1287)
    X = np.array(data).reshape(-1, 1)
    optimum = functions()["tiny_two_means_optimum"](X)
    result = functions()["exact_two_seed_potential"](X)
    assert optimum["inertia"] == pytest.approx(float(min(costs)))
    assert result["expected_potential"] == pytest.approx(float(expected))
    assert result["total_probability"] == pytest.approx(1.)
    expectation_bound = 8 * (math.log(2) + 2) * float(min(costs))
    assert float(expected) < expectation_bound < 64.
    repaired = functions()["fit_kmeans"](X, 2, init=[[0.], [2.]])
    assert repaired["inertia"] == pytest.approx(8 / 3)


@pytest.mark.parametrize("name", ["tiny_two_means_optimum", "exact_two_seed_potential"])
def test_exponential_oracles_enforce_small_input_budget(name):
    with pytest.raises(ValueError):
        functions()[name](np.arange(11).reshape(-1, 1))
    with pytest.raises(ValueError):
        functions()[name]([[0.]])


def test_blob_experiment_declares_reproducible_disjoint_draws():
    train, probe = functions()["clustering_blob_data"]()
    assert train.shape == (180, 2) and probe.shape == (120, 2)
    again = functions()["clustering_blob_data"]()
    np.testing.assert_array_equal(train, again[0])
    np.testing.assert_array_equal(probe, again[1])
    assert not np.any(np.all(train[:, None] == probe[None, :], axis=2))
