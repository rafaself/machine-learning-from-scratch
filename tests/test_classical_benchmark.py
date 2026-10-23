"""Independent mathematical/procedural checks for the Stage 02 project.

Only displayed author definitions are loaded. Unfinished learner answers and
the full benchmark/final-test experiment are not executed by these tests.
"""

import json
import math

import numpy as np
import pytest

from test_numerics import central_difference, reference_functions


def functions():
    return reference_functions("classical-benchmark")


def test_benchmark_tasks_are_deterministic_and_match_declared_artifacts():
    f = functions()
    a, b = f["benchmark_datasets"](), f["benchmark_datasets"]()
    for name, rows, task in (("regression",180,"regression"),("radial",160,"classification"),("iris",150,"classification")):
        assert a[name]["X"].shape == (rows,4) and a[name]["y"].shape == (rows,)
        assert a[name]["task"] == task
        for key in ("X","y","groups"):
            np.testing.assert_array_equal(a[name][key], b[name][key])
    np.testing.assert_array_equal(np.bincount(a["radial"]["y"]), [80,80])
    assert len(np.unique(a["iris"]["groups"])) == 147
    np.testing.assert_allclose(a["iris"]["X"][34], [4.9,3.1,1.5,.1])


@pytest.mark.parametrize("name", ["regression","radial","iris"])
def test_outer_and_cv_partitions_cover_rows_without_any_group_overlap(name):
    f = functions(); data = f["benchmark_datasets"]()[name]
    parts = f["benchmark_partition"](data["y"], data["groups"], data["task"])
    repeated = f["benchmark_partition"](data["y"], data["groups"], data["task"])
    np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(len(data["y"])))
    for i, a in enumerate(parts):
        np.testing.assert_array_equal(a, repeated[i])
        for b in parts[i+1:]:
            assert not np.intersect1d(a,b).size
            assert not np.intersect1d(data["groups"][a],data["groups"][b]).size
        if data["task"] == "classification":
            np.testing.assert_array_equal(np.unique(data["y"][a]),np.unique(data["y"]))
    if name != "iris":
        assert [len(part) for part in parts] == ([108,36,36] if name == "regression" else [96,32,32])
    train = parts[0]
    groups, y = data["groups"][train],data["y"][train]
    folds = f["benchmark_folds"](y,groups,data["task"])
    np.testing.assert_array_equal(np.sort(np.concatenate([held for _,held in folds])),np.arange(len(train)))
    for fitted,held in folds:
        np.testing.assert_array_equal(np.sort(np.r_[fitted,held]),np.arange(len(train)))
        assert not np.intersect1d(groups[fitted],groups[held]).size
        if data["task"] == "classification":
            np.testing.assert_array_equal(np.unique(y[fitted]),np.unique(y))
            np.testing.assert_array_equal(np.unique(y[held]),np.unique(y))


def test_group_scheme_rejects_mixed_labels_insufficient_groups_and_invalid_fold_count():
    f = functions()
    with pytest.raises(ValueError):
        f["benchmark_group_strata"]([0,1],[3,3],"classification")
    with pytest.raises(ValueError):
        f["benchmark_group_strata"]([0,1],[3.,4.],"classification")
    with pytest.raises(ValueError):
        f["benchmark_partition"]([0,0,1,1],[0,1,2,3],"classification")
    with pytest.raises(ValueError):
        f["benchmark_folds"]([0,0,1,1],[0,1,2,3],"classification",folds=3)
    with pytest.raises(ValueError):
        f["benchmark_folds"](np.arange(9.),np.arange(9),"regression",folds=True)


def test_scaler_uses_only_fit_rows_and_preserves_constant_columns():
    f = functions()
    X = np.array([[1.,0.],[3.,0.]])
    scaler = f["benchmark_scale_fit"](X)
    np.testing.assert_array_equal(scaler["mean"],[2.,0.])
    np.testing.assert_array_equal(scaler["scale"],[1.,1.])
    np.testing.assert_array_equal(f["benchmark_scale_apply"](X,scaler),[[-1.,0.],[1.,0.]])
    held = f["benchmark_scale_apply"]([[1e9,9.]],scaler)
    np.testing.assert_array_equal(held,[[1e9-2,9.]])
    np.testing.assert_array_equal(scaler["mean"],[2.,0.])
    np.testing.assert_array_equal(X,[[1.,0.],[3.,0.]])
    assert f["benchmark_scale_apply"](np.empty((0,2)),scaler).shape == (0,2)
    with pytest.raises(ValueError):
        f["benchmark_scale_apply"]([[1.]],scaler)


def test_ridge_closed_form_matches_independent_scalar_covariance_and_replication():
    fit = functions()["benchmark_ridge_fit"]
    X,y = np.array([[0.],[1.],[2.]]),np.array([1.,2.,2.])
    model = fit(X,y,.1)
    assert model["weights"][0] == pytest.approx(10/23)
    assert model["bias"] == pytest.approx(85/69)
    assert model["gradient_norm"] < 1e-12
    duplicate = fit(np.repeat(X,3,axis=0),np.repeat(y,3),.1)
    np.testing.assert_allclose(duplicate["weights"],model["weights"])
    assert duplicate["bias"] == pytest.approx(model["bias"])
    independent = lambda theta: sum((float(row[0])*theta[0]+theta[1]-float(target))**2 for row,target in zip(X,y))/6 + .1*theta[0]**2/2
    np.testing.assert_allclose(central_difference(independent,np.r_[model["weights"],model["bias"]]),[0.,0.],atol=1e-9)


@pytest.mark.parametrize("n", [1,5])
@pytest.mark.parametrize("penalty", [0.,.13])
def test_capstone_softmax_all_parameter_gradients_against_independent_objective(n, penalty):
    evaluate = functions()["benchmark_softmax_objective"]
    X = np.linspace(-.8,.9,n*2).reshape(n,2); y = np.arange(n)%3
    theta = np.linspace(-.3,.4,9)
    def independent(candidate):
        W,b = candidate[:6].reshape(2,3),candidate[6:]
        total = 0.
        for row,label in zip(X,y):
            logits = [sum(float(row[j])*float(W[j,k]) for j in range(2))+float(b[k]) for k in range(3)]
            total += math.log(sum(math.exp(s) for s in logits))-logits[label]
        return total/n+penalty*sum(float(w)**2 for w in W.ravel())/2
    loss,dW,db = evaluate(X,y,theta[:6].reshape(2,3),theta[6:],penalty)
    assert loss == pytest.approx(independent(theta))
    np.testing.assert_allclose(np.r_[dW.ravel(),db],central_difference(independent,theta),rtol=1e-5,atol=1e-9)
    repeated = evaluate(np.repeat(X,2,axis=0),np.repeat(y,2),theta[:6].reshape(2,3),theta[6:],penalty)
    assert repeated[0] == pytest.approx(loss)
    np.testing.assert_allclose(repeated[1],dW); np.testing.assert_allclose(repeated[2],db)


def test_softmax_fixed_budget_trace_and_stable_log_probabilities():
    f = functions()
    X = np.linspace(-2,2,12)[:,None]; y = np.repeat([0,1,2],4)
    fit = f["benchmark_softmax_fit"](X,y,.1,steps=40)
    assert fit["iterations"] == 40 and fit["loss_history"].shape == (41,)
    assert np.all(np.diff(fit["loss_history"]) <= 1e-12)
    value,dW,db = f["benchmark_softmax_objective"](X,y,fit["weights"],fit["bias"],.1)
    assert value == pytest.approx(fit["loss_history"][-1])
    assert fit["gradient_norm"] == pytest.approx(np.sqrt(np.sum(dW*dW)+np.sum(db*db)))
    np.testing.assert_allclose(f["benchmark_log_probabilities"]([[1000.,-1000.]]),[[0.,-2000.]])


def test_capstone_numeric_range_failures_are_explicit():
    f = functions()
    with pytest.raises(FloatingPointError):
        f["benchmark_scale_fit"]([[1e308],[-1e308]])
    with pytest.raises(FloatingPointError):
        f["benchmark_ridge_fit"]([[1e308],[-1e308]],[0.,1.],.1)
    with pytest.raises(FloatingPointError):
        f["benchmark_log_probabilities"]([[np.finfo(float).max,-np.finfo(float).max]])


@pytest.mark.parametrize("family", ["baseline","ridge","neighbor","tree","forest","boost"])
def test_regression_adapters_keep_primary_metric_separate_from_training_loss(family):
    f = functions(); X = np.column_stack([np.linspace(-2,2,18),np.sin(np.arange(18))]); y = X[:,0]+.4*X[:,1]
    spec = dict(family=family)
    spec.update(penalty=.1,k=3,depth=2,features=2,rate=.1)
    model = f["benchmark_fit_model"](X,y,"regression",spec)
    output = f["benchmark_predict_model"](model,X[:4])
    assert output["prediction"].shape == (4,) and np.isfinite(output["prediction"]).all()
    assert output["probability"] is None and output["log_probability"] is None
    metrics = f["benchmark_metrics"](y[:4],output,"regression")
    expected = sum((float(a)-float(b))**2 for a,b in zip(y[:4],output["prediction"]))/4
    assert metrics["primary"] == pytest.approx(expected)
    assert metrics["rmse"] == pytest.approx(math.sqrt(expected))


@pytest.mark.parametrize("family", ["baseline","linear","neighbor","gaussian","tree","forest","boost","svm"])
def test_classification_adapters_preserve_labels_and_probability_semantics(family):
    f = functions(); X = np.column_stack([np.linspace(-2,2,18),np.sin(np.arange(18))]); y = np.repeat([-7,9,100],6)
    spec = dict(family=family,penalty=.1,k=3,kind="shared" if family == "gaussian" else "linear",depth=2,features=2,rate=.1,gamma=.5)
    model = f["benchmark_fit_model"](X,y,"classification",spec)
    output = f["benchmark_predict_model"](model,X[:4])
    assert np.isin(output["prediction"],[-7,9,100]).all()
    np.testing.assert_array_equal(model["classes"],[-7,9,100])
    metrics = f["benchmark_metrics"](y[:4],output,"classification",model["classes"])
    assert metrics["primary"] == pytest.approx(np.mean(output["prediction"] != y[:4]))
    if family in ("svm","boost"):
        assert output["probability"] is None and math.isnan(metrics["log_loss"]) and math.isnan(metrics["brier"])
    else:
        assert output["probability"].shape == (4,3)
        np.testing.assert_allclose(output["probability"].sum(axis=1),1.)
        np.testing.assert_allclose(np.exp(output["log_probability"]),output["probability"])
        assert np.isfinite(metrics["log_loss"]) and np.isfinite(metrics["brier"])
    if family == "svm":
        assert model["diagnostics"]["C"] == pytest.approx(1/(18*.1))
        assert len(model["diagnostics"]["gaps"]) == 3


def test_frequency_mixture_and_metric_hand_oracles_without_silent_clipping():
    f = functions(); X = np.arange(18.)[:,None]; y = np.repeat([0,1,2],6)
    model = f["benchmark_fit_model"](X,y,"classification",dict(family="tree",depth=2))
    output = f["benchmark_predict_model"](model,[[0.],[8.],[17.]])
    expected = np.full((3,3),1/60); np.fill_diagonal(expected,29/30)
    np.testing.assert_allclose(output["probability"],expected)
    raw = dict(prediction=np.array([0,0]),probability=np.array([[1.,0.],[1.,0.]]),log_probability=np.array([[0.,-np.inf],[0.,-np.inf]]))
    metrics = f["benchmark_metrics"]([0,1],raw,"classification",np.array([0,1]))
    assert metrics["primary"] == .5 and metrics["macro_recall"] == .5
    assert math.isinf(metrics["log_loss"]) and metrics["brier"] == 1.
    np.testing.assert_array_equal(metrics["confusion"],[[1,0],[1,0]])
    stable = dict(prediction=np.array([1,1]),probability=np.array([[0.,1.],[0.,1.]]),log_probability=np.array([[-1000.,0.],[-1000.,0.]]))
    assert f["benchmark_metrics"]([0,1],stable,"classification",np.array([0,1]))["log_loss"] == 500.


def audited_selection():
    f = functions(); X = np.column_stack([np.arange(10.),np.zeros(10)]); y = np.arange(10.)
    groups = np.array([0,0,1,2,3,4,5,6,7,8]); calls = []
    def fit(A,b,task,spec,seed):
        calls.append((A.copy(),b.copy(),dict(spec),seed))
        return dict(value=float(b.mean()),classes=None,diagnostics={})
    def predict(model,A):
        return dict(prediction=np.full(len(A),model["value"]),probability=None,log_probability=None)
    bundle = f["benchmark_select"](X,y,groups,[[1e9,0.],[2e9,0.]],[777.,888.],"regression",
                                    specs=[dict(family="audit",candidate=0),dict(family="audit",candidate=1)],fit_fn=fit,predict_fn=predict)
    return f,X,y,groups,bundle,calls,predict


def test_selection_callbacks_prove_fold_local_scaling_row_isolation_and_pooled_ties():
    f,X,y,groups,bundle,calls,_ = audited_selection()
    assert len(calls) == 8 and len(bundle["fold_records"]) == 6
    assert bundle["selected_indices"] == {"audit":0} and bundle["champion"] == "baseline"
    expected_sum,total = 0.,0
    for index,(training,held) in enumerate(bundle["folds"]):
        A,b,_,seed = calls[index]
        np.testing.assert_array_equal(b,y[training])
        mean = np.array([sum(X[training,j])/len(training) for j in range(2)])
        scale = np.array([math.sqrt(sum((float(v)-mean[j])**2 for v in X[training,j])/len(training)) for j in range(2)])
        scale[scale == 0] = 1.
        np.testing.assert_allclose(A,(X[training]-mean)/scale)
        assert seed == 244+index and not np.isin(b,[777.,888.]).any()
        assert not np.intersect1d(groups[training],groups[held]).size
        expected_sum += sum((float(target)-float(y[training].mean()))**2 for target in y[held]); total += len(held)
    np.testing.assert_allclose(bundle["pooled_scores"],[expected_sum/total]*2)
    for A,b,_,_ in calls[-2:]:
        np.testing.assert_array_equal(b,y)
        np.testing.assert_allclose(A[:,0],(X[:,0]-4.5)/np.std(X[:,0],ddof=0))
    np.testing.assert_array_equal(bundle["models"]["audit"]["scaler"]["mean"],[4.5,0.])
    assert sorted(len(held) for _,held in bundle["folds"]) == [3,3,4]


def test_checkpoint_round_trip_retains_folds_choices_and_safe_numeric_payloads(tmp_path):
    f,X,y,groups,bundle,_,_ = audited_selection()
    full_groups = np.r_[groups,[100,101,200,201]]
    parts = (np.arange(10),np.array([10,11]),np.array([12,13]))
    data = dict(task='regression',X=np.column_stack([np.arange(14.),np.zeros(14)]),
                y=np.r_[y,[777.,888.,1.,2.]],groups=full_groups)
    path = tmp_path/"selection.npz"
    f["benchmark_save_checkpoint"](path,bundle,parts,data,"oracle")
    metadata,arrays = f["benchmark_load_checkpoint"](path)
    assert metadata["champion"] == "baseline" and not metadata["test_used"]
    assert metadata["selected"] == {"audit":0}
    np.testing.assert_array_equal(arrays["train_rows"],parts[0])
    np.testing.assert_allclose(arrays["pooled_scores"],bundle["pooled_scores"])
    assert all(not value.dtype.hasobject for value in arrays.values())
    for candidate in (0,1):
        records = [r for r in metadata["fold_records"] if r["candidate"] == candidate]
        assert sum(r["loss_sum"] for r in records)/sum(r["held_rows"] for r in records) == pytest.approx(arrays["pooled_scores"][candidate])
    bad = dict(arrays); bad["fold_assignment"] = bad["fold_assignment"].copy()
    bad["fold_assignment"][1] = (bad["fold_assignment"][0]+1)%3
    np.savez_compressed(tmp_path/"bad.npz",metadata=np.array(json.dumps(metadata)),**bad)
    with pytest.raises(ValueError):
        f["benchmark_load_checkpoint"](tmp_path/"bad.npz")


def test_checkpoint_resume_refits_only_frozen_training_choices_and_respects_identity_and_used_flag(tmp_path):
    f,_,y,groups,bundle,_,_ = audited_selection()
    data = dict(task='regression',X=np.column_stack([np.arange(14.),np.zeros(14)]),
                y=np.r_[y,[777.,888.,1.,2.]],groups=np.r_[groups,[100,101,200,201]])
    parts = (np.arange(10),np.array([10,11]),np.array([12,13])); path = tmp_path/'resume.npz'
    f['benchmark_save_checkpoint'](path,bundle,parts,data,'oracle')
    fitted = []
    def fit(A,b,task,spec,seed):
        fitted.append((A.copy(),b.copy(),dict(spec),seed))
        return dict(value=float(b.mean()),classes=None,diagnostics={})
    resumed = f['benchmark_resume_checkpoint'](path,data,fit_fn=fit)
    assert len(fitted) == 2 and resumed['champion'] == bundle['champion']
    assert resumed['selected_indices'] == {'audit':0} and not resumed['test_used']
    for A,b,spec,seed in fitted:
        np.testing.assert_array_equal(b,np.arange(10.))
        np.testing.assert_allclose(A[:,0],(np.arange(10.)-4.5)/np.std(np.arange(10.)))
        assert seed == 244
        assert spec.get('candidate',0) == 0
    changed = {key:value.copy() if isinstance(value,np.ndarray) else value for key,value in data.items()}
    changed['X'][0,0] += 1
    with pytest.raises(ValueError):
        f['benchmark_resume_checkpoint'](path,changed,fit_fn=fit)
    bundle['test_used'] = True
    f['benchmark_save_checkpoint'](path,bundle,parts,data,'oracle')
    with pytest.raises(ValueError):
        f['benchmark_resume_checkpoint'](path,data,fit_fn=fit)


def test_final_evaluation_uses_frozen_scalers_without_fits_and_rejects_second_call():
    f,_,_,_,bundle,calls,predict = audited_selection()
    fit_count = len(calls)
    result = f["benchmark_evaluate_once"](bundle,[[100.,0.],[200.,0.]],[1.,2.],predict_fn=predict)
    assert bundle["test_used"] and len(calls) == fit_count
    assert list(result) == ["baseline","audit"]
    for record in result.values():
        np.testing.assert_array_equal(record["prediction"],[4.5,4.5])
        np.testing.assert_allclose(record["losses"],[3.5**2,2.5**2])
    with pytest.raises(ValueError):
        f["benchmark_evaluate_once"](bundle,[[100.,0.],[200.,0.]],[1.,2.],predict_fn=predict)
    bundle["test_used"] = False; bundle["frozen"] = False
    with pytest.raises(ValueError):
        f["benchmark_evaluate_once"](bundle,[[100.,0.]],[1.],predict_fn=predict)


@pytest.mark.parametrize('differences', [[-3.,1.,-1.,-1.],[-3.,1.,2.,-1.]])
def test_paired_group_resampling_matches_independent_complete_row_reconstruction(differences):
    interval = functions()["benchmark_paired_group_interval"]
    differences = np.array(differences); groups = np.array([0,0,1,2])
    mean,quantiles,actual = interval(differences,groups,repeats=60,seed=245)
    rng = np.random.default_rng(245); expected = []
    for _ in range(60):
        chosen = rng.integers(0,3,size=3)
        rows = [i for group in chosen for i,value in enumerate(groups) if value == group]
        expected.append(sum(float(differences[i]) for i in rows)/len(rows))
    assert mean == sum(float(v) for v in differences)/4
    np.testing.assert_allclose(actual,expected)
    np.testing.assert_allclose(quantiles,np.quantile(expected,[.025,.975]))
    assert sum([-2.,-2.,-1.])/sum([2,2,1]) == -1.
    assert interval(np.zeros(4),groups,repeats=10)[1].tolist() == [0.,0.]


@pytest.mark.parametrize("repeats", [0,10001,True,1.5])
def test_group_resampling_rejects_invalid_work_budgets(repeats):
    with pytest.raises(ValueError):
        functions()["benchmark_paired_group_interval"]([0.,1.],[0,1],repeats=repeats)


def test_capstone_hand_fold_tie_and_declared_grid_counts():
    f = functions()
    A,B = np.array([.5,0.]),np.array([0.,1/3])
    sizes = np.array([2,3])
    assert np.average(A,weights=sizes) == pytest.approx(.2)
    assert np.average(B,weights=sizes) == pytest.approx(.2)
    assert A.mean() > B.mean()
    for task,expected in (("regression",10),("classification",14)):
        specs = f["benchmark_specs"](task)
        assert len(specs) == expected
        assert all(sum(s["family"] == name for s in specs) == 2 for name in {s["family"] for s in specs})
