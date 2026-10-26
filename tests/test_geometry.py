"""Independent scalar curvature, finite-population and diagnostic experiments."""

import copy
import math

import numpy as np
import pytest

from test_numerics import central_difference, reference_functions


def functions():
    return reference_functions("optimization-geometry-and-diagnostics")


def scalar_definition(theta, X, y, widths, l2=0., activation="tanh"):
    """No displayed unpack, forward, gradient or tangent implementation."""
    layers, cursor, squared_weights = [], 0, 0.
    for a, b in zip(widths[:-1], widths[1:]):
        W = []
        for i in range(a):
            row = []
            for j in range(b):
                value = float(theta[cursor]); cursor += 1
                row.append(value); squared_weights += value * value
            W.append(row)
        bias = [float(v) for v in theta[cursor:cursor+b]]; cursor += b
        layers.append((W, bias))
    predictions = []
    for row in X:
        current = [float(v) for v in row]
        for layer, (W, bias) in enumerate(layers):
            result = []
            for j in range(len(bias)):
                value = bias[j] + sum(current[i] * W[i][j] for i in range(len(current)))
                if layer < len(layers)-1 and activation == "tanh":
                    value = math.tanh(value)
                elif layer < len(layers)-1 and activation == "relu":
                    value = max(value, 0.)
                result.append(value)
            current = result
        predictions.append(current[0])
    loss = sum((p-float(t))**2 for p,t in zip(predictions,y))/(2*len(y)) + l2*squared_weights/2
    return loss, np.array(predictions)


def mask_for(widths):
    return np.concatenate([np.r_[np.ones(a*b),np.zeros(b)] for a,b in zip(widths[:-1],widths[1:])])


@pytest.mark.parametrize("widths", [(1,1),(1,2,1),(2,3,1),(2,2,2,1)])
@pytest.mark.parametrize("n", [1,4])
@pytest.mark.parametrize("l2", [0.,.03])
@pytest.mark.parametrize("activation", ["tanh","linear"])
def test_manual_all_parameter_input_tangent_and_hessian_products_against_scalar_objective(widths,n,l2,activation):
    f=functions(); size=sum((a+1)*b for a,b in zip(widths[:-1],widths[1:]))
    theta=np.linspace(-.3,.4,size); X=np.linspace(-.7,.8,n*widths[0]).reshape(n,widths[0])
    y=np.linspace(.2,-.4,n); direction=np.random.default_rng(84).normal(size=size)
    direction/=math.hypot(*direction); original=(theta.copy(),X.copy(),y.copy(),direction.copy())
    result=f["geo_network"](theta,X,y,widths,activation,l2,direction)
    objective=lambda value:scalar_definition(value,X,y,widths,l2,activation)[0]
    assert result["loss"]==pytest.approx(objective(theta),abs=2e-15)
    np.testing.assert_allclose(result["prediction"],scalar_definition(theta,X,y,widths,l2,activation)[1],atol=2e-15)
    np.testing.assert_allclose(result["gradient"],central_difference(objective,theta),atol=2e-9,rtol=2e-6)
    h=1e-4; product=np.empty(size)
    for j in range(size):
        basis=np.zeros(size);basis[j]=h
        product[j]=(objective(theta+basis+h*direction)-objective(theta+basis-h*direction)
                    -objective(theta-basis+h*direction)+objective(theta-basis-h*direction))/(4*h*h)
    np.testing.assert_allclose(result["hvp"],product,atol=8e-8,rtol=2e-5)
    input_objective=lambda value:scalar_definition(theta,value.reshape(X.shape),y,widths,l2,activation)[0]
    np.testing.assert_allclose(result["input_gradient"].ravel(),central_difference(input_objective,X.ravel()),atol=2e-9,rtol=2e-6)
    expected_tangent=(scalar_definition(theta+1e-5*direction,X,y,widths,l2,activation)[1]
                      -scalar_definition(theta-1e-5*direction,X,y,widths,l2,activation)[1])/(2e-5)
    np.testing.assert_allclose(result["prediction_tangent"],expected_tangent,atol=2e-9,rtol=2e-6)
    for actual,before in zip((theta,X,y,direction),original):
        np.testing.assert_array_equal(actual,before)


def test_row_replication_penalty_mask_and_hvp_linearity():
    f=functions(); widths=(2,3,1);theta=np.linspace(-.3,.4,13)
    X=np.array([[-.7,.2],[.3,-.5],[.8,.4]]);y=np.array([.4,-.2,.7])
    v=np.linspace(.1,.5,13);u=np.linspace(-.4,.2,13)
    base=f["geo_network"](theta,X,y,widths,l2=0.,direction=v)
    penalized=f["geo_network"](theta,X,y,widths,l2=.07,direction=v)
    repeated=f["geo_network"](theta,np.repeat(X,3,axis=0),np.repeat(y,3),widths,l2=.07,direction=v)
    mask=mask_for(widths)
    assert penalized["loss"]-base["loss"]==pytest.approx(.07*sum(theta[mask==1]**2)/2)
    np.testing.assert_allclose(penalized["gradient"]-base["gradient"],.07*mask*theta,atol=2e-15)
    np.testing.assert_allclose(penalized["hvp"]-base["hvp"],.07*mask*v,atol=2e-15)
    for key in ("gradient","hvp"):
        np.testing.assert_allclose(repeated[key],penalized[key],atol=2e-15)
    hu=f["geo_network"](theta,X,y,widths,l2=.07,direction=u)["hvp"]
    combined=f["geo_network"](theta,X,y,widths,l2=.07,direction=2*v-.5*u)["hvp"]
    np.testing.assert_allclose(combined,2*penalized["hvp"]-.5*hu,atol=3e-15)
    assert float(u@penalized["hvp"])==pytest.approx(float(v@hu),abs=3e-15)


@pytest.mark.parametrize("l2",[0.,.2])
@pytest.mark.parametrize("activation",["tanh","linear"])
def test_exact_stationary_saddle_hand_matrix_and_negative_direction(l2,activation):
    f=functions();widths=(1,1,1);theta=np.zeros(4);X=np.array([[-1.],[1.]]);y=np.array([-1.,1.])
    result=f["geo_network"](theta,X,y,widths,activation,l2)
    expected=np.array([[l2,0.,-1.,0.],[0.,0.,0.,0.],[-1.,0.,l2,0.],[0.,0.,0.,1.]])
    actual=np.column_stack([f["geo_network"](theta,X,y,widths,activation,l2,v)["hvp"] for v in np.eye(4)])
    np.testing.assert_array_equal(result["gradient"],np.zeros(4))
    assert result["loss"]==.5
    np.testing.assert_allclose(actual,expected,atol=1e-15)
    d=np.array([1.,0.,1.,0.])/math.sqrt(2)
    assert float(d@actual@d)==pytest.approx(l2-1.)
    assert f["geo_network"](theta+.1*d,X,y,widths,activation,l2)["loss"]<.5
    if activation=="tanh":
        spectrum=f["geo_curvature"](theta,X,y,widths,l2)
        np.testing.assert_allclose(spectrum["gauss_newton"],np.diag([l2,0.,l2,1.]),atol=1e-15)
        np.testing.assert_allclose(spectrum["eigenvalues"],[l2-1,0.,1.,l2+1],atol=1e-15)


def test_dense_curvature_zero_residual_jacobian_and_single_linear_layer():
    f=functions();widths=(2,3,1);theta=np.linspace(-.3,.4,13)
    X=np.array([[-.7,.2],[.3,-.5],[.8,.4]]);y=np.array([.4,-.2,.7])
    result=f["geo_curvature"](theta,X,y,widths,l2=.03)
    expected_jac=np.array([central_difference(lambda v:scalar_definition(v,X[i:i+1],[0.],widths)[1][0],theta) for i in range(3)])
    np.testing.assert_allclose(result["jacobian"],expected_jac,atol=2e-9)
    expected_g=expected_jac.T@expected_jac/3+.03*np.diag(mask_for(widths))
    np.testing.assert_allclose(result["gauss_newton"],expected_g,atol=2e-9)
    assert result["asymmetry"]<1e-12 and result["gauss_newton_eigenvalues"].min()>-1e-12
    shifted=f["geo_curvature"](theta,X,y+2,widths,l2=.03)
    np.testing.assert_array_equal(result["gauss_newton"],shifted["gauss_newton"])
    assert np.max(abs(result["hessian"]-shifted["hessian"]))>.01
    target=scalar_definition(theta,X,np.zeros(3),widths)[1].copy()
    fitted=f["geo_curvature"](theta,X,target,widths,l2=.03)
    np.testing.assert_allclose(fitted["hessian"],fitted["gauss_newton"],atol=2e-15)
    simple=f["geo_curvature"]([.3,-.2],X[:,0:1],y,(1,1),l2=.03)
    design=np.column_stack((X[:,0],np.ones(3)))
    np.testing.assert_allclose(simple["hessian"],design.T@design/3+.03*np.diag([1.,0.]),atol=2e-15)
    np.testing.assert_allclose(simple["residual_curvature"],0.,atol=2e-15)


@pytest.mark.parametrize("scale",[.1,1.,10.])
def test_same_two_weight_function_has_coordinate_dependent_restricted_curvature(scale):
    f=functions();theta=np.array([scale,0.,1/scale,0.]);widths=(1,1,1)
    X=np.array([[-.7],[1.3],[2.]])
    predictions=f["geo_network"](theta,X,np.zeros(3),widths,activation="linear")["prediction"]
    np.testing.assert_allclose(predictions,X[:,0],atol=2e-15)
    # At exact fit x=y=1 the restricted (w,v) block is an independent outer-product oracle.
    H=np.column_stack([f["geo_network"](theta,[[1.]],[1.],widths,"linear",direction=v)["hvp"] for v in np.eye(4)])
    restricted=H[np.ix_([0,2],[0,2])]
    np.testing.assert_allclose(restricted,[[scale**-2,1.],[1.,scale**2]],atol=2e-14)
    np.testing.assert_allclose(np.linalg.eigvalsh(restricted),[0.,scale**2+scale**-2],atol=2e-14)


def test_displayed_scalar_auditor_matches_independent_definition_and_detects_wrong_product(monkeypatch):
    f=functions();theta=np.linspace(-.3,.4,13);X=np.array([[-.7,.2],[.3,-.5],[.8,.4]])
    y=np.array([.4,-.2,.7]);v=np.linspace(.1,.5,13);v/=math.hypot(*v)
    assert f["geo_scalar_objective"](theta,X,y,(2,3,1),.02)==pytest.approx(scalar_definition(theta,X,y,(2,3,1),.02)[0],abs=2e-15)
    report=f["geo_audit"](theta,X,y,(2,3,1),v)
    assert report["loss_error"]<1e-12 and report["gradient_error"]<1e-6 and report["hvp_error"]<1e-6
    original=f["geo_network"]
    def corrupted(*args,**kwargs):
        result=original(*args,**kwargs)
        if result["hvp"] is not None:
            result["hvp"]=np.zeros_like(result["hvp"])
        return result
    monkeypatch.setitem(f,"geo_network",corrupted)
    assert f["geo_audit"](theta,X,y,(2,3,1),v)["hvp_error"]>.01


@pytest.mark.parametrize("scheme",["zero","small","fan_in","glorot","he","large"])
def test_initialization_uses_paired_draws_declared_hidden_scales_and_common_linear_output(scheme):
    f=functions();widths=(2,3,2,1);seed=17
    actual=f["geo_initialize"](widths,seed,scheme)
    rng=np.random.default_rng(seed);expected=[]
    for layer,(a,b) in enumerate(zip(widths[:-1],widths[1:])):
        base=rng.normal(size=(a,b))
        scales=dict(zero=0.,small=.03/math.sqrt(a),fan_in=1/math.sqrt(a),
                    glorot=math.sqrt(2/(a+b)),he=math.sqrt(2/a),large=3/math.sqrt(a))
        scale=0. if scheme=="zero" else (1/math.sqrt(a) if layer==len(widths)-2 else scales[scheme])
        expected.extend((scale*base).ravel());expected.extend([0.]*b)
    np.testing.assert_allclose(actual,expected,atol=3e-16)
    pieces=f["geo_unpack"](actual,widths)
    assert [W.shape for W,b in pieces]==[(2,3),(3,2),(2,1)]
    assert [b.shape for W,b in pieces]==[(3,),(2,),(1,)]
    np.testing.assert_array_equal(f["geo_pack"](pieces),actual)
    pieces[0][0][0,0]=999.
    assert actual[0]!=999.


@pytest.mark.parametrize("activation",["tanh","relu","linear"])
def test_independent_output_vjp_and_layer_moment_records(activation):
    f=functions();widths=(2,3,1);theta=np.linspace(-.5,.7,13)
    X=np.array([[-.4,.2],[.5,-.8],[.7,.3]]);probe=np.array([[.2],[-.3],[.8]])
    result=f["geo_transmission"](theta,X,widths,activation,probe)
    def probe_objective(flat):
        predictions=scalar_definition(theta,flat.reshape(X.shape),np.zeros(3),widths,activation=activation)[1]
        return sum(float(a)*float(b) for a,b in zip(predictions,probe[:,0]))
    expected=central_difference(probe_objective,X.ravel())
    np.testing.assert_allclose(result["input_vjp"].ravel(),expected,atol=2e-9,rtol=3e-6)
    cache=f["geo_forward"](theta,X,widths,activation)
    assert len(result["layers"])==2
    for j,row in enumerate(result["layers"]):
        values=cache["A"][j+1].ravel()
        mean=sum(float(v) for v in values)/len(values)
        second=sum(float(v)**2 for v in values)/len(values)
        variance=sum((float(v)-mean)**2 for v in values)/len(values)
        assert row["activation_mean"]==pytest.approx(mean,abs=2e-15)
        assert row["activation_variance"]==pytest.approx(variance,abs=2e-15)
        assert row["activation_second_moment"]==pytest.approx(second,abs=2e-15)
        assert second==pytest.approx(variance+mean**2,abs=2e-15)
    assert result["layers"][-1]["sensitivity_rms"]==pytest.approx(math.hypot(*probe[:,0])/math.sqrt(3))
    scaled=f["geo_transmission"](theta,X,widths,activation,3*probe)
    np.testing.assert_allclose(scaled["input_vjp"],3*result["input_vjp"],atol=2e-15)
    assert result["upstream_rms"]==pytest.approx(math.hypot(*probe[:,0])/math.sqrt(3))


def test_relu_zero_derivative_convention_and_curvature_rejection():
    f=functions();widths=(1,1,1)
    # w=b=0, v=1, c=0: the declared zero derivative blocks the input probe.
    theta=np.array([0.,0.,1.,0.])
    result=f["geo_transmission"](theta,[[1.],[-1.]],widths,"relu",np.ones((2,1)))
    np.testing.assert_array_equal(result["input_vjp"],np.zeros((2,1)))
    assert result["layers"][0]["low_derivative_fraction"]==1.
    with pytest.raises(ValueError,match="smooth"):
        f["geo_network"](theta,[[1.]],[1.],widths,"relu",direction=np.ones(4))


@pytest.mark.parametrize("batch",[1,2,3,4])
@pytest.mark.parametrize("replacement",[True,False])
def test_exact_enumeration_covariance_and_counts_against_closed_finite_population(batch,replacement):
    f=functions();rows=np.arange(24,dtype=float).reshape(8,3)/7
    result=f["geo_exact_batch_panel"](rows,batch,replacement)
    expected_count=8**batch if replacement else math.comb(8,batch)
    assert result["count"]==expected_count and result["batch_means"].shape==(expected_count,3)
    mean=np.array([sum(float(v) for v in rows[:,j])/8 for j in range(3)])
    S=sum(np.outer(row-mean,row-mean) for row in rows)/8
    multiplier=1/batch if replacement else (8-batch)/(batch*7)
    np.testing.assert_allclose(result["mean"],mean,atol=2e-15)
    np.testing.assert_allclose(result["batch_means"].mean(axis=0),mean,atol=1e-13)
    np.testing.assert_allclose(result["covariance"],multiplier*S,atol=2e-13)
    np.testing.assert_allclose(result["expected_covariance"],multiplier*S,atol=2e-15)
    assert result["maximum_error"]<2e-13


def test_hand_covariance_ddof_all_rows_and_clipping_counterexample():
    f=functions();rows=np.array([[-4.],[2.],[2.]])
    assert f["geo_exact_batch_panel"](rows,2,True)["covariance"][0,0]==pytest.approx(4.)
    assert f["geo_exact_batch_panel"](rows,2,False)["covariance"][0,0]==pytest.approx(2.)
    all_rows=f["geo_exact_batch_panel"](rows,3,False)
    assert all_rows["count"]==1
    np.testing.assert_array_equal(all_rows["covariance"],[[0.]])
    assert rows.mean()==0. and np.clip(rows,-1,1).mean()==pytest.approx(1/3)


def test_per_row_gradients_match_empirical_mean_and_shared_penalty_leaves_covariance():
    f=functions();theta=np.linspace(-.3,.4,13);widths=(2,3,1)
    X=np.linspace(-.8,.9,16).reshape(8,2);y=np.sin(X[:,0])
    rows=f["geo_row_gradients"](theta,X,y,widths)
    penalty=f["geo_row_gradients"](theta,X,y,widths,l2=.1)
    expected=np.array([central_difference(lambda v:scalar_definition(v,X[i:i+1],y[i:i+1],widths)[0],theta) for i in range(8)])
    np.testing.assert_allclose(rows,expected,atol=2e-9,rtol=2e-6)
    np.testing.assert_allclose(rows.mean(axis=0),f["geo_network"](theta,X,y,widths)["gradient"],atol=2e-15)
    np.testing.assert_allclose(penalty-rows,np.broadcast_to(.1*mask_for(widths)*theta,rows.shape),atol=2e-15)
    np.testing.assert_allclose(f["geo_exact_batch_panel"](rows,2)["covariance"],
                               f["geo_exact_batch_panel"](penalty,2)["covariance"],atol=2e-15)


@pytest.mark.parametrize("rate",[.03,.07,.08,.09])
def test_rotated_quadratic_trajectories_against_independent_eigenbasis(rate):
    f=functions();angle=np.pi/5
    U=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    Q=U@np.diag([1.,25.])@U.T;initial=np.array([2.,-1.]);original=Q.copy()
    result=f["geo_quadratic"](Q,initial,rate,80)
    multipliers=(1-rate*np.array([1.,25.]))[None,:]**np.arange(81)[:,None]
    expected=(multipliers*(U.T@initial))@U.T
    np.testing.assert_allclose(result["states"],expected,atol=2e-12,rtol=3e-12)
    expected_loss=np.sum(((expected@U)**2)*[1.,25.],axis=1)/2
    np.testing.assert_allclose(result["losses"],expected_loss,atol=2e-12,rtol=3e-12)
    np.testing.assert_allclose(result["eigenvalues"],[1.,25.],atol=2e-14)
    np.testing.assert_array_equal(Q,original);np.testing.assert_array_equal(initial,[2.,-1.])
    if rate<.08:
        assert result["stable_rate"] is True and result["losses"][-1]<result["losses"][0]
    if rate>.08:
        assert result["stable_rate"] is False and result["losses"][-1]>1e10
    assert f["geo_quadratic"](np.diag([1.,25.]),[1.,0.],.09,80)["losses"][-1]<.5
    assert f["geo_quadratic"](np.diag([1.,25.]),[1.,0.],.09,80)["stable_rate"] is False


def test_full_initialization_panel_counts_pairing_and_unaveraged_upstream():
    f=functions();panel=f["geo_profile_panel"]()
    assert panel["widths"]==(8,16,16,16,16,16,1) and panel["rows"]==128
    assert len(panel["records"])==24 and f["geo_widths"](panel["widths"])[1]==1249
    expected=[(activation,scheme,seed) for activation in ("tanh","relu") for scheme in ("small","glorot","he","large")
              for seed in (40601,40602,40603)]
    assert [(r["activation"],r["scheme"],r["seed"]) for r in panel["records"]]==expected
    for record in panel["records"]:
        assert len(record["layers"])==6 and record["input_vjp"].shape==(128,8)
        assert record["upstream_rms"]==pytest.approx(1.)
        assert record["layers"][-1]["sensitivity_rms"]==pytest.approx(1.)
        assert record["layers"][-1]["derivative_rms"]==1.
        for row in record["layers"]:
            assert row["activation_second_moment"]==pytest.approx(row["activation_variance"]+row["activation_mean"]**2,rel=2e-14,abs=2e-13)
            assert 0<=row["low_derivative_fraction"]<=1
    tiny=[r for r in panel["records"] if r["scheme"]=="small"]
    assert max(r["input_rms"] for r in tiny)<1e-6
    for seed in (40601,40602,40603):
        paired=[r for r in panel["records"] if r["activation"]=="relu" and r["seed"]==seed]
        reference=[row["derivative_rms"] for row in paired[0]["layers"]]
        for record in paired[1:]:
            np.testing.assert_array_equal([row["derivative_rms"] for row in record["layers"]],reference)


def test_failure_data_training_only_scaler_mirrored_dependence_and_complete_outcomes():
    f=functions();data=f["geo_failure_data"]()
    samples=[]
    for child in np.random.SeedSequence(40610).spawn(2):
        rng=np.random.default_rng(child);a=rng.uniform(-1.5,1.5,48);b=.7*a+.25*rng.normal(size=48)
        X=np.column_stack((a,b));y=np.sin(1.7*a)+.2*b
        samples.append((np.r_[X,-X],np.r_[y,-y]))
    mean=samples[0][0].mean(axis=0);scale=samples[0][0].std(axis=0,ddof=0)
    np.testing.assert_array_equal(data["mean"],mean);np.testing.assert_array_equal(data["scale"],scale)
    for key,(X,y) in zip(("train","validation"),samples):
        np.testing.assert_array_equal(data[key][0],(X-mean)/scale)
        np.testing.assert_array_equal(data[key][1],y)
        np.testing.assert_array_equal(y[48:],-y[:48])
    assert not np.allclose(data["validation"][0].std(axis=0),np.ones(2))
    panel=f["geo_failure_panel"]()
    assert len(panel["records"])==15 and panel["selection"] is None
    assert panel["completed_updates"]==panel["attempted_updates"]==3620
    assert {r["status"] for r in panel["records"]}=={"fixed_budget","objective_limit"}
    for record in panel["records"]:
        assert len(record["trace"])==record["completed_updates"]+1
        assert [r["t"] for r in record["trace"]]==list(range(record["completed_updates"]+1))
        assert record["last_gradient_norm"]==record["trace"][-1]["gradient_norm"]
        assert record["small_gradient"]==(record["last_gradient_norm"]<1e-6)
        assert np.isfinite(record["validation_half_mse"]) and record["seconds"]>=0
        if record["name"]=="zero":
            assert record["small_gradient"] and record["trace"][-1]["loss"]>.4
            assert record["completed_updates"]==300
            np.testing.assert_allclose(record["theta"],0.,atol=1e-15)
        if record["name"]=="oversized rate":
            assert record["status"]=="objective_limit" and record["completed_updates"]<10
            assert record["trace"][-2]["loss"]<=1e12<record["trace"][-1]["loss"]
            paired=next(r for r in panel["records"] if r["name"]=="Glorot" and r["seed"]==record["seed"])
            np.testing.assert_array_equal(record["initial"],paired["initial"])
        if record["name"]=="tiny weights":
            assert record["trace"][-1]["loss"]<.03


def test_failure_fit_keeps_last_finite_snapshot_and_actual_attempt_count(monkeypatch):
    f=functions();theta=np.linspace(-.3,.4,13);X=np.array([[-.7,.2],[.3,-.5]]);y=np.array([.4,-.2])
    before=theta.copy();original=f["geo_network"];calls=0
    def fail_third(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==3:
            raise FloatingPointError("injected candidate failure")
        return original(*args,**kwargs)
    monkeypatch.setitem(f,"geo_network",fail_third)
    result=f["geo_fit"](theta,X,y,(2,3,1),steps=5)
    assert result["status"]=="numerical_failure" and "injected" in result["error"]
    assert result["completed_updates"]==1 and result["attempted_updates"]==2 and len(result["trace"])==2
    np.testing.assert_array_equal(theta,before)
    first=original(theta,X,y,(2,3,1))
    np.testing.assert_array_equal(result["theta"],theta-.05*first["gradient"])


@pytest.mark.parametrize("bad",[(0,1),(1,2),(1,33,1),(True,1),(1,)*9])
def test_invalid_width_contracts(bad):
    with pytest.raises(ValueError):
        functions()["geo_widths"](bad)


def test_nonfinite_shapes_dense_bounds_and_visible_numeric_range_failures():
    f=functions()
    for value in ([np.inf],[1j],[True],[[1.]]):
        with pytest.raises(ValueError):
            f["geo_unpack"](value,(1,1))
    with pytest.raises(ValueError):
        f["geo_network"]([.2,0.],np.ones((513,1)),np.zeros(513),(1,1))
    with pytest.raises(ValueError):
        f["geo_network"]([.2,0.],[[1.]],[[1.]],(1,1))
    with pytest.raises(ValueError):
        f["geo_network"]([.2,0.],[[1.]],[1.],(1,1),direction=[1.])
    with pytest.raises(ValueError):
        f["geo_curvature"](f["geo_initialize"]((2,32,1)),[[0.,0.]],[0.],(2,32,1))
    # A finite huge unused weight must not be squared for a zero penalty.
    theta=np.array([1e200,0.,0.,0.])
    assert f["geo_network"](theta,[[0.]],[0.],(1,1,1),l2=0.)["loss"]==0.
    with pytest.raises(FloatingPointError):
        f["geo_network"](theta,[[0.]],[0.],(1,1,1),l2=.1)
    with pytest.raises(FloatingPointError):
        f["geo_network"]([1e308,0.,0.,0.],[[2.]],[0.],(1,1,1))
    for rows,batch,replacement in [(np.zeros((1,1)),1,True),(np.zeros((12,1)),4,True),(np.zeros((3,1)),0,True)]:
        with pytest.raises(ValueError):
            f["geo_exact_batch_panel"](rows,batch,replacement)
    with pytest.raises(FloatingPointError):
        f["geo_exact_batch_panel"]([[1e200],[-1e200]],1,True)
    with pytest.raises(ValueError):
        f["geo_quadratic"]([[1.,1.],[0.,1.]],[1.,1.])
    with pytest.raises(ValueError):
        f["geo_quadratic"](np.diag([0.,1.]),[1.,1.])
