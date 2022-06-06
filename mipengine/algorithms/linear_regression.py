import json
from typing import List

import numpy
import scipy.stats as stats
from pydantic import BaseModel

from mipengine.algorithms.preprocessing import DummyEncoder
from mipengine.algorithms.preprocessing import relation_to_vector
from mipengine.udfgen import literal
from mipengine.udfgen import relation
from mipengine.udfgen import secure_transfer
from mipengine.udfgen import state
from mipengine.udfgen import tensor
from mipengine.udfgen import transfer
from mipengine.udfgen import udf

ALPHA = 0.05  # NOTE maybe this should be a model parameter

RealVector = tensor(dtype=float, ndims=1)


class LinearRegressionResult(BaseModel):
    dependent_var: str
    n_obs: int
    df_resid: float
    df_model: float
    rse: float
    r_squared: float
    r_squared_adjusted: float
    f_stat: float
    f_pvalue: float
    indep_vars: List[str]
    coefficients: List[float]
    std_err: List[float]
    t_stats: List[float]
    pvalues: List[float]
    lower_ci: List[float]
    upper_ci: List[float]


def run(executor):
    X, y = executor.create_primary_data_views(
        variable_groups=[executor.x_variables, executor.y_variables],
    )

    dummy_encoder = DummyEncoder(executor)
    X = dummy_encoder.transform(X)

    p = len(dummy_encoder.new_varnames) - 1

    lr = LinearRegression(executor)
    lr.fit(X=X, y=y)
    y_pred: RealVector = lr.predict(X)
    lr.compute_summary(
        y_test=relation_to_vector(y, executor),
        y_pred=y_pred,
        p=p,
    )

    result = LinearRegressionResult(
        dependent_var=executor.y_variables[0],
        n_obs=lr.n_obs,
        df_resid=lr.df,
        df_model=p,
        rse=lr.rse,
        r_squared=lr.r_squared,
        r_squared_adjusted=lr.r_squared_adjusted,
        f_stat=lr.f_stat,
        f_pvalue=lr.f_p_value,
        indep_vars=dummy_encoder.new_varnames,
        coefficients=[c[0] for c in lr.coefficients],
        std_err=lr.std_err.tolist(),
        t_stats=lr.t_stat.tolist(),
        pvalues=lr.t_p_values.tolist(),
        lower_ci=lr.ci[0].tolist(),
        upper_ci=lr.ci[1].tolist(),
    )
    return result


class LinearRegression:
    def __init__(self, executor):
        self.local_run = executor.run_udf_on_local_nodes
        self.global_run = executor.run_udf_on_global_node

    def fit(self, X, y):
        local_transfers = self.local_run(
            func=self._fit_local,
            keyword_args={"x": X, "y": y},
            share_to_global=[True],
        )
        self.global_state, self.global_transfer = self.global_run(
            func=self._fit_global,
            keyword_args=dict(local_transfers=local_transfers),
            share_to_locals=[False, True],
        )
        global_transfer_data = json.loads(self.global_transfer.get_table_data()[1][0])
        self.coefficients = global_transfer_data["coefficients"]

    @staticmethod
    @udf(x=relation(), y=relation(), return_type=[secure_transfer(sum_op=True)])
    def _fit_local(x, y):
        xTx = x.T @ x
        xTy = x.T @ y
        n_obs_train = len(y)

        stransfer = {}
        stransfer["xTx"] = {"data": xTx.to_numpy().tolist(), "operation": "sum"}
        stransfer["xTy"] = {"data": xTy.to_numpy().tolist(), "operation": "sum"}
        stransfer["n_obs_train"] = {"data": n_obs_train, "operation": "sum"}
        return stransfer

    @staticmethod
    @udf(
        local_transfers=secure_transfer(sum_op=True),
        return_type=[state(), transfer()],
    )
    def _fit_global(local_transfers):
        xTx = numpy.array(local_transfers["xTx"])
        xTy = numpy.array(local_transfers["xTy"])
        n_obs_train = local_transfers["n_obs_train"]

        xTx_inv = numpy.linalg.pinv(xTx)
        coefficients = xTx_inv @ xTy

        state_ = {}
        state_["xTx_inv"] = xTx_inv  # Needed for SE(β) calculation
        state_["n_obs_train"] = n_obs_train

        transfer_ = {}
        transfer_["coefficients"] = coefficients.tolist()
        return state_, transfer_

    def predict(self, X):
        return self.local_run(
            func=self._predict_local,
            keyword_args=dict(x=X, coefficients=self.coefficients),
            share_to_global=[False],
        )

    @staticmethod
    @udf(x=relation(), coefficients=literal(), return_type=RealVector)
    def _predict_local(x, coefficients):
        x = x.values
        coefficients = numpy.array(coefficients)

        y_pred = x @ coefficients
        return y_pred

    def compute_summary(self, y_test, y_pred, p):
        local_transfers = self.local_run(
            func=self._compute_summary_local,
            keyword_args=dict(y_test=y_test, y_pred=y_pred),
            share_to_global=[True],
        )
        global_transfer = self.global_run(
            func=self._compute_summary_global,
            keyword_args=dict(
                local_transfers=local_transfers, fit_gstate=self.global_state
            ),
        )
        global_transfer_data = json.loads(global_transfer.get_table_data()[1][0])
        rss = global_transfer_data["rss"]
        tss = global_transfer_data["tss"]
        sum_abs_resid = global_transfer_data["sum_abs_resid"]
        xTx_inv = numpy.array(global_transfer_data["xTx_inv"])
        coefficients = numpy.array(self.coefficients)
        n_obs_train = global_transfer_data["n_obs_train"]
        n_obs_test = global_transfer_data["n_obs_test"]
        df = n_obs_train - p - 1
        self.n_obs = n_obs_train
        self.df = df
        self.rse = (rss / df) ** 0.5
        self.std_err = ((self.rse**2) * numpy.diag(xTx_inv)) ** 0.5
        self.t_stat = coefficients.T[0] / self.std_err
        self.ci = (
            coefficients.T[0] - stats.t.ppf(1 - ALPHA / 2, df) * self.std_err,
            coefficients.T[0] + stats.t.ppf(1 - ALPHA / 2, df) * self.std_err,
        )
        self.r_squared = 1.0 - rss / tss
        self.r_squared_adjusted = 1 - (1 - self.r_squared) * (n_obs_train - 1) / df
        self.f_stat = (tss - rss) * df / (p * rss)
        self.t_p_values = stats.t.sf(abs(self.t_stat), df=df) * 2
        self.f_p_value = stats.f.sf(self.f_stat, dfn=p, dfd=df)
        # Quanities below are only used in cross validation
        self.rmse = (rss / n_obs_test) ** 0.5
        self.mae = sum_abs_resid / n_obs_test

    @staticmethod
    @udf(
        y_test=RealVector,
        y_pred=RealVector,
        return_type=secure_transfer(sum_op=True),
    )
    def _compute_summary_local(y_test, y_pred):
        rss = sum((y_test - y_pred) ** 2)
        sum_y_test = sum(y_test)
        sum_sq_y_test = sum(y_test**2)
        sum_abs_resid = sum(abs(y_test - y_pred))
        n_obs_test = len(y_test)

        stransfer = {}
        stransfer["rss"] = {"data": float(rss), "operation": "sum"}
        stransfer["sum_y_test"] = {"data": float(sum_y_test), "operation": "sum"}
        stransfer["sum_sq_y_test"] = {"data": float(sum_sq_y_test), "operation": "sum"}
        stransfer["sum_abs_resid"] = {"data": float(sum_abs_resid), "operation": "sum"}
        stransfer["n_obs_test"] = {"data": n_obs_test, "operation": "sum"}
        return stransfer

    @staticmethod
    @udf(
        fit_gstate=state(),
        local_transfers=secure_transfer(sum_op=True),
        return_type=transfer(),
    )
    def _compute_summary_global(fit_gstate, local_transfers):
        xTx_inv = fit_gstate["xTx_inv"]
        n_obs_train = fit_gstate["n_obs_train"]
        n_obs_test = local_transfers["n_obs_test"]
        rss = local_transfers["rss"]
        sum_abs_resid = local_transfers["sum_abs_resid"]
        sum_y_test = local_transfers["sum_y_test"]
        sum_sq_y_test = local_transfers["sum_sq_y_test"]

        y_mean_test = sum_y_test / n_obs_test

        # Federated computation of TSS in a single round
        # \sum_i (y_i - ymean)^2 = \sum_i yi^2 - 2 ymean \sum_i yi + n ymean^2
        tss = (
            sum_sq_y_test - 2 * y_mean_test * sum_y_test + n_obs_test * y_mean_test**2
        )

        transfer_ = {}
        transfer_["rss"] = rss
        transfer_["tss"] = tss
        transfer_["sum_abs_resid"] = sum_abs_resid
        transfer_["n_obs_train"] = n_obs_train
        transfer_["n_obs_test"] = n_obs_test
        transfer_["xTx_inv"] = xTx_inv.tolist()
        return transfer_
