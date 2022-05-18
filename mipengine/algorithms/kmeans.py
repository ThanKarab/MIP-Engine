import numpy
import pandas as pd

from typing import TypeVar

from mipengine.algorithm_result_DTOs import TabularDataResult
from mipengine.table_data_DTOs import ColumnDataFloat
from mipengine.table_data_DTOs import ColumnDataStr
from mipengine.udfgen import TensorBinaryOp
from mipengine.udfgen import TensorUnaryOp
from mipengine.udfgen import literal
from mipengine.udfgen import merge_tensor
from mipengine.udfgen import relation
from mipengine.udfgen import tensor
from mipengine.udfgen import udf

from mipengine.udfgen import transfer
from mipengine.udfgen import merge_transfer
from mipengine.udfgen import state

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_random_state

T = TypeVar('T')
S = TypeVar("S")

def run(algo_interface):
    local_run = algo_interface.run_udf_on_local_nodes
    global_run = algo_interface.run_udf_on_global_node

    X_relation = algo_interface.initial_view_tables["x"]

    n_clusters = algo_interface.algorithm_parameters["k"]
    tol = algo_interface.algorithm_parameters["tol"]
    maxiter = algo_interface.algorithm_parameters["maxiter"]

    X = local_run(
        func=relation_to_matrix,
        positional_args=[X_relation],
    )

    X_not_null = local_run(
        func=remove_nulls,
        positional_args=[X],
    )

    centers_local = local_run(
        func=init_centers_local,
        positional_args=[X_not_null],
        share_to_global=[True]
    )

    global_state,global_result = global_run(
        func=init_centers_global,
        positional_args=[local_result],
        share_to_locals=[False,True]
    )

    curr_iter =0
    while True:
        label_state = local_run(
            func=compute_cluster_labels,
            positional_args=[X_not_null,global_result],
            share_to_global=[False]
        )

        metrics_local = local_run(
            func=compute_metrics,
            positional_args=[X_not_null,labels_state,n_clusters],
            share_to_global=[True]
        )

        new_centers = local_run(
            func=compute_centers_from_metrics,
            positional_args=[metrics_local,n_clusters],
            share_to_locals=[True]
        )

        curr_iter+=1
        if (curr_iter > maxiter) or (diff<tol):
            break
    return ret_obj

@udf(rel=relation(S), return_type=tensor(float, 2))
def relation_to_matrix(rel):
    return rel

@udf(a=tensor(T, 2), return_type=tensor(T, 2))
def remove_nulls(a):
    a_sel = a[~numpy.isnan(a).any(axis=1)]
    return a_sel


@udf(X= tensor(T, 2),n_clusters=scalar(int),return_type=[transfer()])
def init_centers_local(X,n_clusters):
   seed = 123
   n_samples = X.shape[1]
   random_state = check_random_state(seed)
   seeds = random_state.permutation(n_samples)[:n_clusters]
   centers = X[seeds]
   #np.random.rand(n_samples,n_clusters)
   transfer_ = {'centers':centers.tolist()}
   return transfer_

@udf(centers_transfer=merge_transfer(),n_clusters=scalar(int),return_type=[state(),transfer()])
def init_centers_global(centers_transfer):
   centers_all = []
   for curr_transfer in centers_transfer:
       centers_all.append(curr_transfer['centers'])
    centers_merged = numpy.vstack(centers_all)
    centers_global = centers_merged[:n_clusters]

    transfer_ = {'centers':centers_global.tolist()}
    state_ = {'centers':centers_global.tolist()}
    return state_,transfer_



@udf(X=tensor(dtype=T, ndims=2), global_transfer=transfer(), return_type=state())
def compute_cluster_labels(X,global_transfer):
    centers = numpy.array(global_transfer['centers'])
    distances = euclidean_distances(X,centers)

    labels = numpy.argmin(distances, axis=1)

    return_labels = {'labels': labels.tolist()}

    return return_labels

@udf(X=tensor(dtype=T, ndims=2), label_state=state(),n_clusters = scalar(int), return_type=transfer())
def compute_metrics(X,labels_state,n_clusters):
    labels = numpy.array(label_state)
    metrics = {}
    for i in range(n_clusters):
        relevant_features = np.where(labels == i)
        X_clust = X[labels == relevant_features,:]
        X_sum = numpy.sum([X_clust, axis=0)
        X_count = X_clust.shape[0]
        metrics[i] = {'X_sum': X_sum.tolist(),'X_count':X_count.tolist()}
    return metrics

@udf(X=tensor(transfers =merge_transfer(),n_clusters = scalar(int), return_type=transfer())
def compute_centers_from_metrics(transfers,n_clusters):
    centers = []
    n_dim = numpy.array(transfers[0][0]['X_sum']).shape[1]
    for i in range(n_clusters):
        curr_sum = numpy.zeros(1,n_dim)
        curr_count = numpy.zeros(1,n_dim)
        for curr_transfer in transfers:
            X_sum = numpy.array(curr_transfer[i]['X_sum'])
            X_count = numpy.array(curr_transfer[i]['X_count'])
            curr_sum += X_sum
            curr_count += X_count
        final_i = np.divide(curr_sum, curr_count, out=np.zeros_like(curr_sum), where=curr_count!=0)
        centers.append(final_i)
    centers = numpy.array(centers)
    return centers.to_list()
