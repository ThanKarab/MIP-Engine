import numpy
import pandas as pd

from typing import TypeVar

from mipengine.algorithm_result_DTOs import TabularDataResult
from mipengine.table_data_DTOs import ColumnDataFloat
from mipengine.table_data_DTOs import ColumnDataStr
from mipengine.udfgen import TensorBinaryOp
from mipengine.udfgen import TensorUnaryOp
from mipengine.udfgen import literal
#from mipengine.udfgen import scalar
from mipengine.udfgen import merge_tensor
from mipengine.udfgen import relation
from mipengine.udfgen import tensor
from mipengine.udfgen import udf

from mipengine.udfgen import transfer
from mipengine.udfgen import merge_transfer
from mipengine.udfgen import state


from pydantic import BaseModel
from typing import List

import json

T = TypeVar('T')
S = TypeVar("S")

class KmeansResult(BaseModel):
    title: str
    centers: List[List[float]]
    init_centers: List[List[float]]

def run(algo_interface):
    local_run = algo_interface.run_udf_on_local_nodes
    global_run = algo_interface.run_udf_on_global_node

    #X_relation = algo_interface.initial_view_tables["x"]

    X_relation, *_ = algo_interface.create_primary_data_views(
        variable_groups=[algo_interface.x_variables],dropna=True
    )

    n_clusters = algo_interface.algorithm_parameters["k"]
    tol = algo_interface.algorithm_parameters["tol"]
    maxiter = algo_interface.algorithm_parameters["maxiter"]

    X_not_null = local_run(
        func=relation_to_matrix,
        positional_args=[X_relation],
    )

    #X_not_null = local_run(
    #    func=remove_nulls,
    #    positional_args=[X],
    #)

    centers_local = local_run(
        func=init_centers_local,
        positional_args=[X_not_null,n_clusters],
        share_to_global=[True]
    )

    global_state,global_result = global_run(
        func=init_centers_global,
        positional_args=[centers_local,n_clusters],
        share_to_locals=[False,True]
    )

    curr_iter =0
    centers_to_compute = global_result

    init_centers = json.loads(centers_to_compute.get_table_data()[1][0])["centers"]
    init_centers_array = numpy.array(init_centers)
    init_centers_list = init_centers_array.tolist()
    while True:
        label_state = local_run(
            func=compute_cluster_labels,
            positional_args=[X_not_null,centers_to_compute],
            share_to_global=[False]
        )

        metrics_local = local_run(
            func=compute_metrics,
            positional_args=[X_not_null,label_state,n_clusters],
            share_to_global=[True]
        )

        new_centers = global_run(
            func=compute_centers_from_metrics,
            positional_args=[metrics_local,n_clusters],
            share_to_locals=[True]
        )

        curr_iter+=1

        old_centers = json.loads(centers_to_compute.get_table_data()[1][0])["centers"]
        old_centers_array = numpy.array(old_centers)

        new_centers_obj = json.loads(new_centers.get_table_data()[1][0])["centers"]
        new_centers_array = numpy.array(new_centers_obj)

        diff = numpy.sum((new_centers_array-old_centers_array)**2)

        if (curr_iter > maxiter) or (diff<tol):
            ret_obj = KmeansResult(
                title="K-Means Centers",
                centers=new_centers_array.tolist(),
                init_centers = init_centers_list
            )
            return ret_obj

        else:
            centers_to_compute = new_centers
    return ret_obj

@udf(rel=relation(S), return_type=tensor(float, 2))
def relation_to_matrix(rel):
    return rel

@udf(a=tensor(T, 2), return_type=tensor(T, 2))
def remove_nulls(a):
    a_sel = a[~numpy.isnan(a).any(axis=1)]
    return a_sel


@udf(X= tensor(T, 2),n_clusters=literal(),return_type=[transfer()])
def init_centers_local(X,n_clusters):
   from sklearn.utils import check_random_state
   seed = 123
   n_samples = X.shape[1]
   random_state = check_random_state(seed)
   seeds = random_state.permutation(n_samples)[:n_clusters]
   centers = X[seeds]
   #np.random.rand(n_samples,n_clusters)
   transfer_ = {'centers':centers.tolist()}
   return transfer_

@udf(centers_transfer=merge_transfer(),n_clusters=literal(),return_type=[state(),transfer()])
def init_centers_global(centers_transfer,n_clusters):
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
    from sklearn.metrics.pairwise import euclidean_distances
    centers = numpy.array(global_transfer['centers'])
    distances = euclidean_distances(X,centers)

    labels = numpy.argmin(distances, axis=1)

    return_labels = {'labels': labels.tolist()}

    return return_labels

@udf(X=tensor(dtype=T, ndims=2), label_state=state(),n_clusters=literal(), return_type=transfer())
def compute_metrics(X,label_state,n_clusters):
    labels = numpy.array(label_state['labels'])
    metrics = {}
    for i in range(n_clusters):
        relevant_features = numpy.where(labels == i)
        X_clust = X[relevant_features,:]
        X_clust = X_clust[0,:,:]
        X_sum = numpy.sum(X_clust, axis=0)
        X_count = X_clust.shape[0]
        #raise ValueError(str(relevant_features)+''+str(labels.shape)+' '+str(X.shape)+' '+str(X_sum.shape)+' '+str(X_clust.shape))
        metrics[i] = {'X_sum': X_sum.tolist(),'X_count':X_count}
    return metrics

@udf(transfers =merge_transfer(),n_clusters =literal(), return_type=transfer())
def compute_centers_from_metrics(transfers,n_clusters):
    centers = []
    #raise ValueError(transfers)
    n_dim = numpy.array(transfers[0][str(0)]['X_sum']).shape[0]
    for i in range(n_clusters):
        curr_sum = numpy.zeros((1,n_dim))
        curr_count = 0
        for curr_transfer in transfers:
            X_sum = numpy.array(curr_transfer[str(i)]['X_sum'])
            X_count = curr_transfer[str(i)]['X_count']
            curr_sum += X_sum
            curr_count += X_count
        if curr_count !=0:
            final_i = curr_sum/curr_count
        else:
            final_i = curr_sum
        centers.append(final_i)
    centers_array = numpy.vstack(centers)
    #raise ValueError(centers_array.shape)
    ret_val = centers_array.tolist()
    ret_val2 = {'centers':ret_val}
    return ret_val2
