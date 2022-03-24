import asyncio
import sys
from typing import Dict
from typing import List
from typing import Tuple

from asgiref.sync import sync_to_async

from mipengine.controller import config as controller_config
from mipengine.controller import controller_logger as ctrl_logger
from mipengine.controller.celery_app import get_node_celery_app
from mipengine.controller.data_model_registry import data_model_registry
from mipengine.controller.node_address import _get_nodes_addresses
from mipengine.controller.node_registry import node_registry
from mipengine.node_info_DTOs import NodeInfo
from mipengine.node_info_DTOs import NodeRole
from mipengine.node_tasks_DTOs import CommonDataElement
from mipengine.node_tasks_DTOs import CommonDataElements

NODE_LANDSCAPE_AGGREGATOR_REQUEST_ID = "NODE_LANDSCAPE_AGGREGATOR"
# TODO remove import get_node_celery_app, pass the celery app  (inverse dependency)
# so the module can be easily unit tested

logger = ctrl_logger.get_background_service_logger()

GET_NODE_INFO_SIGNATURE = "mipengine.node.tasks.common.get_node_info"
GET_NODE_DATASETS_PER_DATA_MODEL_SIGNATURE = (
    "mipengine.node.tasks.common.get_node_datasets_per_data_model"
)
GET_DATA_MODEL_CDES_SIGNATURE = "mipengine.node.tasks.common.get_data_model_cdes"
NODE_LANDSCAPE_AGGREGATOR_UPDATE_INTERVAL = (
    controller_config.node_landscape_aggregator_update_interval
)

CELERY_TASKS_TIMEOUT = controller_config.rabbitmq.celery_tasks_timeout


async def _get_nodes_info(nodes_socket_addr: List[str]) -> List[NodeInfo]:
    celery_apps = [
        get_node_celery_app(socket_addr) for socket_addr in nodes_socket_addr
    ]
    nodes_task_signature = {
        celery_app: celery_app.signature(GET_NODE_INFO_SIGNATURE)
        for celery_app in celery_apps
    }

    tasks_coroutines = [
        _task_to_async(task, connection=app.broker_connection())(
            request_id=NODE_LANDSCAPE_AGGREGATOR_REQUEST_ID
        )
        for app, task in nodes_task_signature.items()
    ]
    results = await asyncio.gather(*tasks_coroutines, return_exceptions=True)
    nodes_info = [
        NodeInfo.parse_raw(result)
        for result in results
        if not isinstance(result, Exception)
    ]
    return nodes_info


async def _get_node_datasets_per_data_model(
    node_socket_addr: str,
) -> Dict[str, Dict[str, str]]:
    celery_app = get_node_celery_app(node_socket_addr)
    task_signature = celery_app.signature(GET_NODE_DATASETS_PER_DATA_MODEL_SIGNATURE)

    result = await _task_to_async(
        task_signature, connection=celery_app.broker_connection()
    )(request_id=NODE_LANDSCAPE_AGGREGATOR_REQUEST_ID)

    datasets_per_data_model = {}
    if not isinstance(result, Exception):
        datasets_per_data_model = {
            data_model: datasets for data_model, datasets in result.items()
        }
    return datasets_per_data_model


async def _get_node_cdes(node_socket_addr: str, data_model: str) -> CommonDataElements:
    celery_app = get_node_celery_app(node_socket_addr)
    task_signature = celery_app.signature(GET_DATA_MODEL_CDES_SIGNATURE)

    result = await _task_to_async(
        task_signature, connection=celery_app.broker_connection()
    )(data_model=data_model, request_id=NODE_LANDSCAPE_AGGREGATOR_REQUEST_ID)

    if not isinstance(result, Exception):
        return CommonDataElements.parse_raw(result)


def _task_to_async(task, connection):
    """
    Converts a Celery task to an async function
    Celery doesn't currently support asyncio "await" while "getting" a result
    Copied from https://github.com/celery/celery/issues/6603
    when broker(rabbitmq) is down, if the existing broker connection is not passed in
    apply_async (in _task_to_async::wrapper), celery (or anyway some internal celery
    component) will try to create a new connection to the broker until the apply_async
    succeeds, which causes the call to apply_async to hang indefinitely until the
    broker is back up. This way(passing the existing broker connection to apply_async)
    it raises a ConnectionResetError or an OperationalError and it does not hang
    """

    async def wrapper(*args, **kwargs):
        total_delay = 0
        delay = 0.1
        # Since apply_async is used instead of delay so that we can pass the connection as an argument,
        # the args and kwargs need to be passed as named arguments.
        async_result = await sync_to_async(task.apply_async)(
            args=args, kwargs=kwargs, connection=connection
        )
        while not async_result.ready():
            total_delay += delay
            if total_delay > CELERY_TASKS_TIMEOUT:
                raise TimeoutError(
                    f"Celery task: {task} didn't respond in {CELERY_TASKS_TIMEOUT}s."
                )
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 2)  # exponential backoff, max 2 seconds
        return async_result.get(timeout=CELERY_TASKS_TIMEOUT - total_delay)

    return wrapper


def _get_node_socket_addr(node_info: NodeInfo):
    return f"{node_info.ip}:{node_info.port}"


class NodeLandscapeAggregator:
    def __init__(self):
        self.keep_updating = True

    async def update(self):
        """
        Node Landscape Aggregator(NLA) is a module that handles the aggregation of necessary information,
        to keep up-to-date and in sync the Node Registry and the Data Model Registry.
        The Node Registry contains information about the node such as id, ip, port etc.
        The Data Model Registry contains two types of information, data_models and datasets_location.
        data_models contains information about the data models and their corresponding cdes.
        datasets_location contains information about datasets and their locations(nodes).
        NLA periodically will send requests (get_node_info, get_node_datasets_per_data_model, get_data_model_cdes),
        to the nodes to retrieve the current information that they contain.
        Once all information about data models and cdes is aggregated,
        any data model that is incompatible across nodes will be removed.
        A data model is incompatible when the cdes across nodes are not identical, except one edge case.
        The edge case is that the cdes can only contain a difference in the field of 'enumerations' in
        the cde with code 'dataset' and still be considered compatible.
        For each data model the 'enumerations' field in the cde with code 'dataset' is updated with all datasets across nodes.
        Once all the information is aggregated and validated the NLA will provide the information to the Node Registry and to the Data Model Registry.
        """
        while self.keep_updating:
            nodes_addresses = _get_nodes_addresses()
            nodes_info = await _get_nodes_info(nodes_addresses)
            local_nodes = [
                node for node in nodes_info if node.role == NodeRole.LOCALNODE
            ]
            datasets_locations = await _get_datasets_locations(local_nodes)
            datasets_labels = await _get_datasets_labels(local_nodes)
            data_model_cdes_across_nodes = await _get_cdes_across_nodes(local_nodes)
            compatible_data_models = _get_compatible_data_models(
                data_model_cdes_across_nodes
            )
            data_models = _update_data_model_dataset_enumerations(
                compatible_data_models, datasets_labels
            )
            datasets_locations = {
                common_data_model: datasets_locations[common_data_model]
                for common_data_model in data_models
            }

            node_registry.set_nodes(
                {node_info.id: node_info for node_info in nodes_info}
            )
            data_model_registry.set_data_models(data_models)
            data_model_registry.set_datasets_location(datasets_locations)
            logger.debug(f"Nodes:{[node for node in node_registry.nodes]}")
            # ..to print full nodes info
            # from devtools import debug
            # debug(self.nodes)
            # DEBUG end

            await asyncio.sleep(NODE_LANDSCAPE_AGGREGATOR_UPDATE_INTERVAL)


async def _get_datasets_locations(nodes: List[NodeInfo]) -> Dict[str, Dict[str, str]]:
    datasets_locations = {}
    for node_info in nodes:
        node_socket_addr = _get_node_socket_addr(node_info)
        datasets_per_data_model = await _get_node_datasets_per_data_model(
            node_socket_addr
        )
        for data_model, datasets in datasets_per_data_model.items():
            if data_model not in datasets_locations:
                datasets_locations[data_model] = {dataset: [] for dataset in datasets}
            for dataset in datasets:
                datasets_locations[data_model][dataset].append(node_info.id)
    return datasets_locations


async def _get_datasets_labels(nodes: List[NodeInfo]) -> Dict[str, Dict[str, str]]:
    datasets_labels = {}
    for node_info in nodes:
        node_socket_addr = _get_node_socket_addr(node_info)
        datasets_per_data_model = await _get_node_datasets_per_data_model(
            node_socket_addr
        )
        for data_model, datasets in datasets_per_data_model.items():
            datasets_labels[data_model] = {}
            for dataset in datasets:
                datasets_labels[data_model][dataset] = datasets[dataset]
    return datasets_labels


async def _get_cdes_across_nodes(
    nodes: List[NodeInfo],
) -> Dict[str, List[Tuple[str, CommonDataElements]]]:
    nodes_cdes = {}
    for node_info in nodes:
        node_socket_addr = _get_node_socket_addr(node_info)
        datasets_per_data_model = await _get_node_datasets_per_data_model(
            node_socket_addr
        )
        for data_model in datasets_per_data_model:
            cdes = await _get_node_cdes(node_socket_addr, data_model)
            if data_model not in nodes_cdes:
                nodes_cdes[data_model] = []
            nodes_cdes[data_model].append((node_info.id, cdes))
    return nodes_cdes


def _get_compatible_data_models(
    data_model_cdes_across_nodes: Dict[str, List[Tuple[str, CommonDataElements]]]
) -> Dict[str, CommonDataElements]:
    """
    This function accepts data_models and a list of  it's cdes across nodes.
    And returns a dictionary with key data model and value the first cdes in case all the cdes are compatible across nodes.
    Args:
        data_model_cdes_across_nodes: data_models and their cdes across nodes
    """
    data_models = {}
    for data_model, cdes_from_all_nodes in data_model_cdes_across_nodes.items():
        first_node, first_cdes = cdes_from_all_nodes[0]
        for node, cdes in cdes_from_all_nodes[1:]:
            if not first_cdes == cdes:
                logger.info(
                    f"Node '{first_node}' and node '{node}' on data model '{data_model}' have incompatibility on the following cdes: {first_cdes} and {cdes}"
                )
                break
        else:
            data_models[data_model] = first_cdes

    return data_models


def _update_data_model_dataset_enumerations(
    data_models: Dict[str, CommonDataElements],
    datasets_labels: Dict[str, Dict[str, str]],
) -> Dict[str, CommonDataElements]:
    for data_model in data_models:
        dataset_cde = data_models[data_model].values["dataset"]
        new_dataset_cde = CommonDataElement(
            code=dataset_cde.code,
            label=dataset_cde.label,
            sql_type=dataset_cde.sql_type,
            is_categorical=dataset_cde.is_categorical,
            enumerations=datasets_labels[data_model],
            min=dataset_cde.min,
            max=dataset_cde.max,
        )
        data_models[data_model].values["dataset"] = new_dataset_cde
    return data_models


node_landscape_aggregator = NodeLandscapeAggregator()
