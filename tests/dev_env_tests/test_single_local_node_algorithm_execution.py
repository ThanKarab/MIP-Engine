import os
from unittest.mock import patch

import pytest

from mipengine import AttrDict
from mipengine.algorithm_result_DTOs import TabularDataResult
from mipengine.common_data_elements import CommonDataElement
from mipengine.common_data_elements import CommonDataElements
from mipengine.common_data_elements import MetadataEnumeration
from mipengine.common_data_elements import MetadataVariable
from mipengine.controller.algorithm_execution_DTOs import AlgorithmExecutionDTO
from mipengine.controller.algorithm_execution_DTOs import NodesTasksHandlersDTO
from mipengine.controller.algorithm_executor import AlgorithmExecutor
from mipengine.controller.api.algorithm_request_dto import AlgorithmInputDataDTO
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO
from mipengine.controller.node_tasks_handler_celery import NodeTasksHandlerCelery
from tests.dev_env_tests.nodes_communication import get_node_config_by_id


@pytest.fixture(scope="function")
def mock_cdes():
    common_data_elements = CommonDataElements()
    common_data_elements.pathologies = {
        "dementia": {
            "lefthippocampus": CommonDataElement(
                MetadataVariable(
                    code="lefthippocampus",
                    label="Left Hippocampus",
                    sql_type="real",
                    isCategorical=False,
                )
            ),
            "righthippocampus": CommonDataElement(
                MetadataVariable(
                    code="righthippocampus",
                    label="Right Hippocampus",
                    sql_type="real",
                    isCategorical=False,
                )
            ),
            "rightppplanumpolare": CommonDataElement(
                MetadataVariable(
                    code="rightppplanumpolare",
                    label="Right planum polare",
                    sql_type="real",
                    isCategorical=False,
                )
            ),
            "leftamygdala": CommonDataElement(
                MetadataVariable(
                    code="leftamygdala",
                    label="Left Amygdala",
                    sql_type="real",
                    isCategorical=False,
                )
            ),
            "rightamygdala": CommonDataElement(
                MetadataVariable(
                    code="rightamygdala",
                    label="Right Amygdala",
                    sql_type="real",
                    isCategorical=False,
                )
            ),
            "alzheimerbroadcategory": CommonDataElement(
                MetadataVariable(
                    code="alzheimerbroadcategory",
                    label="There will be two broad categories taken into account. Alzheimer s disease (AD) in which the diagnostic is 100% certain and <Other> comprising the rest of Alzheimer s related categories. The <Other> category refers to Alzheime s related diagnosis which origin can be traced to other pathology eg. vascular. In this category MCI diagnosis can also be found. In summary  all Alzheimer s related diagnosis that are not pure.",
                    sql_type="text",
                    isCategorical=True,
                    enumerations=[
                        MetadataEnumeration(
                            code="AD",
                            label="Alzheimer’s disease",
                        ),
                        MetadataEnumeration(
                            code="CN",
                            label="Cognitively Normal",
                        ),
                    ],
                )
            ),
        },
    }
    with patch(
        "mipengine.controller.algorithm_executor.controller_common_data_elements",
        common_data_elements,
    ):
        yield


@pytest.fixture(scope="function")
def mock_ctrl_config():
    ctrl_config = AttrDict(
        {
            "smpc": {
                "enabled": False,
                "optional": False,
            }
        }
    )

    with patch(
        "mipengine.controller.algorithm_executor.ctrl_config",
        ctrl_config,
    ):
        yield


@pytest.fixture(scope="function")
def mock_algorithms_modules():
    import mipengine

    mipengine.ALGORITHM_FOLDERS = "./mipengine/algorithms,./tests/algorithms"
    algorithm_modules = mipengine.import_algorithm_modules()

    with patch(
        "mipengine.controller.algorithm_executor.algorithm_modules",
        algorithm_modules,
    ):
        yield


def get_parametrization_list_success_cases():
    parametrization_list = []

    # ~~~~~~~~~~success case 1~~~~~~~~~~
    algo_execution_dto = AlgorithmExecutionDTO(
        context_id="123",
        algorithm_name="logistic_regression",
        algorithm_request_dto=AlgorithmRequestDTO(
            inputdata=AlgorithmInputDataDTO(
                pathology="dementia",
                datasets=["edsd"],
                filters={
                    "condition": "AND",
                    "rules": [
                        {
                            "id": "dataset",
                            "type": "string",
                            "value": ["edsd"],
                            "operator": "in",
                        },
                        {
                            "condition": "AND",
                            "rules": [
                                {
                                    "id": variable,
                                    "type": "string",
                                    "operator": "is_not_null",
                                    "value": None,
                                }
                                for variable in [
                                    "lefthippocampus",
                                    "righthippocampus",
                                    "rightppplanumpolare",
                                    "leftamygdala",
                                    "rightamygdala",
                                    "alzheimerbroadcategory",
                                ]
                            ],
                        },
                    ],
                    "valid": True,
                },
                x=[
                    "lefthippocampus",
                    "righthippocampus",
                    "rightppplanumpolare",
                    "leftamygdala",
                    "rightamygdala",
                ],
                y=["alzheimerbroadcategory"],
            ),
            parameters={"classes": ["AD", "CN"]},
        ),
    )
    parametrization_list.append(algo_execution_dto)
    # END ~~~~~~~~~~success case 1~~~~~~~~~~

    # ~~~~~~~~~~success case 2~~~~~~~~~~
    algo_execution_dto = AlgorithmExecutionDTO(
        context_id="1234",
        algorithm_name="smpc_standard_deviation",
        algorithm_request_dto=AlgorithmRequestDTO(
            inputdata=AlgorithmInputDataDTO(
                pathology="dementia",
                datasets=["edsd"],
                filters={
                    "condition": "AND",
                    "rules": [
                        {
                            "id": "dataset",
                            "type": "string",
                            "value": ["edsd"],
                            "operator": "in",
                        },
                        {
                            "condition": "AND",
                            "rules": [
                                {
                                    "id": variable,
                                    "type": "string",
                                    "operator": "is_not_null",
                                    "value": None,
                                }
                                for variable in [
                                    "lefthippocampus",
                                ]
                            ],
                        },
                    ],
                    "valid": True,
                },
                x=[
                    "lefthippocampus",
                ],
            ),
            parameters={"classes": ["AD", "CN"]},
            flags={"smpc": False},
        ),
    )
    parametrization_list.append(algo_execution_dto)
    # END ~~~~~~~~~~success case 2~~~~~~~~~~
    return parametrization_list


@pytest.mark.parametrize(
    "algo_execution_dto",
    get_parametrization_list_success_cases(),
)
def test_single_local_node_algorithm_execution(
    mock_cdes, mock_ctrl_config, mock_algorithms_modules, algo_execution_dto
):
    local_node_id = "localnode1"

    node_config = get_node_config_by_id(local_node_id)
    queue_addr = str(node_config.rabbitmq.ip) + ":" + str(node_config.rabbitmq.port)
    db_addr = str(node_config.monetdb.ip) + ":" + str(node_config.monetdb.port)
    local_node_task_handler = NodeTasksHandlerCelery(
        node_id=local_node_id,
        node_queue_addr=queue_addr,
        node_db_addr=db_addr,
        tasks_timeout=45,
    )

    single_node_task_handler = NodesTasksHandlersDTO(
        global_node_tasks_handler=local_node_task_handler,
        local_nodes_tasks_handlers=[local_node_task_handler],
    )
    algo_executor = AlgorithmExecutor(
        algorithm_execution_dto=algo_execution_dto,
        nodes_tasks_handlers_dto=single_node_task_handler,
    )
    result = algo_executor.run()

    assert isinstance(result, TabularDataResult)
