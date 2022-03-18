import asyncio
from os import path
from unittest.mock import patch

import pytest
import toml

from mipengine import AttrDict
from mipengine.controller.controller import Controller
from tests.standalone_tests.conftest import ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
from tests.standalone_tests.conftest import LOCALNODETMP_CONFIG_FILE
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODETMP_NAME
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODETMP_PORT
from tests.standalone_tests.conftest import TEST_ENV_CONFIG_FOLDER
from tests.standalone_tests.conftest import _create_node_service
from tests.standalone_tests.conftest import _create_rabbitmq_container
from tests.standalone_tests.conftest import _load_data_monetdb_container
from tests.standalone_tests.conftest import _remove_data_model_from_localnodetmp_monetdb
from tests.standalone_tests.conftest import _remove_data_model_from_monetdb_container
from tests.standalone_tests.conftest import kill_node_service
from tests.standalone_tests.conftest import remove_localnodetmp_rabbitmq

MAX_RETRIES = 30


@pytest.fixture(scope="session")
def controller_config_mock():
    controller_config = AttrDict(
        {
            "log_level": "DEBUG",
            "framework_log_level": "INFO",
            "deployment_type": "LOCAL",
            "node_landscape_aggregator_update_interval": 2,  # 5,
            "nodes_cleanup_interval": 2,
            "localnodes": {
                "config_file": "./tests/standalone_tests/testing_env_configs/test_localnodes_addresses.json",
                "dns": "",
                "port": "",
            },
            "rabbitmq": {
                "user": "user",
                "password": "password",
                "vhost": "user_vhost",
                "celery_tasks_timeout": 30,  # 60,
                "celery_tasks_max_retries": 3,
                "celery_tasks_interval_start": 0,
                "celery_tasks_interval_step": 0.2,
                "celery_tasks_interval_max": 0.5,
            },
            "smpc": {
                "enabled": False,
                "optional": False,
                "coordinator_address": "$SMPC_COORDINATOR_ADDRESS",
            },
        }
    )
    return controller_config


@pytest.fixture(autouse=True, scope="session")
def patch_controller(controller_config_mock):
    with patch(
        "mipengine.controller.controller.controller_config",
        controller_config_mock,
    ):
        yield


@pytest.fixture(autouse=True, scope="session")
def patch_node_landscape_aggregator(controller_config_mock):
    with patch(
        "mipengine.controller.node_landscape_aggregator.controller_config",
        controller_config_mock,
    ), patch(
        "mipengine.controller.node_landscape_aggregator.NODE_LANDSCAPE_AGGREGATOR_UPDATE_INTERVAL",
        controller_config_mock.node_landscape_aggregator_update_interval,
    ), patch(
        "mipengine.controller.node_landscape_aggregator.CELERY_TASKS_TIMEOUT",
        controller_config_mock.rabbitmq.celery_tasks_timeout,
    ):
        yield


@pytest.fixture(autouse=True, scope="session")
def patch_celery_app(controller_config_mock):
    with patch(
        "mipengine.controller.celery_app.controller_config", controller_config_mock
    ):
        yield


@pytest.fixture(autouse=True, scope="session")
def patch_algorithm_executor(controller_config_mock):
    with patch(
        "mipengine.controller.algorithm_executor.ctrl_config", controller_config_mock
    ):
        yield


@pytest.fixture(autouse=True, scope="session")
def patch_node_address(controller_config_mock):
    with patch(
        "mipengine.controller.node_address.controller_config", controller_config_mock
    ):
        yield


@pytest.mark.slow
@pytest.mark.asyncio
async def test_update_loop_node_service_down(
    load_data_localnodetmp,
    globalnode_node_service,
    localnodetmp_node_service,
):

    # get tmp localnode node_id from config file
    localnodetmp_node_id = get_localnodetmp_node_id()
    controller = Controller()

    # wait until node registry gets the nodes info
    await controller.start_node_landscape_aggregator()

    for _ in range(MAX_RETRIES):
        if (
            localnodetmp_node_id in controller.get_all_local_nodes()
            and controller.get_cdes_per_data_model()
        ):
            cdes_per_data_model = controller.get_cdes_per_data_model()
            assert (
                "tbi:0.1" in cdes_per_data_model
                and "dementia:0.1" in cdes_per_data_model
            )
            assert (
                len(cdes_per_data_model["tbi:0.1"].values) == 21
                and len(cdes_per_data_model["dementia:0.1"].values) == 186
            )
            break
        await asyncio.sleep(2)
    else:
        pytest.fail(
            "Exceeded max retries while waiting for the node registry to contain the tmplocalnode"
        )

    kill_node_service(localnodetmp_node_service)

    # wait until node registry removes tmplocalnode
    for _ in range(MAX_RETRIES):
        if (
            localnodetmp_node_id not in controller.get_all_local_nodes()
            and not controller.get_cdes_per_data_model()
        ):
            break
        await asyncio.sleep(2)
    else:
        pytest.fail(
            "Exceeded max retries while waiting for the node registry to remove the tmplocalnode"
        )

    # restart tmplocalnode node service (the celery app)
    localnodetmp_node_service_proc = start_localnodetmp_node_service()

    # wait until node registry re-added tmplocalnode
    for _ in range(MAX_RETRIES):
        if (
            localnodetmp_node_id in controller.get_all_local_nodes()
            and controller.get_cdes_per_data_model()
        ):
            cdes_per_data_model = controller.get_cdes_per_data_model()
            assert (
                "tbi:0.1" in cdes_per_data_model
                and "dementia:0.1" in cdes_per_data_model
            )
            assert (
                len(cdes_per_data_model["tbi:0.1"].values) == 21
                and len(cdes_per_data_model["dementia:0.1"].values) == 186
            )
            break
        await asyncio.sleep(2)
    else:
        pytest.fail(
            "Exceeded max retries while waiting for the node registry to re-add the tmplocalnode"
        )

    await controller.stop_node_landscape_aggregator()

    # the node service was started in here, so it must manually be killed, otherwise it is
    # alive through the whole pytest session and is erroneously accessed by other tests
    # where the node service is supposedly down
    kill_node_service(localnodetmp_node_service_proc)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_update_loop_rabbitmq_down(
    load_data_localnodetmp,
    globalnode_node_service,
    localnodetmp_node_service,
):

    # get tmp localnode node_id from config file
    localnodetmp_node_id = get_localnodetmp_node_id()
    controller = Controller()

    # wait until node registry gets the nodes info
    await controller.start_node_landscape_aggregator()

    # wait until node registry and data model registry is up-to-date
    for _ in range(MAX_RETRIES):
        if (
            localnodetmp_node_id in controller.get_all_local_nodes()
            and controller.get_cdes_per_data_model()
        ):
            cdes_per_data_model = controller.get_cdes_per_data_model()
            assert (
                "tbi:0.1" in cdes_per_data_model
                and "dementia:0.1" in cdes_per_data_model
            )
            assert (
                len(cdes_per_data_model["tbi:0.1"].values) == 21
                and len(cdes_per_data_model["dementia:0.1"].values) == 186
            )
            break
        await asyncio.sleep(2)
    else:
        pytest.fail(
            "Exceeded max retries while waiting for the node registry to contain the tmplocalnode"
        )

    remove_localnodetmp_rabbitmq()

    # wait until node registry no longer contains tmplocalnode
    for _ in range(MAX_RETRIES):
        if (
            localnodetmp_node_id not in controller.get_all_local_nodes()
            and not controller.get_cdes_per_data_model()
        ):
            break
        await asyncio.sleep(2)
    else:
        pytest.fail(
            "Exceeded max retries while waiting for the node registry to remove the tmplocalnode"
        )

    # restart tmplocalnode rabbitmq container
    _create_rabbitmq_container(RABBITMQ_LOCALNODETMP_NAME, RABBITMQ_LOCALNODETMP_PORT)

    # wait until node registry contains tmplocalnode
    for _ in range(MAX_RETRIES):
        if (
            localnodetmp_node_id in controller.get_all_local_nodes()
            and controller.get_cdes_per_data_model()
        ):
            cdes_per_data_model = controller.get_cdes_per_data_model()
            assert (
                "tbi:0.1" in cdes_per_data_model
                and "dementia:0.1" in cdes_per_data_model
            )
            assert (
                len(cdes_per_data_model["tbi:0.1"].values) == 21
                and len(cdes_per_data_model["dementia:0.1"].values) == 186
            )
            break
        await asyncio.sleep(2)
    else:
        pytest.fail(
            "Exceeded max retries while waiting for the node registry to re-add the tmplocalnode"
        )

    await controller.stop_node_landscape_aggregator()

    # the node service was started in here, so it must manually be killed, otherwise it is
    # alive through the whole pytest session and is erroneously accessed by other tests
    # where the node service is supposedly down
    kill_node_service(localnodetmp_node_service)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_update_loop_data_models_removed(
    load_data_localnodetmp,
    globalnode_node_service,
    localnodetmp_node_service,
):

    # get tmp localnode node_id from config file
    localnodetmp_node_id = get_localnodetmp_node_id()
    controller = Controller()

    # wait until node registry gets the nodes info
    await controller.start_node_landscape_aggregator()

    # wait until node registry and data model registry is up-to-date
    for _ in range(MAX_RETRIES):
        if (
            localnodetmp_node_id in controller.get_all_local_nodes()
            and controller.get_cdes_per_data_model()
        ):
            cdes_per_data_model = controller.get_cdes_per_data_model()
            assert (
                "tbi:0.1" in cdes_per_data_model
                and "dementia:0.1" in cdes_per_data_model
            )
            assert (
                len(cdes_per_data_model["tbi:0.1"].values) == 21
                and len(cdes_per_data_model["dementia:0.1"].values) == 186
            )
            break
        await asyncio.sleep(2)
    else:
        pytest.fail(
            "Exceeded max retries while waiting for the node registry to contain the tmplocalnode"
        )

    remove_data_model_from_localnodetmp_monetdb("dementia:0.1")
    remove_data_model_from_localnodetmp_monetdb("dementia:0.1")

    # wait until data model registry no longer contains 'dementia:0.1'
    for _ in range(MAX_RETRIES):
        if (
            localnodetmp_node_id in controller.get_all_local_nodes()
            and controller.get_cdes_per_data_model()
            and "dementia:0.1" not in controller.get_cdes_per_data_model()
        ):
            cdes_per_data_model = controller.get_cdes_per_data_model()
            assert "tbi:0.1" in cdes_per_data_model
            assert len(cdes_per_data_model["tbi:0.1"].values) == 21
            break
        await asyncio.sleep(2)
    else:
        pytest.fail(
            "Exceeded max retries while waiting for the data model registry to be updated and not contain 'dementia:0.1'"
        )

    remove_data_model_from_localnodetmp_monetdb("tbi:0.1")

    # wait until data model registry no longer contains 'tbi:0.1'
    for _ in range(MAX_RETRIES):
        if (
            localnodetmp_node_id in controller.get_all_local_nodes()
            and not controller.get_cdes_per_data_model()
        ):
            break
        await asyncio.sleep(2)
    else:
        pytest.fail(
            "Exceeded max retries while waiting for the data model registry to be updated and not contain 'tbi:0.1'"
        )

    local_nodes = controller.get_all_local_nodes()
    _load_data_monetdb_container(
        db_ip=local_nodes[localnodetmp_node_id].db_ip,
        db_port=local_nodes[localnodetmp_node_id].db_port,
    )

    # wait until data models are re-loaded in the data model registry
    for _ in range(MAX_RETRIES):
        if (
            localnodetmp_node_id in controller.get_all_local_nodes()
            and controller.get_cdes_per_data_model()
        ):
            cdes_per_data_model = controller.get_cdes_per_data_model()
            assert (
                "tbi:0.1" in cdes_per_data_model
                and "dementia:0.1" in cdes_per_data_model
            )
            assert (
                len(cdes_per_data_model["tbi:0.1"].values) == 21
                and len(cdes_per_data_model["dementia:0.1"].values) == 186
            )
            break
        await asyncio.sleep(2)
    else:
        pytest.fail(
            "Exceeded max retries while waiting for the node registry to contain the tmplocalnode"
        )

    await controller.stop_node_landscape_aggregator()

    # the node service was started in here, so it must manually be killed, otherwise it is
    # alive through the whole pytest session and is erroneously accessed by other tests
    # where the node service is supposedly down
    kill_node_service(localnodetmp_node_service)


def remove_data_model_from_localnodetmp_monetdb(data_model):
    data_model_code, data_model_version = data_model.split(":")
    _remove_data_model_from_localnodetmp_monetdb(
        data_model_code=data_model_code,
        data_model_version=data_model_version,
    )


def get_localnodetmp_node_id():
    local_node_filepath = path.join(TEST_ENV_CONFIG_FOLDER, LOCALNODETMP_CONFIG_FILE)
    with open(local_node_filepath) as fp:
        tmp = toml.load(fp)
        node_id = tmp["identifier"]
    return node_id


def start_localnodetmp_node_service():
    node_config_file = LOCALNODETMP_CONFIG_FILE
    algo_folders_env_variable_val = ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, node_config_file)
    proc = _create_node_service(algo_folders_env_variable_val, node_config_filepath)
    return proc
