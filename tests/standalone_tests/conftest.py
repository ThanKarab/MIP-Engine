import os
import re
import subprocess
import time
from os import path
from pathlib import Path

import docker
import psutil
import pytest
import sqlalchemy as sql
import toml

from mipengine.controller.controller_logger import get_request_logger
from mipengine.controller.data_model_registry import DataModelRegistry
from mipengine.controller.node_landscape_aggregator import NodeLandscapeAggregator
from mipengine.controller.node_registry import NodeRegistry
from mipengine.controller.node_tasks_handler_celery import NodeTasksHandlerCelery
from mipengine.udfgen import udfio

ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE = "./mipengine/algorithms,./tests/algorithms"
TESTING_RABBITMQ_CONT_IMAGE = "madgik/mipengine_rabbitmq:latest"
TESTING_MONETDB_CONT_IMAGE = "madgik/mipenginedb:latest"

this_mod_path = os.path.dirname(os.path.abspath(__file__))
TEST_ENV_CONFIG_FOLDER = path.join(this_mod_path, "testing_env_configs")
TEST_DATA_FOLDER = Path(this_mod_path).parent / "test_data"

OUTDIR = Path("/tmp/mipengine/")
if not OUTDIR.exists():
    OUTDIR.mkdir()

COMMON_IP = "172.17.0.1"
RABBITMQ_GLOBALNODE_NAME = "rabbitmq_test_globalnode"
RABBITMQ_LOCALNODE1_NAME = "rabbitmq_test_localnode1"
RABBITMQ_LOCALNODE2_NAME = "rabbitmq_test_localnode2"

RABBITMQ_LOCALNODETMP_NAME = "rabbitmq_test_localnodetmp"
RABBITMQ_SMPC_GLOBALNODE_NAME = "rabbitmq_test_smpc_globalnode"
RABBITMQ_SMPC_LOCALNODE1_NAME = "rabbitmq_test_smpc_localnode1"
RABBITMQ_SMPC_LOCALNODE2_NAME = "rabbitmq_test_smpc_localnode2"

RABBITMQ_GLOBALNODE_PORT = 60000
RABBITMQ_LOCALNODE1_PORT = 60001
RABBITMQ_LOCALNODE2_PORT = 60002
RABBITMQ_LOCALNODETMP_PORT = 60003
RABBITMQ_SMPC_GLOBALNODE_PORT = 60004
RABBITMQ_SMPC_LOCALNODE1_PORT = 60005
RABBITMQ_SMPC_LOCALNODE2_PORT = 60006

MONETDB_GLOBALNODE_NAME = "monetdb_test_globalnode"
MONETDB_LOCALNODE1_NAME = "monetdb_test_localnode1"
MONETDB_LOCALNODE2_NAME = "monetdb_test_localnode2"
MONETDB_LOCALNODETMP_NAME = "monetdb_test_localnodetmp"
MONETDB_SMPC_GLOBALNODE_NAME = "monetdb_test_smpc_globalnode"
MONETDB_SMPC_LOCALNODE1_NAME = "monetdb_test_smpc_localnode1"
MONETDB_SMPC_LOCALNODE2_NAME = "monetdb_test_smpc_localnode2"
MONETDB_GLOBALNODE_PORT = 61000
MONETDB_LOCALNODE1_PORT = 61001
MONETDB_LOCALNODE2_PORT = 61002
MONETDB_LOCALNODETMP_PORT = 61003
MONETDB_SMPC_GLOBALNODE_PORT = 61004
MONETDB_SMPC_LOCALNODE1_PORT = 61005
MONETDB_SMPC_LOCALNODE2_PORT = 61006

GLOBALNODE_CONFIG_FILE = "testglobalnode.toml"
LOCALNODE1_CONFIG_FILE = "testlocalnode1.toml"
LOCALNODE2_CONFIG_FILE = "testlocalnode2.toml"
LOCALNODETMP_CONFIG_FILE = "testlocalnodetmp.toml"
GLOBALNODE_SMPC_CONFIG_FILE = "smpc_globalnode.toml"
LOCALNODE1_SMPC_CONFIG_FILE = "smpc_localnode1.toml"
LOCALNODE2_SMPC_CONFIG_FILE = "smpc_localnode2.toml"

TASKS_TIMEOUT = 10
RUN_UDF_TASK_TIMEOUT = 120

########### SMPC Cluster ############
SMPC_CLUSTER_IMAGE = "gpikra/coordinator:latest"
SMPC_COORD_DB_IMAGE = "gpikra/mongodb:latest"
SMPC_COORD_QUEUE_IMAGE = "redis:alpine3.15"

SMPC_COORD_CONT_NAME = "smpc_test_coordinator"
SMPC_COORD_DB_CONT_NAME = "smpc_test_coordinator_db"
SMPC_COORD_QUEUE_CONT_NAME = "smpc_test_coordinator_queue"
SMPC_PLAYER1_CONT_NAME = "smpc_test_player1"
SMPC_PLAYER2_CONT_NAME = "smpc_test_player2"
SMPC_PLAYER3_CONT_NAME = "smpc_test_player3"
SMPC_CLIENT1_CONT_NAME = "smpc_test_client1"
SMPC_CLIENT2_CONT_NAME = "smpc_test_client2"

SMPC_COORD_PORT = 12314
SMPC_COORD_DB_PORT = 27017
SMPC_COORD_QUEUE_PORT = 6379
SMPC_PLAYER1_PORT1 = 5000
SMPC_PLAYER1_PORT2 = 7000
SMPC_PLAYER1_PORT3 = 14000
SMPC_PLAYER2_PORT1 = 5001
SMPC_PLAYER2_PORT2 = 7001
SMPC_PLAYER2_PORT3 = 14001
SMPC_PLAYER3_PORT1 = 5002
SMPC_PLAYER3_PORT2 = 7002
SMPC_PLAYER3_PORT3 = 14002
SMPC_CLIENT1_PORT = 9000
SMPC_CLIENT2_PORT = 9001
#####################################


# TODO Instead of the fixtures having scope session, it could be function,
# but when the fixture start, it should check if it already exists, thus
# not creating it again (fast). This could solve the problem of some
# tests destroying some containers to test things.


class MonetDBSetupError(Exception):
    """Raised when the MonetDB container is unable to start."""


def _create_monetdb_container(cont_name, cont_port):
    print(f"\nCreating monetdb container '{cont_name}' at port {cont_port}...")
    client = docker.from_env()
    container_names = [container.name for container in client.containers.list(all=True)]
    if cont_name not in container_names:
        # A volume is used to pass the udfio inside the monetdb container.
        # This is done so that we don't need to rebuild every time the udfio.py file is changed.
        udfio_full_path = path.abspath(udfio.__file__)
        container = client.containers.run(
            TESTING_MONETDB_CONT_IMAGE,
            detach=True,
            ports={"50000/tcp": cont_port},
            volumes=[f"{udfio_full_path}:/home/udflib/udfio.py"],
            name=cont_name,
            publish_all_ports=True,
        )
    else:
        container = client.containers.get(cont_name)
        # After a machine restart the container exists, but it is stopped. (Used only in development)
        if container.status == "exited":
            container.start()

    # The time needed to start a monetdb container varies considerably. We need
    # to wait until a phrase appears in the logs to avoid starting the tests
    # too soon. The process is abandoned after 100 tries (50 sec).
    for _ in range(100):
        if b"new database mapi:monetdb" in container.logs():
            break
        time.sleep(0.5)
    else:
        raise MonetDBSetupError
    print(f"Monetdb container '{cont_name}' started.")


def _remove_monetdb_container(cont_name):
    print(f"\nRemoving monetdb container '{cont_name}'.")
    client = docker.from_env()
    container = client.containers.get(cont_name)
    container.remove(v=True, force=True)
    print(f"Removed monetdb container '{cont_name}'.")


@pytest.fixture(scope="session")
def monetdb_globalnode():
    cont_name = MONETDB_GLOBALNODE_NAME
    cont_port = MONETDB_GLOBALNODE_PORT
    _create_monetdb_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_monetdb_container(cont_name)


@pytest.fixture(scope="session")
def monetdb_localnode1():
    cont_name = MONETDB_LOCALNODE1_NAME
    cont_port = MONETDB_LOCALNODE1_PORT
    _create_monetdb_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_monetdb_container(cont_name)


@pytest.fixture(scope="session")
def monetdb_localnode2():
    cont_name = MONETDB_LOCALNODE2_NAME
    cont_port = MONETDB_LOCALNODE2_PORT
    _create_monetdb_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_monetdb_container(cont_name)


@pytest.fixture(scope="session")
def monetdb_smpc_globalnode():
    cont_name = MONETDB_SMPC_GLOBALNODE_NAME
    cont_port = MONETDB_SMPC_GLOBALNODE_PORT
    _create_monetdb_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_monetdb_container(cont_name)


@pytest.fixture(scope="session")
def monetdb_smpc_localnode1():
    cont_name = MONETDB_SMPC_LOCALNODE1_NAME
    cont_port = MONETDB_SMPC_LOCALNODE1_PORT
    _create_monetdb_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_monetdb_container(cont_name)


@pytest.fixture(scope="session")
def monetdb_smpc_localnode2():
    cont_name = MONETDB_SMPC_LOCALNODE2_NAME
    cont_port = MONETDB_SMPC_LOCALNODE2_PORT
    _create_monetdb_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_monetdb_container(cont_name)


@pytest.fixture(scope="function")
def monetdb_localnodetmp():
    cont_name = MONETDB_LOCALNODETMP_NAME
    cont_port = MONETDB_LOCALNODETMP_PORT
    _create_monetdb_container(cont_name, cont_port)
    yield
    _remove_monetdb_container(cont_name)


def _init_database_monetdb_container(db_ip, db_port):
    # Check if the database is already initialized
    cmd = f"mipdb list-data-models --ip {db_ip} --port {db_port} "
    res = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if res.returncode == 0:
        print(f"\nDatabase ({db_ip}:{db_port}) already initialized, continuing.")
        return

    print(f"\nInitializing database ({db_ip}:{db_port})")
    cmd = f"mipdb init --ip {db_ip} --port {db_port} "
    subprocess.run(
        cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    print(f"\nDatabase ({db_ip}:{db_port}) initialized.")


def _load_data_monetdb_container(db_ip, db_port):
    # Check if the database is already loaded
    cmd = f"mipdb list-datasets --ip {db_ip} --port {db_port} "
    res = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if "There are no datasets" not in str(res.stdout):
        print(f"\nDatabase ({db_ip}:{db_port}) already loaded, continuing.")
        return

    print(f"\nLoading data to database ({db_ip}:{db_port})")
    cmd = f"mipdb load-folder {TEST_DATA_FOLDER}  --ip {db_ip} --port {db_port} "
    subprocess.run(
        cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    print(f"\nData loaded to database ({db_ip}:{db_port})")


def _remove_data_model_from_localnodetmp_monetdb(data_model_code, data_model_version):
    # Remove data_model
    cmd = f"mipdb delete-data-model {data_model_code} -v {data_model_version} -f  --ip {COMMON_IP} --port {MONETDB_LOCALNODETMP_PORT} "
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)


@pytest.fixture(scope="session")
def init_data_globalnode(monetdb_globalnode):
    _init_database_monetdb_container(COMMON_IP, MONETDB_GLOBALNODE_PORT)
    yield


@pytest.fixture(scope="session")
def load_data_localnode1(monetdb_localnode1):
    _init_database_monetdb_container(COMMON_IP, MONETDB_LOCALNODE1_PORT)
    _load_data_monetdb_container(COMMON_IP, MONETDB_LOCALNODE1_PORT)
    yield


@pytest.fixture(scope="session")
def load_data_localnode2(monetdb_localnode2):
    _init_database_monetdb_container(COMMON_IP, MONETDB_LOCALNODE2_PORT)
    _load_data_monetdb_container(COMMON_IP, MONETDB_LOCALNODE2_PORT)
    yield


@pytest.fixture(scope="function")
def load_data_localnodetmp(monetdb_localnodetmp):
    _init_database_monetdb_container(COMMON_IP, MONETDB_LOCALNODETMP_PORT)
    _load_data_monetdb_container(COMMON_IP, MONETDB_LOCALNODETMP_PORT)
    yield


def _create_db_cursor(db_port):
    class MonetDBTesting:
        """MonetDB class used for testing."""

        def __init__(self) -> None:
            username = "monetdb"
            password = "monetdb"
            port = db_port
            dbfarm = "db"
            url = f"monetdb://{username}:{password}@{COMMON_IP}:{port}/{dbfarm}:"
            self._executor = sql.create_engine(url, echo=True)

        def execute(self, query, *args, **kwargs) -> list:
            return self._executor.execute(query, *args, **kwargs)

    return MonetDBTesting()


@pytest.fixture(scope="session")
def globalnode_db_cursor():
    return _create_db_cursor(MONETDB_GLOBALNODE_PORT)


@pytest.fixture(scope="session")
def localnode1_db_cursor():
    return _create_db_cursor(MONETDB_LOCALNODE1_PORT)


@pytest.fixture(scope="session")
def localnode2_db_cursor():
    return _create_db_cursor(MONETDB_LOCALNODE2_PORT)


@pytest.fixture(scope="session")
def globalnode_smpc_db_cursor():
    return _create_db_cursor(MONETDB_SMPC_GLOBALNODE_PORT)


@pytest.fixture(scope="session")
def localnode1_smpc_db_cursor():
    return _create_db_cursor(MONETDB_SMPC_LOCALNODE1_PORT)


@pytest.fixture(scope="session")
def localnode2_smpc_db_cursor():
    return _create_db_cursor(MONETDB_SMPC_LOCALNODE2_PORT)


@pytest.fixture(scope="function")
def localnodetmp_db_cursor():
    return _create_db_cursor(MONETDB_LOCALNODETMP_PORT)


def _clean_db(cursor):
    table_types = {0, 1, 3, 5}  # 0=table, 1=view, 3=merge_table, 5=remote_table

    for table_type in table_types:
        select_user_tables = f"SELECT name FROM sys.tables WHERE system=FALSE AND schema_id=2000 AND type={table_type}"
        user_tables = cursor.execute(select_user_tables).fetchall()
        for table_name, *_ in user_tables:
            if table_type == 1:  # view
                cursor.execute(f"DROP VIEW {table_name}")
            else:
                cursor.execute(f"DROP TABLE {table_name}")


@pytest.fixture(scope="function")
def clean_globalnode_db(globalnode_db_cursor):
    yield
    _clean_db(globalnode_db_cursor)


@pytest.fixture(scope="function")
def clean_localnode1_db(localnode1_db_cursor):
    yield
    _clean_db(localnode1_db_cursor)


@pytest.fixture(scope="function")
def clean_smpc_globalnode_db(globalnode_smpc_db_cursor):
    yield
    _clean_db(globalnode_smpc_db_cursor)


@pytest.fixture(scope="function")
def clean_smpc_localnode1_db(localnode1_smpc_db_cursor):
    yield
    _clean_db(localnode1_smpc_db_cursor)


@pytest.fixture(scope="function")
def clean_smpc_localnode2_db(localnode2_smpc_db_cursor):
    yield
    _clean_db(localnode2_smpc_db_cursor)


@pytest.fixture(scope="function")
def clean_localnode2_db(localnode2_db_cursor):
    yield
    _clean_db(localnode2_db_cursor)


@pytest.fixture(scope="function")
def use_globalnode_database(monetdb_globalnode, clean_globalnode_db):
    pass


@pytest.fixture(scope="function")
def use_localnode1_database(monetdb_localnode1, clean_localnode1_db):
    pass


@pytest.fixture(scope="function")
def use_localnode2_database(monetdb_localnode2, clean_localnode2_db):
    pass


@pytest.fixture(scope="function")
def use_smpc_globalnode_database(monetdb_smpc_globalnode, clean_smpc_globalnode_db):
    pass


@pytest.fixture(scope="function")
def use_smpc_localnode1_database(monetdb_smpc_localnode1, clean_smpc_localnode1_db):
    pass


@pytest.fixture(scope="function")
def use_smpc_localnode2_database(monetdb_smpc_localnode2, clean_smpc_localnode2_db):
    pass


def _create_rabbitmq_container(cont_name, cont_port):
    print(f"\nCreating rabbitmq container '{cont_name}' at port {cont_port}...")
    client = docker.from_env()
    container_names = [container.name for container in client.containers.list(all=True)]
    if cont_name not in container_names:
        container = client.containers.run(
            TESTING_RABBITMQ_CONT_IMAGE,
            detach=True,
            ports={"5672/tcp": cont_port},
            name=cont_name,
        )
    else:
        container = client.containers.get(cont_name)
        # After a machine restart the container exists, but it is stopped. (Used only in development)
        if container.status == "exited":
            container.start()

    while (
        "Health" not in container.attrs["State"]
        or container.attrs["State"]["Health"]["Status"] != "healthy"
    ):
        container.reload()  # attributes are cached, this refreshes them..
        time.sleep(1)

    print(f"Rabbitmq container '{cont_name}' started.")


def _remove_rabbitmq_container(cont_name):
    print(f"\nRemoving rabbitmq container '{cont_name}'.")
    try:
        client = docker.from_env()
        container = client.containers.get(cont_name)
        container.remove(v=True, force=True)
    except docker.errors.NotFound:
        pass  # container was removed by other means...
    print(f"Removed rabbitmq container '{cont_name}'.")


@pytest.fixture(scope="session")
def rabbitmq_globalnode():
    cont_name = RABBITMQ_GLOBALNODE_NAME
    cont_port = RABBITMQ_GLOBALNODE_PORT
    _create_rabbitmq_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_rabbitmq_container(cont_name)


@pytest.fixture(scope="session")
def rabbitmq_localnode1():
    cont_name = RABBITMQ_LOCALNODE1_NAME
    cont_port = RABBITMQ_LOCALNODE1_PORT
    _create_rabbitmq_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_rabbitmq_container(cont_name)


@pytest.fixture(scope="session")
def rabbitmq_localnode2():
    cont_name = RABBITMQ_LOCALNODE2_NAME
    cont_port = RABBITMQ_LOCALNODE2_PORT
    _create_rabbitmq_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_rabbitmq_container(cont_name)


@pytest.fixture(scope="session")
def rabbitmq_smpc_globalnode():
    cont_name = RABBITMQ_SMPC_GLOBALNODE_NAME
    cont_port = RABBITMQ_SMPC_GLOBALNODE_PORT
    _create_rabbitmq_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_rabbitmq_container(cont_name)


@pytest.fixture(scope="session")
def rabbitmq_smpc_localnode1():
    cont_name = RABBITMQ_SMPC_LOCALNODE1_NAME
    cont_port = RABBITMQ_SMPC_LOCALNODE1_PORT
    _create_rabbitmq_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_rabbitmq_container(cont_name)


@pytest.fixture(scope="session")
def rabbitmq_smpc_localnode2():
    cont_name = RABBITMQ_SMPC_LOCALNODE2_NAME
    cont_port = RABBITMQ_SMPC_LOCALNODE2_PORT
    _create_rabbitmq_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_rabbitmq_container(cont_name)


@pytest.fixture(scope="function")
def rabbitmq_localnodetmp():
    cont_name = RABBITMQ_LOCALNODETMP_NAME
    cont_port = RABBITMQ_LOCALNODETMP_PORT
    _create_rabbitmq_container(cont_name, cont_port)
    yield
    _remove_rabbitmq_container(cont_name)


def remove_localnodetmp_rabbitmq():
    cont_name = RABBITMQ_LOCALNODETMP_NAME
    _remove_rabbitmq_container(cont_name)


def _create_node_service(algo_folders_env_variable_val, node_config_filepath):
    with open(node_config_filepath) as fp:
        tmp = toml.load(fp)
        node_id = tmp["identifier"]

    print(f"\nCreating node service with id '{node_id}'...")

    logpath = OUTDIR / (node_id + ".out")
    if os.path.isfile(logpath):
        os.remove(logpath)

    env = os.environ.copy()
    env["ALGORITHM_FOLDERS"] = algo_folders_env_variable_val
    env["MIPENGINE_NODE_CONFIG_FILE"] = node_config_filepath

    cmd = f"poetry run celery -A mipengine.node.node worker -l DEBUG >> {logpath} --purge 2>&1 "

    # if executed without "exec" it is spawned as a child process of the shell, so it is difficult to kill it
    # https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true
    proc = subprocess.Popen(
        "exec " + cmd,
        shell=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=env,
    )

    # Check that celery started
    for _ in range(100):
        try:
            with open(logpath) as logfile:
                if bool(
                    re.search("CELERY - FRAMEWORK - celery@.* ready.", logfile.read())
                ):
                    break
        except FileNotFoundError:
            pass
        time.sleep(0.5)
    else:
        with open(logpath) as logfile:
            raise TimeoutError(
                f"The node service '{node_id}' didn't manage to start in the designated time. Logs: \n{logfile.read()}"
            )

    print(f"Created node service with id '{node_id}' and process id '{proc.pid}'.")
    return proc


def kill_node_service(proc):
    print(f"\nKilling node service with process id '{proc.pid}'...")
    psutil_proc = psutil.Process(proc.pid)
    proc.kill()
    for _ in range(100):
        if psutil_proc.status() == "zombie" or psutil_proc.status() == "sleeping":
            break
        time.sleep(0.1)
    else:
        raise TimeoutError(
            f"Node service is still running, status: '{psutil_proc.status()}'."
        )
    print(f"Killed node service with process id '{proc.pid}'.")


@pytest.fixture(scope="session")
def globalnode_node_service(rabbitmq_globalnode, monetdb_globalnode):
    node_config_file = GLOBALNODE_CONFIG_FILE
    algo_folders_env_variable_val = ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, node_config_file)
    proc = _create_node_service(algo_folders_env_variable_val, node_config_filepath)
    yield
    kill_node_service(proc)


@pytest.fixture(scope="session")
def localnode1_node_service(rabbitmq_localnode1, monetdb_localnode1):
    node_config_file = LOCALNODE1_CONFIG_FILE
    algo_folders_env_variable_val = ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, node_config_file)
    proc = _create_node_service(algo_folders_env_variable_val, node_config_filepath)
    yield
    kill_node_service(proc)


@pytest.fixture(scope="session")
def localnode2_node_service(rabbitmq_localnode2, monetdb_localnode2):
    node_config_file = LOCALNODE2_CONFIG_FILE
    algo_folders_env_variable_val = ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, node_config_file)
    proc = _create_node_service(algo_folders_env_variable_val, node_config_filepath)
    yield
    kill_node_service(proc)


@pytest.fixture(scope="session")
def smpc_globalnode_node_service(rabbitmq_smpc_globalnode, monetdb_smpc_globalnode):
    node_config_file = GLOBALNODE_SMPC_CONFIG_FILE
    algo_folders_env_variable_val = ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, node_config_file)
    proc = _create_node_service(algo_folders_env_variable_val, node_config_filepath)
    yield
    kill_node_service(proc)


@pytest.fixture(scope="session")
def smpc_localnode1_node_service(rabbitmq_smpc_localnode1, monetdb_smpc_localnode1):
    node_config_file = LOCALNODE1_SMPC_CONFIG_FILE
    algo_folders_env_variable_val = ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, node_config_file)
    proc = _create_node_service(algo_folders_env_variable_val, node_config_filepath)
    yield
    kill_node_service(proc)


@pytest.fixture(scope="session")
def smpc_localnode2_node_service(rabbitmq_smpc_localnode2, monetdb_smpc_localnode2):
    node_config_file = LOCALNODE2_SMPC_CONFIG_FILE
    algo_folders_env_variable_val = ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, node_config_file)
    proc = _create_node_service(algo_folders_env_variable_val, node_config_filepath)
    yield
    kill_node_service(proc)


@pytest.fixture(scope="function")
def localnodetmp_node_service(rabbitmq_localnodetmp, monetdb_localnodetmp):
    """
    ATTENTION!
    This node service fixture is the only one returning the process, so it can be killed.
    The scope of the fixture is function, so it won't break tests if the node service is killed.
    The rabbitmq and monetdb containers have also 'function' scope so this is VERY slow.
    This should be used only when the service should be killed e.g. for testing.
    """
    node_config_file = LOCALNODETMP_CONFIG_FILE
    algo_folders_env_variable_val = ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, node_config_file)
    proc = _create_node_service(algo_folders_env_variable_val, node_config_filepath)
    yield proc
    kill_node_service(proc)


def create_node_tasks_handler_celery(node_config_filepath):
    with open(node_config_filepath) as fp:
        tmp = toml.load(fp)
        node_id = tmp["identifier"]
        queue_domain = tmp["rabbitmq"]["ip"]
        queue_port = tmp["rabbitmq"]["port"]
        db_domain = tmp["monetdb"]["ip"]
        db_port = tmp["monetdb"]["port"]
    queue_address = ":".join([str(queue_domain), str(queue_port)])
    db_address = ":".join([str(db_domain), str(db_port)])

    return NodeTasksHandlerCelery(
        node_id=node_id,
        node_queue_addr=queue_address,
        node_db_addr=db_address,
        tasks_timeout=TASKS_TIMEOUT,
        run_udf_task_timoeut=RUN_UDF_TASK_TIMEOUT,
    )


@pytest.fixture(scope="function")
def globalnode_tasks_handler(globalnode_node_service):
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, GLOBALNODE_CONFIG_FILE)
    tasks_handler = create_node_tasks_handler_celery(node_config_filepath)
    yield tasks_handler
    tasks_handler.close()


@pytest.fixture(scope="function")
def localnode1_tasks_handler(localnode1_node_service):
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, LOCALNODE1_CONFIG_FILE)
    tasks_handler = create_node_tasks_handler_celery(node_config_filepath)
    yield tasks_handler
    tasks_handler.close()


@pytest.fixture(scope="function")
def localnode2_tasks_handler(localnode2_node_service):
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, LOCALNODE2_CONFIG_FILE)
    tasks_handler = create_node_tasks_handler_celery(node_config_filepath)
    yield tasks_handler
    tasks_handler.close()


@pytest.fixture(scope="function")
def localnodetmp_tasks_handler(localnodetmp_node_service):
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, LOCALNODETMP_CONFIG_FILE)
    tasks_handler = create_node_tasks_handler_celery(node_config_filepath)
    yield tasks_handler
    tasks_handler.close()


@pytest.fixture(scope="function")
def reset_node_landscape_aggregator():
    nla = NodeLandscapeAggregator()
    nla.keep_updating = False
    nla._node_registry = NodeRegistry(get_request_logger("NODE-REGISTRY"))
    nla._data_model_registry = DataModelRegistry(
        get_request_logger("DATA-MODEL-REGISTRY")
    )


@pytest.fixture(scope="session")
def smpc_coordinator():
    docker_cli = docker.from_env()

    # Start coordinator db
    try:
        container = docker_cli.containers.get(SMPC_COORD_DB_CONT_NAME)
    except docker.errors.NotFound:
        docker_cli.containers.run(
            image=SMPC_COORD_DB_IMAGE,
            name=SMPC_COORD_DB_CONT_NAME,
            detach=True,
            ports={27017: SMPC_COORD_DB_PORT},
            environment={
                "MONGO_INITDB_DATABASE": "agoradb",
                "MONGO_INITDB_ROOT_USERNAME": "sysadmin",
                "MONGO_INITDB_ROOT_PASSWORD": "123qwe",
            },
        )

    # Start coordinator queue
    try:
        container = docker_cli.containers.get(SMPC_COORD_QUEUE_CONT_NAME)
    except docker.errors.NotFound:
        docker_cli.containers.run(
            image=SMPC_COORD_QUEUE_IMAGE,
            name=SMPC_COORD_QUEUE_CONT_NAME,
            detach=True,
            ports={6379: SMPC_COORD_QUEUE_PORT},
            environment={
                "REDIS_REPLICATION_MODE": "master",
            },
            command="redis-server --requirepass agora",
        )

    # Start coordinator
    try:
        container = docker_cli.containers.get(SMPC_COORD_CONT_NAME)
    except docker.errors.NotFound:
        docker_cli.containers.run(
            image=SMPC_CLUSTER_IMAGE,
            name=SMPC_COORD_CONT_NAME,
            detach=True,
            ports={12314: SMPC_COORD_PORT},
            environment={
                "PLAYER_REPO_0": f"http://{COMMON_IP}:{SMPC_PLAYER1_PORT2}",
                "PLAYER_REPO_1": f"http://{COMMON_IP}:{SMPC_PLAYER2_PORT2}",
                "PLAYER_REPO_2": f"http://{COMMON_IP}:{SMPC_PLAYER3_PORT2}",
                "DB_URL": f"{COMMON_IP}:{SMPC_COORD_DB_PORT}",
                "REDIS_HOST": f"{COMMON_IP}",
                "REDIS_PORT": f"{SMPC_COORD_QUEUE_PORT}",
                "REDIS_PSWD": "agora",
            },
            command="python coordinator.py",
        )

    yield

    # TODO Very slow development if containers are always removed afterwards
    # db_cont = docker_cli.containers.get(SMPC_COORD_DB_CONT_NAME)
    # db_cont.remove(v=True, force=True)
    # queue_cont = docker_cli.containers.get(SMPC_COORD_QUEUE_CONT_NAME)
    # queue_cont.remove(v=True, force=True)
    # coord_cont = docker_cli.containers.get(SMPC_COORD_CONT_NAME)
    # coord_cont.remove(v=True, force=True)


@pytest.fixture(scope="session")
def smpc_players():
    docker_cli = docker.from_env()

    # Start player 1
    try:
        container = docker_cli.containers.get(SMPC_PLAYER1_CONT_NAME)
    except docker.errors.NotFound:
        docker_cli.containers.run(
            image=SMPC_CLUSTER_IMAGE,
            name=SMPC_PLAYER1_CONT_NAME,
            detach=True,
            ports={
                5000: SMPC_PLAYER1_PORT1,
                7100: SMPC_PLAYER1_PORT2,
                14000: SMPC_PLAYER1_PORT3,
            },
            environment={
                "PLAYER_REPO_0": f"http://{COMMON_IP}:{SMPC_PLAYER1_PORT2}",
                "PLAYER_REPO_1": f"http://{COMMON_IP}:{SMPC_PLAYER2_PORT2}",
                "PLAYER_REPO_2": f"http://{COMMON_IP}:{SMPC_PLAYER3_PORT2}",
                "DB_URL": f"{COMMON_IP}:{SMPC_COORD_DB_PORT}",
                "COORDINATOR_URL": f"http://{COMMON_IP}:{SMPC_COORD_PORT}",
            },
            command="python player.py 0",
        )

    # Start player 2
    try:
        container = docker_cli.containers.get(SMPC_PLAYER2_CONT_NAME)
    except docker.errors.NotFound:
        docker_cli.containers.run(
            image=SMPC_CLUSTER_IMAGE,
            name=SMPC_PLAYER2_CONT_NAME,
            detach=True,
            ports={
                5001: SMPC_PLAYER2_PORT1,
                7101: SMPC_PLAYER2_PORT2,
                14001: SMPC_PLAYER2_PORT3,
            },
            environment={
                "PLAYER_REPO_0": f"http://{COMMON_IP}:{SMPC_PLAYER1_PORT2}",
                "PLAYER_REPO_1": f"http://{COMMON_IP}:{SMPC_PLAYER2_PORT2}",
                "PLAYER_REPO_2": f"http://{COMMON_IP}:{SMPC_PLAYER3_PORT2}",
                "DB_URL": f"{COMMON_IP}:{SMPC_COORD_DB_PORT}",
                "COORDINATOR_URL": f"http://{COMMON_IP}:{SMPC_COORD_PORT}",
            },
            command="python player.py 1",
        )

    # Start player 3
    try:
        container = docker_cli.containers.get(SMPC_PLAYER3_CONT_NAME)
    except docker.errors.NotFound:
        docker_cli.containers.run(
            image=SMPC_CLUSTER_IMAGE,
            name=SMPC_PLAYER3_CONT_NAME,
            detach=True,
            ports={
                5002: SMPC_PLAYER3_PORT1,
                7102: SMPC_PLAYER3_PORT2,
                14002: SMPC_PLAYER3_PORT3,
            },
            environment={
                "PLAYER_REPO_0": f"http://{COMMON_IP}:{SMPC_PLAYER1_PORT2}",
                "PLAYER_REPO_1": f"http://{COMMON_IP}:{SMPC_PLAYER2_PORT2}",
                "PLAYER_REPO_2": f"http://{COMMON_IP}:{SMPC_PLAYER3_PORT2}",
                "DB_URL": f"{COMMON_IP}:{SMPC_COORD_DB_PORT}",
                "COORDINATOR_URL": f"http://{COMMON_IP}:{SMPC_COORD_PORT}",
            },
            command="python player.py 2",
        )

    yield

    # TODO Very slow development if containers are always removed afterwards
    # player1_cont = docker_cli.containers.get(SMPC_PLAYER1_CONT_NAME)
    # player1_cont.remove(v=True, force=True)
    # player2_cont = docker_cli.containers.get(SMPC_PLAYER2_CONT_NAME)
    # player2_cont.remove(v=True, force=True)
    # player3_cont = docker_cli.containers.get(SMPC_PLAYER3_CONT_NAME)
    # player3_cont.remove(v=True, force=True)


@pytest.fixture(scope="session")
def smpc_clients():
    docker_cli = docker.from_env()

    # Start client 1
    try:
        container = docker_cli.containers.get(SMPC_CLIENT1_CONT_NAME)
    except docker.errors.NotFound:
        with open(path.join(TEST_ENV_CONFIG_FOLDER, LOCALNODE1_SMPC_CONFIG_FILE)) as fp:
            tmp = toml.load(fp)
            client_id = tmp["smpc"]["client_id"]
        docker_cli.containers.run(
            image=SMPC_CLUSTER_IMAGE,
            name=SMPC_CLIENT1_CONT_NAME,
            detach=True,
            ports={
                SMPC_CLIENT1_PORT: SMPC_CLIENT1_PORT,
            },
            environment={
                "PLAYER_REPO_0": f"http://{COMMON_IP}:{SMPC_PLAYER1_PORT2}",
                "PLAYER_REPO_1": f"http://{COMMON_IP}:{SMPC_PLAYER2_PORT2}",
                "PLAYER_REPO_2": f"http://{COMMON_IP}:{SMPC_PLAYER3_PORT2}",
                "COORDINATOR_URL": f"http://{COMMON_IP}:{SMPC_COORD_PORT}",
                "ID": client_id,
                "PORT": f"{SMPC_CLIENT1_PORT}",
            },
            command=f"python client.py",
        )

    # Start client 2
    try:
        container = docker_cli.containers.get(SMPC_CLIENT2_CONT_NAME)
    except docker.errors.NotFound:
        with open(path.join(TEST_ENV_CONFIG_FOLDER, LOCALNODE2_SMPC_CONFIG_FILE)) as fp:
            tmp = toml.load(fp)
            client_id = tmp["smpc"]["client_id"]
        docker_cli.containers.run(
            image=SMPC_CLUSTER_IMAGE,
            name=SMPC_CLIENT2_CONT_NAME,
            detach=True,
            ports={
                SMPC_CLIENT2_PORT: SMPC_CLIENT2_PORT,
            },
            environment={
                "PLAYER_REPO_0": f"http://{COMMON_IP}:{SMPC_PLAYER1_PORT2}",
                "PLAYER_REPO_1": f"http://{COMMON_IP}:{SMPC_PLAYER2_PORT2}",
                "PLAYER_REPO_2": f"http://{COMMON_IP}:{SMPC_PLAYER3_PORT2}",
                "COORDINATOR_URL": f"http://{COMMON_IP}:{SMPC_COORD_PORT}",
                "ID": client_id,
                "PORT": f"{SMPC_CLIENT2_PORT}",
            },
            command="python client.py",
        )

    yield

    # TODO Very slow development if containers are always removed afterwards
    # client1_cont = docker_cli.containers.get(SMPC_CLIENT1_CONT_NAME)
    # client1_cont.remove(v=True, force=True)
    # client2_cont = docker_cli.containers.get(SMPC_CLIENT2_CONT_NAME)
    # client2_cont.remove(v=True, force=True)


@pytest.fixture(scope="session")
def smpc_cluster(smpc_coordinator, smpc_players, smpc_clients):
    time.sleep(20)  # TODO Check when the smpc cluster is actually ready
    yield
