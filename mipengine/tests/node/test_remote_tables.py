import pymonetdb
import pytest

from mipengine.common.DTOs import ColumnInfo
from mipengine.common.DTOs import TableInfo
from mipengine.common.DTOs import TableSchema
from mipengine.tests.node import nodes_communication
from mipengine.common.node_catalog import node_catalog

global_node_id = "global_node"
local_node_id = "local_node_1"
global_node = nodes_communication.get_celery_app(global_node_id)
local_node = nodes_communication.get_celery_app(local_node_id)
local_node_create_table = nodes_communication.get_celery_create_table_signature(local_node)
global_node_create_remote_table = nodes_communication.get_celery_create_remote_table_signature(global_node)
global_node_get_remote_tables = nodes_communication.get_celery_get_remote_tables_signature(global_node)
local_node_cleanup = nodes_communication.get_celery_cleanup_signature(local_node)
global_node_cleanup = nodes_communication.get_celery_cleanup_signature(global_node)

context_id = "regrEssion"


@pytest.fixture(autouse=True)
def cleanup_tables():
    yield

    local_node_cleanup.delay(context_id=context_id.lower()).get()


def test_create_and_get_remote_table():
    local_node_data = node_catalog.get_local_node_data(local_node_id)
    local_node_1_url = f'mapi:monetdb://{local_node_data.monetdbHostname}:{local_node_data.monetdbPort}/db'

    table_schema = TableSchema([ColumnInfo("col1", "INT"), ColumnInfo("col2", "FLOAT"), ColumnInfo("col3", "TEXT")])

    table_name = local_node_create_table.delay(context_id=context_id,
                                               command_id=str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                               schema_json=table_schema.to_json()
                                               ).get()

    table_info = TableInfo(table_name, table_schema)

    global_node_create_remote_table.delay(table_info_json=table_info.to_json(),
                                          url=local_node_1_url).get()
    remote_tables = global_node_get_remote_tables.delay(context_id=context_id).get()
    assert table_name.lower() in remote_tables
