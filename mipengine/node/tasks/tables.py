from typing import List
from typing import Union

from celery import shared_task

from mipengine.node import config as node_config
from mipengine.node.monetdb_interface import tables
from mipengine.node.monetdb_interface.common_actions import create_table_name
from mipengine.node.node_logger import initialise_logger
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import TableType


@shared_task
@initialise_logger
def get_tables(request_id: str, context_id: str) -> List[str]:
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging
    context_id : str
    The id of the experiment

    Returns
    ------
    List[str]
        A list of table names
    """
    return tables.get_table_names(context_id)


@shared_task
@initialise_logger
def create_table(
    request_id: str, context_id: str, command_id: str, schema_json: str
) -> str:
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging
    context_id : str
        The id of the experiment
    command_id : str
        The id of the command that the table
    schema_json : str(TableSchema)
        A TableSchema object in a jsonified format

    Returns
    ------
    str
        The name of the created table in lower case
    """
    schema = TableSchema.parse_raw(schema_json)
    table_name = create_table_name(
        TableType.NORMAL,
        node_config.identifier,
        context_id,
        command_id,
    )
    tables.create_table(table_name, schema)
    return table_name


@shared_task
@initialise_logger
def insert_data_to_table(
    request_id: str, table_name: str, values: List[List[Union[str, int, float, bool]]]
):
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging
    table_name : str
        The name of the table
    values : List[List[Union[str, int, float, bool]]
        The data of the table to be inserted
    """
    tables.insert_data_to_table(table_name, values)
