from typing import List

from celery import shared_task

from mipengine.node.monetdb_interface import remote_tables
from mipengine.node.node_logger import initialise_logger
from mipengine.node_tasks_DTOs import TableSchema


@shared_task
@initialise_logger
def get_remote_tables(request_id: str, context_id: str) -> List[str]:
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
        A list of remote table names
    """
    return remote_tables.get_remote_table_names(context_id)


@shared_task
@initialise_logger
def create_remote_table(
    request_id: str,
    table_name: str,
    table_schema_json: str,
    monetdb_socket_address: str,
):
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging
    table_name : str
        The name of the table.
    table_schema_json : str(TableSchema)
        A TableSchema object in a jsonified format
    monetdb_socket_address : str
        The monetdb_socket_address of the monetdb that we want to create the remote table from.
    """
    schema = TableSchema.parse_raw(table_schema_json)
    remote_tables.create_remote_table(
        name=table_name, schema=schema, monetdb_socket_address=monetdb_socket_address
    )
