from typing import List

from celery import shared_task

from mipengine.node.monetdb_interface import tables
from mipengine.node.monetdb_interface.common import config
from mipengine.node.monetdb_interface.common import create_table_name
from mipengine.node.tasks.data_classes import TableInfo
from mipengine.node.tasks.data_classes import TableSchema


@shared_task(serializer="json")
def get_tables(context_id: str) -> List[str]:
    """
        Parameters
        ----------
        context_id : str
        The id of the experiment

        Returns
        ------
        List[str]
            A list of table names
    """
    return tables.get_tables_names(context_id)


@shared_task(serializer="json")
def create_table(context_id: str, command_id: str, schema_json: str) -> str:
    """
        Parameters
        ----------
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
    schema_object = TableSchema.from_json(schema_json)
    table_name = create_table_name("table", command_id, context_id, config["node"]["identifier"])
    table_info = TableInfo(table_name.lower(), schema_object)
    tables.create_table(table_info)
    return table_name.lower()
