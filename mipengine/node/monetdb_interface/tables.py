from typing import List

from mipengine.common.validate_identifier_names import validate_identifier_names
from mipengine.node.monetdb_interface import common
from mipengine.node.monetdb_interface.common import connection
from mipengine.node.monetdb_interface.common import convert_schema_to_sql_query_format
from mipengine.node.monetdb_interface.common import cursor
from mipengine.common.node_tasks_DTOs import TableInfo


def get_tables_names(context_id: str) -> List[str]:
    return common.get_tables_names("normal", context_id)


@validate_identifier_names
def create_table(table_info: TableInfo):
    columns_schema = convert_schema_to_sql_query_format(table_info.schema)
    cursor.execute(f"CREATE TABLE {table_info.name} ( {columns_schema} )")
    connection.commit()
