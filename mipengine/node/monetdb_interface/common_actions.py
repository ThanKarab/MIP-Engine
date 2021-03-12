from typing import List
from typing import Union

from pymonetdb import Connection
from pymonetdb.sql.cursors import Cursor

from mipengine.common.node_exceptions import TableCannotBeFound
from mipengine.common.validate_identifier_names import validate_identifier_names
from mipengine.node.monetdb_interface.connection_pool import execute_with_occ
from mipengine.node.node import config
from mipengine.common.node_tasks_DTOs import ColumnInfo
from mipengine.common.node_tasks_DTOs import TableSchema

MONETDB_VARCHAR_SIZE = 50


@validate_identifier_names
def create_table_name(table_type: str, command_id: str, context_id: str, node_id: str) -> str:
    """
    Creates a table name with the format <tableType>_<commandId>_<contextId>_<nodeId>
    """
    if table_type not in {"table", "view", "merge"}:
        raise TypeError(f"Table type is not acceptable: {table_type} .")
    if node_id not in {"global", config.get("node", "identifier")}:
        raise TypeError(f"Node Identifier is not acceptable: {node_id}.")

    return f"{table_type}_{command_id}_{context_id}_{node_id}"


def convert_schema_to_sql_query_format(schema: TableSchema) -> str:
    """Converts a table's schema to a sql query.

        Parameters
        ----------
        schema : TableSchema
            The schema of a table

        Returns
        ------
        str
            The schema in a sql query formatted string
    """
    return ', '.join(f"{column.name} {__convert_to_monetdb_column_type(column.data_type)}" for column in schema.columns)


@validate_identifier_names
def get_table_schema(cursor: Cursor, table_name: str, table_type: str = None) -> TableSchema:
    """Retrieves a schema for a specific table type and table name  from the monetdb.

        Parameters
        ----------
        cursor : Cursor
            Database cursor to use for queries
        table_type : str
            The type of the table
        table_name : str
            The name of the table

        Returns
        ------
        TableSchema
            A schema which is TableSchema object.
    """

    type_clause = ""
    if table_type is not None:
        type_clause = f" AND tables.type = {str(__convert_mip_table_type_to_monet(table_type))}"

    cursor.execute(
        f"""
        SELECT columns.name, columns.type 
        FROM columns 
        RIGHT JOIN tables 
        ON tables.id = columns.table_id 
        WHERE 
        tables.name = '{table_name}' 
        AND 
        tables.system=false 
        {type_clause}""")

    schema = cursor.fetchall()
    if not schema:
        raise TableCannotBeFound(table_name)
    return TableSchema([ColumnInfo(table[0], __convert_from_monetdb_column_type(table[1])) for table in schema])


@validate_identifier_names
def get_tables_names(cursor: Cursor, table_type: str, context_id: str) -> List[str]:
    """Retrieves a list of table names, which contain the context_id from the monetdb.

        Parameters
        ----------
        cursor : Cursor
            Database cursor to use for queries
        table_type : str
            The type of the table
        context_id : str
            The id of the experiment

        Returns
        ------
        List[str]
            A list of table names.
    """
    cursor.execute(
        f"""
        SELECT name FROM tables 
        WHERE
         type = {str(__convert_mip_table_type_to_monet(table_type))} AND
        name LIKE '%{context_id.lower()}%' AND 
        system = false""")

    return [table[0] for table in cursor]


@validate_identifier_names
def get_table_data(cursor: Cursor, table_name: str, table_type: str = None) -> List[List[Union[str, int, float, bool]]]:
    """Retrieves the data of a table with specific type and name  from the monetdb.

        Parameters
        ----------
        cursor : Cursor
            Database cursor to use for queries
        table_type : str
            The type of the table
        table_name : str
            The name of the table

        Returns
        ------
        List[List[Union[str, int, float, bool]]
            The data of the table.
    """

    type_clause = ""
    if table_type is not None:
        type_clause = f" AND tables.type = {str(__convert_mip_table_type_to_monet(table_type))}"

    cursor.execute(
        f"""
        SELECT {table_name}.* 
        FROM {table_name} 
        INNER JOIN tables ON tables.name = '{table_name}' 
        WHERE tables.system=false 
        {type_clause}
        """)

    return cursor.fetchall()


def clean_up(connection: Connection, cursor: Cursor, context_id: str):
    """Deletes all tables of any type with name that contain a specific context_id from the monetdb.

        Parameters
        ----------
        cursor : Cursor
            Database cursor to use for queries
        context_id : str
            The id of the experiment
    """
    for table_type in ("merge", "remote", "view", "normal"):
        __delete_table_by_type_and_context_id(connection, cursor, table_type, context_id)


def __convert_monet_table_type_to_mip(monet_table_type: int) -> str:
    """ Converts MonetDB's table types to MIP Engine's table types
    0 -> normal,
    1 -> view,
    3 -> merge,
    5 -> remote,
        """
    type_mapping = {
        0: "normal",
        1: "view",
        3: "merge",
        5: "remote",
    }

    if monet_table_type not in type_mapping.keys():
        raise ValueError(f"Type {monet_table_type} cannot be converted to MIP Engine's table types.")

    return type_mapping.get(monet_table_type)


def __convert_mip_table_type_to_monet(table_type: str) -> int:
    """ Converts MIP Engine's table types to MonetDB's table types
     normal -> 0,
     view -> 1,
     merge -> 3,
     remote -> 5,
        """
    type_mapping = {
        "normal": 0,
        "view": 1,
        "merge": 3,
        "remote": 5,
    }

    if table_type not in type_mapping.keys():
        raise ValueError(f"Type {table_type} cannot be converted to monetdb table type.")

    return type_mapping.get(str(table_type).lower())


def __convert_to_monetdb_column_type(column_type: str) -> str:
    """ Converts MIP Engine's int,float,text types to monetdb
    int -> integer
    float -> double
    text -> varchar(50s)
    bool -> boolean
    clob -> clob
    """
    type_mapping = {
        "int": "int",
        "float": "double",
        "text": f"varchar({MONETDB_VARCHAR_SIZE})",
        "bool": "bool",
        "clob": "clob",
    }

    if column_type not in type_mapping.keys():
        raise ValueError(f"Type {column_type} cannot be converted to monetdb column type.")

    return type_mapping.get(str(column_type).lower())


def __convert_from_monetdb_column_type(column_type: str) -> str:
    """ Converts MonetDB's types to MIP Engine's types
    int ->  int
    double  -> float
    varchar(50)  -> text
    boolean -> bool
    clob -> clob
    """
    type_mapping = {
        "int": "int",
        "double": "float",
        "varchar": "text",
        "bool": "bool",
        "clob": "clob",
    }

    if column_type not in type_mapping.keys():
        raise ValueError(f"Type {column_type} cannot be converted to MIP Engine's types.")

    return type_mapping.get(str(column_type).lower())


@validate_identifier_names
def __delete_table_by_type_and_context_id(connection: Connection, cursor: Cursor, table_type: str, context_id: str):
    """Deletes all tables of specific type with name that contain a specific context_id from the monetdb.

        Parameters
        ----------
        cursor : Cursor
            Database cursor to use for queries
        table_type : str
            The type of the table
        context_id : str
            The id of the experiment
    """
    cursor.execute(
        f"""
        SELECT name, type FROM tables 
        WHERE name LIKE '%{context_id.lower()}%'
        AND tables.type = {str(__convert_mip_table_type_to_monet(table_type))} 
        AND system = false
        """)
    for table in cursor.fetchall():
        if table[1] == 1:
            execute_with_occ(connection, cursor, f"DROP VIEW {table[0]}")
        else:
            execute_with_occ(connection, cursor, f"DROP TABLE {table[0]}")
