from typing import List

from celery import shared_task

from worker.tasks.data_classes import TableInfo, TableView, TableData


@shared_task
def get_views() -> List[TableInfo]:
    pass


@shared_task
def create_view(view: TableView) -> TableInfo:
    pass


@shared_task
def get_view(view_name: str) -> TableData:
    pass


@shared_task
def delete_view(view_name: str):
    pass
