import enum

from pydantic import BaseModel
from typing import List


class SMPCRequestType(enum.Enum):
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    UNION = "union"
    PRODUCT = "product"

    def __str__(self):
        return self.name


class SMPCResponseStatus(enum.Enum):
    IN_QUEUE = "IN_QUEUE"
    RUNNING = "RUNNING"
    VALIDATING = "VALIDATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

    def __str__(self):
        return self.name


class SMPCRequestData(BaseModel):
    computationType: SMPCRequestType
    clients: List[int]


class SMPCResponse(BaseModel):
    computationType: SMPCRequestType
    jobId: str
    status: SMPCResponseStatus


class SMPCResponseWithOutput(BaseModel):
    computationOutput: List[int]
    computationType: SMPCRequestType
    jobId: str
    status: SMPCResponseStatus
