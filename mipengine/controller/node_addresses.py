import json
from abc import ABC
from abc import abstractmethod
from typing import List

import dns.resolver

from mipengine.controller import DeploymentType
from mipengine.controller import config as controller_config


class NodeAddresses(ABC):
    @abstractmethod
    def __init__(self):
        self.localnode_addresses = None

    def get_addresses(self) -> List[str]:
        return self.localnode_addresses


class LocalNodeAddresses(NodeAddresses):
    def __init__(self):
        with open(controller_config.localnodes.config_file) as fp:
            self.localnode_addresses = json.load(fp)


class DNSNodeAddresses(NodeAddresses):
    def __init__(self):
        localnode_ips = dns.resolver.query(controller_config.localnodes.dns, "A")
        self.localnode_addresses = [
            f"{ip}:{controller_config.localnodes.port}" for ip in localnode_ips
        ]


class NodeAddressesFactory:
    def __init__(self, depl_type: DeploymentType):
        self.depl_type = depl_type

    def get_node_addresses(self) -> NodeAddresses:
        if self.depl_type == DeploymentType.LOCAL:
            return LocalNodeAddresses()

        if self.depl_type == DeploymentType.KUBERNETES:
            return DNSNodeAddresses()

        raise ValueError(
            f"DeploymentType can be one of the following: {[t.value for t in DeploymentType]}, "
            f"value provided: '{self.depl_type}'"
        )
