"""Frontend-only configuration, independent of the model environment."""
import os
from dataclasses import dataclass, field
from ipaddress import IPv4Address, IPv4Network
from urllib.parse import urlsplit


PRIVATE_NETWORKS = tuple(map(IPv4Network, ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8")))


def private_evaluation_host(host):
    try:
        address = IPv4Address(host)
    except ValueError:
        return False
    return any(address in network for network in PRIVATE_NETWORKS)


@dataclass(frozen=True)
class Config:
    api_url: str = ""
    timeout: float = 15.0
    final_token: str = field(default="", repr=False)

    @classmethod
    def from_env(cls):
        url = os.getenv("PATHSNAP_API_URL", "").strip().rstrip("/")
        if url:
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("PATHSNAP_API_URL must be an HTTP(S) base URL without credentials, query or fragment.")
        timeout = float(os.getenv("PATHSNAP_REQUEST_TIMEOUT", "15"))
        if not 0 < timeout <= 120:
            raise ValueError("PATHSNAP_REQUEST_TIMEOUT must be between 0 and 120 seconds.")
        token = os.getenv("PATHSNAP_FINAL_TOKEN", "")
        if token and not private_evaluation_host(urlsplit(url).hostname):
            raise ValueError("Final evaluation requires a private IPv4 API URL.")
        return cls(url, timeout, token)
