"""Frontend-only configuration, independent of the model environment."""
import os
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Config:
    api_url: str = ""
    timeout: float = 15.0

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
        return cls(url, timeout)
