import logging
import os
import platform
import sys

from posthog import Posthog

import mem0
from mem0.memory.setup import get_or_create_user_id

MEM0_TELEMETRY = os.environ.get("MEM0_TELEMETRY", "True")
PROJECT_API_KEY = "phc_hgJkUVJFYtmaJqrvf6CYN67TIQ8yhXAkWzUn9AMU4yX"
HOST = "https://us.i.posthog.com"

if isinstance(MEM0_TELEMETRY, str):
    MEM0_TELEMETRY = MEM0_TELEMETRY.lower() in ("true", "1", "yes")

if not isinstance(MEM0_TELEMETRY, bool):
    raise ValueError("MEM0_TELEMETRY must be a boolean value.")

logging.getLogger("posthog").setLevel(logging.CRITICAL + 1)
logging.getLogger("urllib3").setLevel(logging.CRITICAL + 1)


class AnonymousTelemetry:
    def __init__(self, vector_store=None):
        self.posthog = Posthog(project_api_key=PROJECT_API_KEY, host=HOST)

        self.user_id = get_or_create_user_id(vector_store)

        if not MEM0_TELEMETRY:
            self.posthog.disabled = True

    def capture_event(self, event_name, properties=None, user_email=None):
        if properties is None:
            properties = {}
        properties = {
            "client_source": "python",
            "client_version": mem0.__version__,
            "python_version": sys.version,
            "os": sys.platform,
            "os_version": platform.version(),
            "os_release": platform.release(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            **properties,
        }
        distinct_id = self.user_id if user_email is None else user_email
        self.posthog.capture(distinct_id=distinct_id, event=event_name, properties=properties)

    def close(self):
        self.posthog.shutdown()


client_telemetry = AnonymousTelemetry()


def capture_event(event_name, memory_instance, additional_data=None):
    # Only use a single AnonymousTelemetry instance across calls
    _anonymous_telemetry.capture_event(event_name, _build_event_data(memory_instance, additional_data))


def capture_client_event(event_name, instance, additional_data=None):
    event_data = {
        "function": f"{instance.__class__.__module__}.{instance.__class__.__name__}",
    }
    if additional_data:
        event_data.update(additional_data)

    client_telemetry.capture_event(event_name, event_data, instance.user_email)


def _build_event_data(mem, additional_data=None):
    # Precompute class/module names only once for each instance; reuse
    if not hasattr(mem, "_event_telemetry_cache"):
        mem._event_telemetry_cache = {
            "collection": mem.collection_name,
            "vector_size": mem.embedding_model.config.embedding_dims,
            "history_store": "sqlite",
            "graph_store": (
                f"{mem.graph.__class__.__module__}.{mem.graph.__class__.__name__}"
                if mem.config.graph_store.config and mem.graph is not None
                else None
            ),
            "vector_store": (f"{mem.vector_store.__class__.__module__}.{mem.vector_store.__class__.__name__}"),
            "llm": (f"{mem.llm.__class__.__module__}.{mem.llm.__class__.__name__}"),
            "embedding_model": (f"{mem.embedding_model.__class__.__module__}.{mem.embedding_model.__class__.__name__}"),
            "function": (f"{mem.__class__.__module__}.{mem.__class__.__name__}.{mem.api_version}"),
        }
    event_data = mem._event_telemetry_cache.copy()
    if additional_data:
        event_data.update(additional_data)
    return event_data


_anonymous_telemetry = AnonymousTelemetry()
