import json
import os
import uuid

# Set up the directory path
VECTOR_ID = str(uuid.uuid4())
home_dir = os.path.expanduser("~")
mem0_dir = os.environ.get("MEM0_DIR") or os.path.join(home_dir, ".mem0")
os.makedirs(mem0_dir, exist_ok=True)


def setup_config():
    config_path = os.path.join(mem0_dir, "config.json")
    if not os.path.exists(config_path):
        user_id = str(uuid.uuid4())
        config = {"user_id": user_id}
        with open(config_path, "w") as config_file:
            json.dump(config, config_file, indent=4)


def get_user_id():
    config_path = os.path.join(mem0_dir, "config.json")
    try:
        with open(config_path, "r") as config_file:
            # Minimize memory and parsing by loading just the user_id field
            for line in config_file:
                if '"user_id"' in line:
                    # Use a lightweight parsing to extract the user_id value
                    line = line.strip()
                    # Assumes: "user_id": "some_value"
                    split = line.split(":", 1)
                    if len(split) == 2:
                        val = split[1].strip().rstrip(",").strip()
                        if val.startswith('"') and val.endswith('"'):
                            return val[1:-1]
                        if val == "null":
                            return None
                        return val
            return "anonymous_user"  # If key not found
    except FileNotFoundError:
        return "anonymous_user"
    except Exception:
        return "anonymous_user"


def get_or_create_user_id(vector_store):
    """Store user_id in vector store and return it."""
    user_id = get_user_id()

    # Try to get existing user_id from vector store
    try:
        existing = vector_store.get(vector_id=user_id)
        if existing and hasattr(existing, "payload") and existing.payload and "user_id" in existing.payload:
            return existing.payload["user_id"]
    except Exception:
        pass

    # If we get here, we need to insert the user_id
    try:
        dims = getattr(vector_store, "embedding_model_dims", 1536)
        vector_store.insert(
            vectors=[[0.1] * dims], payloads=[{"user_id": user_id, "type": "user_identity"}], ids=[user_id]
        )
    except Exception:
        pass

    return user_id
