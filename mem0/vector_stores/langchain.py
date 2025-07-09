import logging
from itertools import zip_longest
from typing import Dict, List, Optional

from langchain_community.vectorstores import VectorStore
from pydantic import BaseModel

from mem0.vector_stores.base import VectorStoreBase

try:
    from langchain_community.vectorstores import VectorStore
except ImportError:
    raise ImportError(
        "The 'langchain_community' library is required. Please install it using 'pip install langchain_community'."
    )


logger = logging.getLogger(__name__)


class OutputData(BaseModel):
    id: Optional[str]  # memory id
    score: Optional[float]  # distance
    payload: Optional[Dict]  # metadata


class Langchain(VectorStoreBase):
    def __init__(self, client: VectorStore, collection_name: str = "mem0"):
        self.client = client
        self.collection_name = collection_name

    def _parse_output(self, data: Dict) -> List[OutputData]:
        """
        Parse the output data.

        Args:
            data (Dict): Output data or list of Document objects.

        Returns:
            List[OutputData]: Parsed output data.
        """
        # Fast path: list of Document-like objects
        if isinstance(data, list):
            docs = data
            # Avoid repeated checks in the loop below
            # If all elements look like Document objects (have __dict__ and metadata)
            is_doc_obj = all(hasattr(doc, "__dict__") and hasattr(doc, "metadata") for doc in docs)
            if is_doc_obj:
                # Use list comprehension for speed
                return [
                    OutputData(
                        id=getattr(doc, "id", None),
                        score=None,
                        payload=getattr(doc, "metadata", {}),
                    )
                    for doc in docs
                ]

        # Original format handling (assume dict with keys)
        keys = ("ids", "distances", "metadatas")
        # Use tuple unpacking for speed and clarity
        # Also, avoid multi-level lists if possible
        ids, distances, metadatas = (data.get(k, []) for k in keys)
        # Unwrap first element if it's a list-of-list (old Chroma output)
        ids = ids[0] if isinstance(ids, list) and ids and isinstance(ids[0], list) else ids
        distances = (
            distances[0] if isinstance(distances, list) and distances and isinstance(distances[0], list) else distances
        )
        metadatas = (
            metadatas[0] if isinstance(metadatas, list) and metadatas and isinstance(metadatas[0], list) else metadatas
        )

        # Parallel iteration, fillvalues set to None to keep semantics and avoid index errors
        result = [
            OutputData(
                id=idx,
                score=dist,
                payload=meta,
            )
            for idx, dist, meta in zip_longest(
                ids if isinstance(ids, list) else [],
                distances if isinstance(distances, list) else [],
                metadatas if isinstance(metadatas, list) else [],
                fillvalue=None,
            )
        ]
        return result

    def create_col(self, name, vector_size=None, distance=None):
        self.collection_name = name
        return self.client

    def insert(
        self, vectors: List[List[float]], payloads: Optional[List[Dict]] = None, ids: Optional[List[str]] = None
    ):
        """
        Insert vectors into the LangChain vectorstore.
        """
        # Check if client has add_embeddings method
        if hasattr(self.client, "add_embeddings"):
            # Some LangChain vectorstores have a direct add_embeddings method
            self.client.add_embeddings(embeddings=vectors, metadatas=payloads, ids=ids)
        else:
            # Fallback to add_texts method
            texts = [payload.get("data", "") for payload in payloads] if payloads else [""] * len(vectors)
            self.client.add_texts(texts=texts, metadatas=payloads, ids=ids)

    def search(self, query: str, vectors: List[List[float]], limit: int = 5, filters: Optional[Dict] = None):
        """
        Search for similar vectors in LangChain.
        """
        # For each vector, perform a similarity search
        if filters:
            results = self.client.similarity_search_by_vector(embedding=vectors, k=limit, filter=filters)
        else:
            results = self.client.similarity_search_by_vector(embedding=vectors, k=limit)

        final_results = self._parse_output(results)
        return final_results

    def delete(self, vector_id):
        """
        Delete a vector by ID.
        """
        self.client.delete(ids=[vector_id])

    def update(self, vector_id, vector=None, payload=None):
        """
        Update a vector and its payload.
        """
        self.delete(vector_id)
        self.insert(vector, payload, [vector_id])

    def get(self, vector_id):
        """
        Retrieve a vector by ID.
        """
        docs = self.client.get_by_ids([vector_id])
        if docs and len(docs) > 0:
            doc = docs[0]
            return self._parse_output([doc])[0]
        return None

    def list_cols(self):
        """
        List all collections.
        """
        # LangChain doesn't have collections
        return [self.collection_name]

    def delete_col(self):
        """
        Delete a collection.
        """
        logger.warning("Deleting collection")
        if hasattr(self.client, "delete_collection"):
            self.client.delete_collection()
        elif hasattr(self.client, "reset_collection"):
            self.client.reset_collection()
        else:
            self.client.delete(ids=None)

    def col_info(self):
        """
        Get information about a collection.
        """
        return {"name": self.collection_name}

    def list(self, filters=None, limit=None):
        """
        List all vectors in a collection.
        """
        try:
            col = getattr(self.client, "_collection", None)
            get_fn = getattr(col, "get", None)
            if get_fn is not None:
                # Convert mem0 filters to Chroma where clause if needed
                where_clause = {"user_id": filters["user_id"]} if filters and "user_id" in filters else None

                result = get_fn(where=where_clause, limit=limit)

                # Convert the result to the expected format
                if result and isinstance(result, dict):
                    return [self._parse_output(result)]
                return []
        except Exception as e:
            logger.error(f"Error listing vectors from Chroma: {e}")
            return []

    def reset(self):
        """Reset the index by deleting and recreating it."""
        logger.warning(f"Resetting collection: {self.collection_name}")
        self.delete_col()
