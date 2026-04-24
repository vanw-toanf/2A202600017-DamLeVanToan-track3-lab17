"""
Semantic memory backend: ChromaDB vector store với FAISS/keyword fallback.
Lưu các chunks kiến thức/FAQ để semantic retrieval.
"""
import os
from typing import List, Optional

# Try ChromaDB first, fallback to simple keyword index
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


CHROMA_DIR = "data/chroma_db"


class SemanticMemory:
    """
    Semantic memory sử dụng ChromaDB (hoặc keyword fallback).
    Lưu trữ chunks kiến thức để retrieval theo semantic similarity.
    
    ChromaDB sử dụng embedded sentence transformer để tạo embeddings.
    Keyword fallback dùng simple TF-IDF-like scoring.
    """

    def __init__(self, collection_name: str = "semantic_memory", persist_dir: str = CHROMA_DIR):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self._use_chroma = False
        self._collection = None
        self._keyword_store: List[dict] = []  # fallback store

        if CHROMA_AVAILABLE:
            self._init_chroma(persist_dir, collection_name)
        else:
            print("[SemanticMemory] ChromaDB không khả dụng, dùng keyword fallback.")

    def _init_chroma(self, persist_dir: str, collection_name: str) -> None:
        try:
            os.makedirs(persist_dir, exist_ok=True)
            client = chromadb.PersistentClient(path=persist_dir)
            self._collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._use_chroma = True
            print(f"[SemanticMemory] ChromaDB khởi tạo thành công: {self._collection.count()} docs.")
        except Exception as e:
            print(f"[SemanticMemory] Lỗi ChromaDB: {e}. Dùng keyword fallback.")
            self._use_chroma = False

    def add_document(self, doc_id: str, text: str, metadata: dict = None) -> None:
        """
        Thêm một document/chunk vào semantic store.
        Args:
            doc_id: Unique ID cho document
            text: Nội dung text
            metadata: Metadata bổ sung (source, topic, ...)
        """
        if self._use_chroma and self._collection is not None:
            try:
                # Check if already exists
                existing = self._collection.get(ids=[doc_id])
                if existing["ids"]:
                    # Update
                    self._collection.update(
                        ids=[doc_id],
                        documents=[text],
                        metadatas=[metadata or {}],
                    )
                else:
                    self._collection.add(
                        ids=[doc_id],
                        documents=[text],
                        metadatas=[metadata or {}],
                    )
                return
            except Exception as e:
                print(f"[SemanticMemory] ChromaDB add error: {e}. Falling back to keyword.")

        # Keyword fallback
        self._keyword_store = [d for d in self._keyword_store if d["id"] != doc_id]
        self._keyword_store.append({
            "id": doc_id,
            "text": text,
            "metadata": metadata or {},
        })

    def semantic_search(self, query: str, top_k: int = 3) -> List[str]:
        """
        Tìm kiếm semantic (hoặc keyword) cho query.
        Args:
            query: Câu hỏi / query text
            top_k: Số kết quả trả về
        Returns:
            List các text chunks liên quan
        """
        if self._use_chroma and self._collection is not None:
            try:
                if self._collection.count() == 0:
                    return []
                results = self._collection.query(
                    query_texts=[query],
                    n_results=min(top_k, self._collection.count()),
                )
                docs = results.get("documents", [[]])[0]
                return docs
            except Exception as e:
                print(f"[SemanticMemory] ChromaDB query error: {e}. Falling back.")

        # Keyword fallback
        return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int = 3) -> List[str]:
        """Simple keyword-based search fallback."""
        query_words = set(query.lower().split())
        scores = []
        for doc in self._keyword_store:
            doc_words = set(doc["text"].lower().split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                scores.append((overlap, doc["text"]))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scores[:top_k]]

    def get_hits_text(self, query: str, top_k: int = 3) -> str:
        """Lấy semantic hits dưới dạng text để inject vào prompt."""
        hits = self.semantic_search(query, top_k)
        if not hits:
            return "(Không tìm thấy thông tin liên quan)"
        lines = [f"- {h}" for h in hits]
        return "\n".join(lines)

    def count(self) -> int:
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        return len(self._keyword_store)

    def clear(self) -> None:
        """Xóa toàn bộ semantic store (dùng cho testing)."""
        if self._use_chroma and self._collection is not None:
            try:
                all_ids = self._collection.get()["ids"]
                if all_ids:
                    self._collection.delete(ids=all_ids)
            except Exception:
                pass
        self._keyword_store = []

    def __repr__(self) -> str:
        backend = "ChromaDB" if self._use_chroma else "KeywordFallback"
        return f"SemanticMemory(backend={backend}, count={self.count()})"
