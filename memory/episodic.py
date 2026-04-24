"""
Episodic memory backend: JSON list log store.
Lưu các sự kiện/episodes có ý nghĩa từ các cuộc hội thoại.
Mỗi episode có timestamp, summary, outcome, tags.
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


EPISODIC_FILE = "data/episodic_log.json"


class EpisodicMemory:
    """
    Episodic memory lưu lại các sự kiện/episodes quan trọng.
    Dùng JSON file list store.
    Có thể search theo keyword hoặc tags.
    """

    def __init__(self, log_path: str = EPISODIC_FILE):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._episodes: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        if os.path.exists(self.log_path):
            with open(self.log_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self._episodes, f, ensure_ascii=False, indent=2)

    def add_episode(
        self,
        summary: str,
        outcome: str = "",
        tags: List[str] = None,
        extra: Dict = None,
    ) -> None:
        """
        Ghi một episode mới vào log.
        Args:
            summary: Tóm tắt những gì đã xảy ra
            outcome: Kết quả (success/fail/pending)
            tags: Tags để tìm kiếm sau này
            extra: Metadata bổ sung
        """
        episode = {
            "id": len(self._episodes) + 1,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "outcome": outcome,
            "tags": tags or [],
            "extra": extra or {},
        }
        self._episodes.append(episode)
        self._save()

    def get_recent_episodes(self, n: int = 5) -> List[Dict]:
        """Lấy n episodes gần nhất."""
        return self._episodes[-n:] if self._episodes else []

    def search_episodes(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Tìm kiếm episodes theo keyword (simple string matching).
        Args:
            query: Từ khóa tìm kiếm
            top_k: Số kết quả trả về
        """
        query_lower = query.lower()
        results = []
        for ep in self._episodes:
            score = 0
            if query_lower in ep["summary"].lower():
                score += 2
            if any(query_lower in tag.lower() for tag in ep["tags"]):
                score += 1
            if query_lower in ep.get("outcome", "").lower():
                score += 1
            if score > 0:
                results.append((score, ep))

        results.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in results[:top_k]]

    def get_episodes_text(self, episodes: List[Dict] = None, n: int = 3) -> str:
        """Convert episodes thành text để inject vào prompt."""
        if episodes is None:
            episodes = self.get_recent_episodes(n)
        if not episodes:
            return "(Chưa có episode nào được ghi nhận)"
        lines = []
        for ep in episodes:
            ts = ep["timestamp"][:10]
            lines.append(f"- [{ts}] {ep['summary']}")
            if ep.get("outcome"):
                lines.append(f"  → Outcome: {ep['outcome']}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Xóa toàn bộ episodic log (dùng cho testing)."""
        self._episodes = []
        self._save()

    def __len__(self) -> int:
        return len(self._episodes)

    def __repr__(self) -> str:
        return f"EpisodicMemory(episodes={len(self._episodes)})"
