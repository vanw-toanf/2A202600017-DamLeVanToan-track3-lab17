"""
Short-term memory backend: ConversationBuffer với sliding window.
Lưu trữ recent conversation turns trong session.
"""
from collections import deque
from typing import List, Dict


class ConversationBufferMemory:
    """
    Sliding window conversation buffer.
    Chỉ giữ N turns gần nhất để tránh vượt token budget.
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._buffer: deque = deque(maxlen=max_turns)

    def add_turn(self, role: str, content: str) -> None:
        """Thêm một turn mới vào buffer."""
        self._buffer.append({"role": role, "content": content})

    def get_recent(self, n: int = None) -> List[Dict]:
        """Lấy n turns gần nhất. Nếu n=None trả về tất cả."""
        turns = list(self._buffer)
        if n is not None:
            turns = turns[-n:]
        return turns

    def get_history_text(self, n: int = None) -> str:
        """Chuyển turns thành text dạng readable cho prompt injection."""
        turns = self.get_recent(n)
        lines = []
        for t in turns:
            role = "User" if t["role"] == "user" else "Assistant"
            lines.append(f"{role}: {t['content']}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Xóa toàn bộ buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"ConversationBufferMemory(max_turns={self.max_turns}, current={len(self._buffer)})"
