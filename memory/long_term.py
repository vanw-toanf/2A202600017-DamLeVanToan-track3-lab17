"""
Long-term memory backend: User Profile KV store.
Dùng dict + JSON file persistence.
Conflict handling: fact mới luôn override fact cũ.
"""
import json
import os
from typing import Any, Dict, Optional


PROFILE_FILE = "data/user_profile.json"


class UserProfileMemory:
    """
    Long-term user profile memory.
    Lưu facts về user: name, preferences, allergies, occupation, ...
    Conflict: overwrite — giá trị mới luôn được ưu tiên.
    """

    def __init__(self, profile_path: str = PROFILE_FILE):
        self.profile_path = profile_path
        os.makedirs(os.path.dirname(profile_path), exist_ok=True)
        self._profile: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.profile_path):
            with open(self.profile_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)
        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(self._profile, f, ensure_ascii=False, indent=2)

    def update_fact(self, key: str, value: Any) -> None:
        """
        Cập nhật một fact trong profile.
        Nếu key đã tồn tại, overwrite (conflict resolution: new > old).
        """
        old_value = self._profile.get(key)
        self._profile[key] = value
        self._save()
        if old_value is not None and old_value != value:
            print(f"[LongTermMemory] Conflict resolved: '{key}' changed from '{old_value}' → '{value}'")

    def get_fact(self, key: str, default: Any = None) -> Any:
        """Lấy một fact từ profile."""
        return self._profile.get(key, default)

    def get_profile(self) -> Dict[str, Any]:
        """Trả về toàn bộ profile dưới dạng dict."""
        return dict(self._profile)

    def get_profile_text(self) -> str:
        """Chuyển profile thành text để inject vào prompt."""
        if not self._profile:
            return "(Chưa có thông tin người dùng)"
        lines = []
        for k, v in self._profile.items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    def update_from_dict(self, facts: Dict[str, Any]) -> None:
        """Cập nhật nhiều facts cùng lúc."""
        for key, value in facts.items():
            self.update_fact(key, value)

    def clear(self) -> None:
        """Xóa toàn bộ profile (dùng cho testing)."""
        self._profile = {}
        self._save()

    def __repr__(self) -> str:
        return f"UserProfileMemory(facts={list(self._profile.keys())})"
