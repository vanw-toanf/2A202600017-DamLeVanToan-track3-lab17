"""
Memory Router: phân tích intent của query để chọn memory type phù hợp.
Sử dụng keyword-based routing với LLM-assisted fallback.

Intent categories:
    - profile_recall: hỏi về thông tin cá nhân ("tên tôi là gì", "tôi dị ứng gì")
    - episodic_recall: hỏi về sự kiện đã xảy ra ("lần trước tôi", "hôm qua chúng ta")
    - semantic: hỏi về kiến thức factual ("FAQ", "cách làm", "giải thích")
    - general: câu hỏi/trò chuyện thông thường
    - update_profile: user đang cung cấp thông tin cá nhân mới
    - update_episodic: user report một task/outcome đã hoàn thành
"""
import re
from typing import List


# Keyword patterns cho từng intent
PROFILE_RECALL_PATTERNS = [
    r"tên (tôi|mình|của tôi)",
    r"tôi (là|làm|thích|dị ứng|ghét|sống ở)",
    r"(tên|tuổi|nghề|sở thích|dị ứng|địa chỉ) của tôi",
    r"(bạn có biết|bạn nhớ).*(tôi|mình)",
    r"(what is|what's) my",
    r"remember my",
    r"tôi đã nói.*(tên|dị ứng|thích|nghề)",
]

EPISODIC_RECALL_PATTERNS = [
    r"lần trước",
    r"hôm (qua|trước)",
    r"trước đây (tôi|chúng ta|mình)",
    r"(chúng ta|tôi|mình) (đã|từng) (nói|làm|thảo luận|học)",
    r"buổi (trước|hôm qua)",
    r"session trước",
    r"nhớ lại",
    r"recall.*(last|previous|before)",
    r"previous (conversation|session|chat)",
]

SEMANTIC_PATTERNS = [
    r"(cách|làm thế nào|how to)",
    r"(giải thích|explain|what is|là gì)",
    r"(hướng dẫn|guide|tutorial|instructions?)",
    r"(định nghĩa|definition|meaning)",
    r"FAQ",
    r"(tài liệu|documentation|docs)",
    r"(quy trình|process|procedure|workflow)",
]

UPDATE_PROFILE_PATTERNS = [
    r"(tên tôi là|tôi tên là|mình tên là)",
    r"tôi (làm nghề|là|học|sống ở|thích|ghét|dị ứng|không thích)",
    r"(cho bạn biết|cập nhật).*(tôi|thông tin)",
    r"(tôi|mình) (bị|có) dị ứng",
    r"à nhầm.*tôi",
    r"thực ra (tôi|mình)",
    r"sửa lại.*(tôi|thông tin)",
]

UPDATE_EPISODIC_PATTERNS = [
    r"(tôi|mình) (vừa|đã|xong) (làm|hoàn thành|giải quyết|fix|sửa)",
    r"(task|công việc|bug|issue|vấn đề) (đã|vừa) (xong|hoàn thành|giải quyết)",
    r"(deployed|deploy xong|chạy được|hoạt động rồi)",
]


def classify_intent(query: str) -> List[str]:
    """
    Phân loại intent của query.
    Returns list các intent (có thể nhiều intent cùng lúc).
    
    Priority order: update_profile > update_episodic > profile_recall > episodic_recall > semantic > general
    """
    query_lower = query.lower()
    intents = []

    # Check update patterns first
    for pattern in UPDATE_PROFILE_PATTERNS:
        if re.search(pattern, query_lower):
            if "update_profile" not in intents:
                intents.append("update_profile")
            break

    for pattern in UPDATE_EPISODIC_PATTERNS:
        if re.search(pattern, query_lower):
            if "update_episodic" not in intents:
                intents.append("update_episodic")
            break

    # Check recall patterns
    for pattern in PROFILE_RECALL_PATTERNS:
        if re.search(pattern, query_lower):
            if "profile_recall" not in intents:
                intents.append("profile_recall")
            break

    for pattern in EPISODIC_RECALL_PATTERNS:
        if re.search(pattern, query_lower):
            if "episodic_recall" not in intents:
                intents.append("episodic_recall")
            break

    for pattern in SEMANTIC_PATTERNS:
        if re.search(pattern, query_lower):
            if "semantic" not in intents:
                intents.append("semantic")
            break

    # Default: general conversation
    if not intents:
        intents.append("general")

    return intents


def should_retrieve_memory(intents: List[str], memory_type: str) -> bool:
    """
    Kiểm tra có nên retrieve memory_type hay không dựa vào intents.
    
    Memory retrieval mapping:
        short_term: luôn retrieve (context mới nhất)
        profile: profile_recall, update_profile
        episodic: episodic_recall, update_episodic
        semantic: semantic
    """
    intent_to_memory = {
        "short_term": True,  # luôn include short-term
        "profile": {"profile_recall", "update_profile", "general"},
        "episodic": {"episodic_recall", "update_episodic"},
        "semantic": {"semantic"},
    }
    
    mapping = intent_to_memory.get(memory_type, set())
    if isinstance(mapping, bool):
        return mapping
    
    return bool(set(intents) & mapping)


def format_memory_budget_summary(
    profile_tokens: int,
    episodic_tokens: int,
    semantic_tokens: int,
    short_term_tokens: int,
    total_budget: int = 2000,
) -> str:
    """Tóm tắt token budget allocation."""
    total_used = profile_tokens + episodic_tokens + semantic_tokens + short_term_tokens
    remaining = total_budget - total_used
    return (
        f"Token Budget: {total_used}/{total_budget} used "
        f"(profile={profile_tokens}, episodic={episodic_tokens}, "
        f"semantic={semantic_tokens}, recent={short_term_tokens}, "
        f"remaining={remaining})"
    )
