"""
Multi-Memory LangGraph Agent.
5 nodes: retrieve_memory → route_memory → build_prompt → call_llm → save_memory

MemoryState chứa toàn bộ thông tin cần thiết để xử lý một turn.
"""
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from memory.short_term import ConversationBufferMemory
from memory.long_term import UserProfileMemory
from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory
from agent.memory_router import classify_intent, should_retrieve_memory, format_memory_budget_summary
from agent.llm_client import call_llm, extract_facts_from_text, estimate_tokens


# ── State definition ──────────────────────────────────────────────────────────

class MemoryState(TypedDict):
    """Shared state xuyên suốt graph."""
    # Input
    user_input: str
    conversation_id: str

    # Retrieved memory
    messages: List[Dict]          # short-term conversation history
    user_profile: Dict            # long-term profile facts
    episodes: List[Dict]          # episodic memory hits
    semantic_hits: List[str]      # semantic search results
    memory_budget: int            # token budget still available

    # Routing
    intents: List[str]

    # Prompt & LLM
    system_prompt: str
    llm_messages: List[Dict]

    # Output
    response: str
    prompt_tokens: int
    completion_tokens: int


# ── Memory singletons (shared across agent lifecycle) ─────────────────────────

_short_term: Optional[ConversationBufferMemory] = None
_long_term: Optional[UserProfileMemory] = None
_episodic: Optional[EpisodicMemory] = None
_semantic: Optional[SemanticMemory] = None


def get_memories():
    global _short_term, _long_term, _episodic, _semantic
    if _short_term is None:
        _short_term = ConversationBufferMemory(max_turns=10)
    if _long_term is None:
        _long_term = UserProfileMemory()
    if _episodic is None:
        _episodic = EpisodicMemory()
    if _semantic is None:
        _semantic = SemanticMemory()
        _seed_semantic(_semantic)
    return _short_term, _long_term, _episodic, _semantic


def _seed_semantic(sem: SemanticMemory):
    """Seed một số FAQ documents vào semantic store."""
    faqs = [
        ("faq_docker", "Khi dùng Docker, service name được dùng thay cho localhost. Ví dụ: postgres thay vì localhost:5432."),
        ("faq_env", "File .env chứa biến môi trường nhạy cảm, không được commit lên git."),
        ("faq_langgraph", "LangGraph là framework của LangChain để build stateful multi-step agents dùng directed graph."),
        ("faq_chromadb", "ChromaDB là vector database dùng để lưu embeddings và thực hiện semantic search."),
        ("faq_rag", "RAG (Retrieval Augmented Generation) là kỹ thuật kết hợp retrieval từ knowledge base với LLM generation."),
        ("faq_memory", "Bộ nhớ agent gồm 4 loại: short-term (ngắn hạn), long-term profile, episodic (sự kiện), semantic (kiến thức)."),
        ("faq_conflict", "Conflict handling trong long-term memory: giá trị mới luôn override giá trị cũ của cùng một key."),
        ("faq_token_budget", "Token budget giới hạn lượng context đưa vào prompt để tránh vượt context window của LLM."),
    ]
    for doc_id, text in faqs:
        sem.add_document(doc_id, text, {"source": "faq"})


# ── Node 1: retrieve_memory ───────────────────────────────────────────────────

def retrieve_memory_node(state: MemoryState) -> MemoryState:
    """
    Gom memory từ 4 backends vào state.
    Luôn lấy short-term. Lấy thêm profile/episodic/semantic tuỳ intent.
    """
    short_term, long_term, episodic, semantic = get_memories()
    user_input = state["user_input"]
    intents = state.get("intents", classify_intent(user_input))

    # Short-term: luôn lấy
    messages = short_term.get_recent()

    # Long-term profile: luôn cần để hiểu user
    user_profile = long_term.get_profile()

    # Episodic: khi recall episodic hoặc update
    if should_retrieve_memory(intents, "episodic"):
        episodes = episodic.search_episodes(user_input, top_k=3)
        if not episodes:
            episodes = episodic.get_recent_episodes(3)
    else:
        episodes = episodic.get_recent_episodes(2)

    # Semantic: khi query về kiến thức
    if should_retrieve_memory(intents, "semantic"):
        semantic_hits = semantic.semantic_search(user_input, top_k=3)
    else:
        semantic_hits = []

    return {
        **state,
        "intents": intents,
        "messages": messages,
        "user_profile": user_profile,
        "episodes": episodes,
        "semantic_hits": semantic_hits,
        "memory_budget": 2000,
    }


# ── Node 2: route_memory_node ─────────────────────────────────────────────────

def route_memory_node(state: MemoryState) -> MemoryState:
    """
    Tính token budget và quyết định giữ/trim bao nhiêu memory.
    Ưu tiên: profile > recent conversation > semantic > episodic.
    """
    _, long_term, episodic, _ = get_memories()
    user_profile = state["user_profile"]
    episodes = state["episodes"]
    semantic_hits = state["semantic_hits"]
    messages = state["messages"]
    budget = state["memory_budget"]

    profile_text = long_term.get_profile_text()
    profile_tokens = estimate_tokens(profile_text)

    short_term_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    short_term_tokens = estimate_tokens(short_term_text)

    episodic_text = episodic.get_episodes_text(episodes)
    episodic_tokens = estimate_tokens(episodic_text)

    semantic_text = "\n".join(f"- {h}" for h in semantic_hits)
    semantic_tokens = estimate_tokens(semantic_text)

    # Trim nếu vượt budget
    used = profile_tokens + short_term_tokens + episodic_tokens + semantic_tokens
    if used > budget:
        # Trim episodic trước
        while episodic_tokens > 200 and used > budget:
            episodes = episodes[1:]
            episodic_text = episodic.get_episodes_text(episodes)
            episodic_tokens = estimate_tokens(episodic_text)
            used = profile_tokens + short_term_tokens + episodic_tokens + semantic_tokens

        # Trim semantic nếu vẫn cần
        while semantic_tokens > 200 and used > budget:
            semantic_hits = semantic_hits[:-1]
            semantic_text = "\n".join(f"- {h}" for h in semantic_hits)
            semantic_tokens = estimate_tokens(semantic_text)
            used = profile_tokens + short_term_tokens + episodic_tokens + semantic_tokens

    budget_summary = format_memory_budget_summary(
        profile_tokens, episodic_tokens, semantic_tokens, short_term_tokens
    )
    print(f"[Router] {budget_summary}")

    return {
        **state,
        "episodes": episodes,
        "semantic_hits": semantic_hits,
        "memory_budget": budget - used,
    }


# ── Node 3: build_prompt_node ─────────────────────────────────────────────────

def build_prompt_node(state: MemoryState) -> MemoryState:
    """
    Inject memory vào system prompt với 4 sections rõ ràng:
    [PROFILE] [EPISODIC] [KNOWLEDGE] [RECENT CONVERSATION]
    """
    _, long_term, episodic, _ = get_memories()
    user_profile = state["user_profile"]
    episodes = state["episodes"]
    semantic_hits = state["semantic_hits"]
    messages = state["messages"]
    intents = state["intents"]

    profile_text = long_term.get_profile_text()
    episodic_text = episodic.get_episodes_text(episodes)
    semantic_text = "\n".join(f"- {h}" for h in semantic_hits) if semantic_hits else "(Không có thông tin liên quan)"
    recent_text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages[-6:]) if messages else "(Chưa có lịch sử)"

    system_prompt = f"""Bạn là một trợ lý AI thông minh với bộ nhớ đa tầng. Hãy sử dụng các thông tin dưới đây để trả lời chính xác và tự nhiên bằng tiếng Việt.

[THÔNG TIN NGƯỜI DÙNG / PROFILE]
{profile_text}

[SỰ KIỆN ĐÃ XẢY RA / EPISODIC]
{episodic_text}

[KIẾN THỨC / SEMANTIC]
{semantic_text}

[LỊCH SỬ HỘI THOẠI GẦN ĐÂY / SHORT-TERM]
{recent_text}

---
Intent phát hiện: {', '.join(intents)}
Hãy trả lời câu hỏi của người dùng dựa trên ngữ cảnh trên. Nếu bạn nhớ thông tin từ profile hay lịch sử, hãy dùng nó."""

    llm_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["user_input"]},
    ]

    return {
        **state,
        "system_prompt": system_prompt,
        "llm_messages": llm_messages,
    }


# ── Node 4: call_llm_node ─────────────────────────────────────────────────────

def call_llm_node(state: MemoryState) -> MemoryState:
    """Gọi LLM và ghi nhận response + token counts."""
    response, prompt_tokens, completion_tokens = call_llm(state["llm_messages"])
    print(f"[LLM] tokens: prompt={prompt_tokens}, completion={completion_tokens}")
    return {
        **state,
        "response": response,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


# ── Rule-based extraction ─────────────────────────────────────────────────────

import re as _re


def _rule_based_extract(text: str) -> dict:
    """
    Extract profile facts nhanh bằng regex patterns.
    Dùng làm first-pass trước khi gọi LLM extraction.
    """
    facts = {}
    t = text.strip()

    # Name patterns: "tên tôi là X", "tôi tên là X", "mình tên là X"
    name_match = _re.search(
        r"(?:tên tôi là|tôi tên là|mình tên là|tôi là)\s+([A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠ-ỮẮẶẲẴỐ-ỘỚ-ỢỤỦỨỪ][a-zàáâãèéêìíòóôõùúăđĩũơưạ-ữắặẳẵố-ộớ-ợụủứừA-Z]+)",
        t, _re.IGNORECASE
    )
    if name_match:
        facts["name"] = name_match.group(1)

    # Occupation: "tôi là bác sĩ", "tôi làm nghề X", "tôi là kỹ sư"
    occ_match = _re.search(
        r"tôi (?:là|làm nghề|làm)\s+(bác sĩ|kỹ sư|lập trình viên|data scientist|giáo viên|giảng viên|nhà nghiên cứu|kế toán|nhân viên|quản lý|sinh viên|học sinh)",
        t, _re.IGNORECASE
    )
    if occ_match:
        facts["occupation"] = occ_match.group(1)

    # Allergy: "tôi dị ứng X", "tôi bị dị ứng X"
    allergy_match = _re.search(
        r"(?:tôi|mình) (?:bị |có )?dị ứng (?:với |)(.+?)(?:\.|,|$)",
        t, _re.IGNORECASE
    )
    if allergy_match:
        facts["allergy"] = allergy_match.group(1).strip()

    # Location: "tôi sống ở X", "tôi ở X"
    loc_match = _re.search(
        r"tôi (?:sống ở|ở|đang ở|chuyển (?:ra|đến|về))\s+(.+?)(?:\.|,|$)",
        t, _re.IGNORECASE
    )
    if loc_match:
        facts["location"] = loc_match.group(1).strip()

    return facts


# ── Node 5: save_memory_node ──────────────────────────────────────────────────

def save_memory_node(state: MemoryState) -> MemoryState:
    """
    Cập nhật 4 memory backends sau mỗi turn:
    1. Short-term: add current turn
    2. Long-term profile: extract & update facts nếu update_profile intent
    3. Episodic: ghi episode nếu có outcome rõ
    4. Semantic: không cần update (static FAQ)
    """
    short_term, long_term, episodic, semantic = get_memories()
    user_input = state["user_input"]
    response = state["response"]
    intents = state["intents"]

    # 1. Short-term: luôn lưu
    short_term.add_turn("user", user_input)
    short_term.add_turn("assistant", response)

    # 2. Long-term profile: nếu user đang cung cấp thông tin cá nhân
    if "update_profile" in intents:
        # Fast rule-based extraction trước
        facts = _rule_based_extract(user_input)
        # LLM extraction làm enrichment thêm
        llm_facts = extract_facts_from_text(user_input)
        facts.update({k: v for k, v in llm_facts.items() if v})
        if facts:
            long_term.update_from_dict(facts)
            print(f"[SaveMemory] Profile updated: {list(facts.keys())}")

    # 3. Episodic: ghi khi task hoàn thành hoặc có outcome rõ
    if "update_episodic" in intents:
        episodic.add_episode(
            summary=f"User: {user_input[:80]}",
            outcome="completed",
            tags=["task", "completed"],
        )
        print("[SaveMemory] Episode recorded.")

    # 4. Ghi episode tổng quát khi semantic search có kết quả
    if "semantic" in intents and state.get("semantic_hits"):
        episodic.add_episode(
            summary=f"Tìm kiếm thông tin về: {user_input[:60]}",
            outcome="retrieved",
            tags=["knowledge", "semantic"],
        )

    return state


# ── Build LangGraph ───────────────────────────────────────────────────────────

def build_graph():
    """Build và compile LangGraph StateGraph."""
    workflow = StateGraph(MemoryState)

    workflow.add_node("retrieve_memory", retrieve_memory_node)
    workflow.add_node("route_memory", route_memory_node)
    workflow.add_node("build_prompt", build_prompt_node)
    workflow.add_node("call_llm", call_llm_node)
    workflow.add_node("save_memory", save_memory_node)

    workflow.set_entry_point("retrieve_memory")
    workflow.add_edge("retrieve_memory", "route_memory")
    workflow.add_edge("route_memory", "build_prompt")
    workflow.add_edge("build_prompt", "call_llm")
    workflow.add_edge("call_llm", "save_memory")
    workflow.add_edge("save_memory", END)

    return workflow.compile()


# ── Public API ────────────────────────────────────────────────────────────────

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(user_input: str, conversation_id: str = "default") -> str:
    """
    Chạy agent với một user input.
    Trả về response text.
    """
    graph = get_graph()
    initial_state: MemoryState = {
        "user_input": user_input,
        "conversation_id": conversation_id,
        "messages": [],
        "user_profile": {},
        "episodes": [],
        "semantic_hits": [],
        "memory_budget": 2000,
        "intents": [],
        "system_prompt": "",
        "llm_messages": [],
        "response": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    result = graph.invoke(initial_state)
    return result["response"]


def reset_memories():
    """Reset tất cả memory backends (dùng cho benchmark/testing)."""
    global _short_term, _long_term, _episodic, _semantic, _graph
    if _short_term:
        _short_term.clear()
    if _long_term:
        _long_term.clear()
    if _episodic:
        _episodic.clear()
    if _semantic:
        _semantic.clear()
        _seed_semantic(_semantic)
    _graph = None
