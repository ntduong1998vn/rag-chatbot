from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple


class InMemoryStore:
    def __init__(self, max_turns: int = 12):
        self.store: Dict[str, Deque[Tuple[str, str]]] = defaultdict(
            lambda: deque(maxlen=max_turns)
        )

    def add(self, session_id: str, user_msg: str, assistant_msg: str):
        self.store[session_id].append(("user", user_msg))
        self.store[session_id].append(("assistant", assistant_msg))

    def get_context(self, session_id: str) -> List[dict]:
        return [{"role": r, "content": c} for (r, c) in list(self.store[session_id])]

    def clear(self, session_id: str):
        self.store.pop(session_id, None)
        self.store.pop(session_id, None)
