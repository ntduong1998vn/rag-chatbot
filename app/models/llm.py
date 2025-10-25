# app/models/llm.py
from typing import Dict, List, Optional

from google import genai
from google.genai import types


class LocalLLM:
    """
    LLM dùng Google Gemini 2.5 Flash-Lite qua SDK google-genai.
    - Mặc định dùng Developer API bằng GEMINI_API_KEY.
    Giữ nguyên interface .chat(...) như phiên bản Qwen trước đó.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        api_key: Optional[str] = None,
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _to_contents(self, messages: List[Dict[str, str]]):
        """
        Map history của app sang định dạng Gemini:
        - 'user'  -> role='user'
        - 'assistant' -> role='model'
        - 'system' -> gom vào system_instruction (xử lý riêng)
        """
        contents = []
        sys_extras = []

        for m in messages:
            role = m.get("role", "user")
            text = str(m.get("content", ""))

            if role == "system":
                sys_extras.append(text)
                continue

            g_role = "user" if role == "user" else "model"
            contents.append(
                types.Content(role=g_role, parts=[types.Part.from_text(text=text)])
            )

        return contents, "\n\n".join(sys_extras) if sys_extras else None

    def chat(
        self,
        system: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 768,
        temperature: float = 0.3,
    ) -> str:
        contents, extra_sys = self._to_contents(messages)
        sys_instruction = system if not extra_sys else f"{system}\n\n{extra_sys}"

        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=sys_instruction,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        # SDK trả về .text tiện lợi
        return getattr(resp, "text", "").strip()
