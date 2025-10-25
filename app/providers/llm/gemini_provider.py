from typing import Dict, List, Optional

from google import genai
from google.genai import types

from .llm_provider_abc import LLMProviderABC


class GeminiProvider(LLMProviderABC):
    """Gemini LLM provider implementation"""

    def __init__(self, model: str = "gemini-2.5-flash-lite", api_key: str = None):
        self._model_name = model
        self.client = genai.Client(api_key=api_key)

    def _to_contents(self, messages: List[Dict[str, str]]):
        """
        Map history of app to Gemini format:
        - 'user' -> role='user'
        - 'assistant' -> role='model'
        - 'system' -> group into system_instruction (handled separately)
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
        """Generate a chat response"""
        contents, extra_sys = self._to_contents(messages)
        sys_instruction = system if not extra_sys else f"{system}\n\n{extra_sys}"

        resp = self.client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=sys_instruction,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        # SDK returns .text conveniently
        return getattr(resp, "text", "").strip()

    @property
    def model_name(self) -> str:
        """Get the name of the LLM model"""
        return self._model_name