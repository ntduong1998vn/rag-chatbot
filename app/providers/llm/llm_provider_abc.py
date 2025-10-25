from abc import ABC, abstractmethod
from typing import Dict, List


class LLMProviderABC(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def chat(
        self,
        system: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 768,
        temperature: float = 0.3,
    ) -> str:
        """
        Generate a chat response

        Args:
            system: System prompt/instruction
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)

        Returns:
            Generated text response as string
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Get the name of the LLM model

        Returns:
            String representing the model name
        """
        pass