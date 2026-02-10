"""
LLM factory for creating the appropriate LLM based on configuration.
Supports Groq (free) and OpenAI (paid).
"""

from langchain_core.language_models import BaseChatModel

from src.utils.config import get_settings


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Get the configured LLM instance.
    
    Args:
        temperature: Temperature for generation (0.0 = deterministic)
    
    Returns:
        Configured LLM instance (ChatGroq or ChatOpenAI)
    """
    settings = get_settings()
    
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq
        
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=temperature,
        )
    else:
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=temperature,
        )


def get_llm_info() -> dict:
    """Get information about the current LLM configuration."""
    settings = get_settings()
    
    return {
        "provider": settings.llm_provider,
        "model": settings.get_llm_model(),
        "has_api_key": bool(settings.get_llm_api_key()),
    }
