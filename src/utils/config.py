"""
Configuration management for Agentic Search.
Supports both Groq (free) and OpenAI as LLM providers.
"""

import os
from pathlib import Path
from typing import Optional, Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Provider Selection
    llm_provider: Literal["groq", "openai"] = Field(
        default="groq", description="LLM provider to use (groq or openai)"
    )

    # Groq Configuration (FREE!)
    groq_api_key: str = Field(default="", description="Groq API key")
    groq_model: str = Field(
        default="llama-3.1-8b-instant", description="Groq model to use"
    )

    # OpenAI Configuration (paid, optional)
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model to use")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model"
    )

    # Embedding Configuration
    embedding_provider: Literal["huggingface", "openai"] = Field(
        default="huggingface", description="Embedding provider (huggingface is free)"
    )
    huggingface_embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="HuggingFace embedding model (free, local)"
    )

    # Tavily Configuration
    tavily_api_key: str = Field(default="", description="Tavily API key for web search")

    # ChromaDB Configuration
    chroma_persist_directory: str = Field(
        default="./data/chroma_db", description="ChromaDB persistence directory"
    )

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")

    # Agent Configuration
    max_reflection_iterations: int = Field(
        default=3, description="Maximum reflection iterations"
    )
    quality_threshold: float = Field(
        default=0.7, description="Quality threshold for answers (0-1)"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def validate_api_keys(self) -> dict[str, bool]:
        """Validate that required API keys are set."""
        return {
            "groq": bool(self.groq_api_key),
            "openai": bool(self.openai_api_key),
            "tavily": bool(self.tavily_api_key),
        }

    def get_llm_api_key(self) -> str:
        """Get the API key for the selected LLM provider."""
        if self.llm_provider == "groq":
            return self.groq_api_key
        return self.openai_api_key

    def get_llm_model(self) -> str:
        """Get the model name for the selected LLM provider."""
        if self.llm_provider == "groq":
            return self.groq_model
        return self.openai_model

    @property
    def chroma_path(self) -> Path:
        """Get ChromaDB path as Path object."""
        return Path(self.chroma_persist_directory)


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
