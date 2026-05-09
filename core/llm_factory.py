# core/llm_factory.py

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel

from core.config import settings, LLMProvider


def _require_api_key(provider: LLMProvider, api_key: str) -> None:
    if not api_key:
        env_name = f"{provider.value.upper()}_API_KEY"
        raise ValueError(f"{env_name} is required when LLM_PROVIDER={provider.value}")


def get_llm(
    provider: LLMProvider | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
) -> BaseChatModel:
    """
    Returns a LangChain chat model for the given provider.
    Falls back to settings defaults if arguments are not passed.

    Usage:
        llm = get_llm()                                  # uses .env defaults
        llm = get_llm(provider=LLMProvider.OPENAI)       # force a provider
        llm = get_llm(temperature=0.0, streaming=True)   # override params
    """

    provider = provider or settings.llm_provider
    temperature = temperature if temperature is not None else settings.llm_temperature
    max_tokens = max_tokens or settings.llm_max_tokens

    if provider == LLMProvider.OPENAI:
        _require_api_key(provider, settings.openai_api_key)
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
        )

    elif provider == LLMProvider.GEMINI:
        _require_api_key(provider, settings.gemini_api_key)
        return ChatGoogleGenerativeAI(
            google_api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    elif provider == LLMProvider.ANTHROPIC:
        _require_api_key(provider, settings.anthropic_api_key)
        return ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
        )

    elif provider == LLMProvider.GROQ:
        _require_api_key(provider, settings.groq_api_key)
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=temperature,
            max_tokens=max_tokens,
            # Groq supports streaming but has rate limits — keep False for agents
            streaming=streaming,
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_fast_llm() -> BaseChatModel:
    """
    Returns a lightweight, fast LLM for simple tasks.
    Used by the router agent — no need for heavy models there.
    """
    if settings.llm_provider == LLMProvider.OPENAI:
        _require_api_key(settings.llm_provider, settings.openai_api_key)
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_fast_model,
            temperature=0.0,
        )

    elif settings.llm_provider == LLMProvider.GEMINI:
        _require_api_key(settings.llm_provider, settings.gemini_api_key)
        return ChatGoogleGenerativeAI(
            google_api_key=settings.gemini_api_key,
            model=settings.gemini_fast_model,
            temperature=0.0,
        )

    elif settings.llm_provider == LLMProvider.ANTHROPIC:
        _require_api_key(settings.llm_provider, settings.anthropic_api_key)
        return ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_fast_model,
            temperature=0.0,
        )

    elif settings.llm_provider == LLMProvider.GROQ:
        _require_api_key(settings.llm_provider, settings.groq_api_key)
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_fast_model,
            temperature=0.0,
        )

    # fallback: just use the default
    return get_llm(temperature=0.0)


def get_critic_llm() -> BaseChatModel:
    """
    Returns a stricter LLM for the critic agent.
    Temperature 0.0 — no creativity, just fact checking.
    """
    return get_llm(temperature=0.0)
