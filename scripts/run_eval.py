"""CLI entry point for running the evaluation suite."""

from core.config import settings


def main() -> None:
    """Run the evaluation pipeline."""
    print("VaultMind evaluation smoke check")
    print(f"LLM provider: {settings.llm_provider.value}")
    print(f"Embedding provider: {settings.embedding_provider.value}")
    print(f"Chunk size: {settings.chunk_size}")
    print(f"Chunk overlap: {settings.chunk_overlap}")
    print("Full RAGAS/DeepEval suites will plug in here once datasets are available.")


if __name__ == "__main__":
    main()
