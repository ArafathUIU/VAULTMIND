"""Tests for FastAPI endpoints."""


def test_api_modules_import() -> None:
    """Verify API scaffold modules are importable."""
    import api.dependencies
    import api.main
    import api.routes.health
    import api.routes.query
    import api.routes.upload
    import api.schemas

    assert api.main is not None
