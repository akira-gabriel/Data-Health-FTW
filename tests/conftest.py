import pytest


@pytest.fixture(scope="session")
def spark():
    from databricks.connect import DatabricksSession

    session = DatabricksSession.builder.profile("data-health").serverless(True).getOrCreate()
    yield session