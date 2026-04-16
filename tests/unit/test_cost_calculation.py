import os
from unittest.mock import AsyncMock

os.environ.setdefault("UPSTREAM_BASE_URL", "http://test")
os.environ.setdefault("UPSTREAM_API_KEY", "test")

import pytest

from routstr.payment.cost_calculation import MaxCostData, calculate_cost  # noqa: E402


@pytest.mark.asyncio
async def test_calculate_cost_uses_reserved_max_cost_when_usage_missing() -> None:
    result = await calculate_cost(
        {"model": "gpt-5-search-api"},
        max_cost=1234,
        session=AsyncMock(),
    )

    assert isinstance(result, MaxCostData)
    assert result.base_msats == 1234
    assert result.total_msats == 1234


@pytest.mark.asyncio
async def test_calculate_cost_uses_reserved_max_cost_when_usage_none() -> None:
    result = await calculate_cost(
        {"model": "gpt-5-search-api", "usage": None},
        max_cost=2345,
        session=AsyncMock(),
    )

    assert isinstance(result, MaxCostData)
    assert result.base_msats == 2345
    assert result.total_msats == 2345


@pytest.mark.asyncio
async def test_calculate_cost_uses_reserved_max_cost_when_usage_has_no_billable_metrics() -> None:
    result = await calculate_cost(
        {"model": "gpt-5-search-api", "usage": {}},
        max_cost=3456,
        session=AsyncMock(),
    )

    assert isinstance(result, MaxCostData)
    assert result.base_msats == 3456
    assert result.total_msats == 3456
