import pytest
from core.vault_router import VaultRouter


@pytest.fixture
def router():
    return VaultRouter()


@pytest.mark.parametrize("strategy,expected", [
    ({'strategy': 'Tier3_HighProfit'}, 'Executed high-profit trade!'),
    ({'strategy': 'Tier2_MidProfit'}, 'Executed mid-profit trade.'),
    ({'strategy': 'Tier1_LowProfit'}, 'Executed low-profit trade.'),
    ({'strategy': 'Tier0_Observe'}, 'No trade executed.'),
])
def test_trigger_execution(router, strategy, expected):
    result = router.trigger_execution(strategy)
    assert expected in result