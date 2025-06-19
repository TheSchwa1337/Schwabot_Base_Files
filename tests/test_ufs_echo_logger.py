import json
import pytest
from core.ufs_echo_logger import UFSEchoLogger


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "ufs_echo_test.jsonl"


def test_log_cluster_memory_creates_and_appends(log_path):
    logger = UFSEchoLogger(log_path=str(log_path))
    logger.log_cluster_memory("C1", "S1", 0.42)
    logger.log_cluster_memory("C2", "S2", 0.87)

    lines = log_path.read_text().splitlines()
    assert len(lines) == 2
    data1 = json.loads(lines[0])
    assert data1['cluster_id'] == "C1"
    assert 'timestamp' in data1