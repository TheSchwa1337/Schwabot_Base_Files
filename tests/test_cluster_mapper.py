import time
import pytest
from core.cluster_mapper import ClusterMapper

@pytest.fixture
def mapper():
    return ClusterMapper()

def test_form_cluster_creates_unique_id(mapper):
    node1 = mapper.form_cluster(0.5, 1.2, time.time())
    node2 = mapper.form_cluster(0.7, 0.8, time.time())
    assert node1['id'] != node2['id']
    assert 'entropy' in node1

def test_link_clusters_by_entropy_groups_similar(mapper):
    t = time.time()
    n1 = mapper.form_cluster(0.2, 0.2, t)
    n2 = mapper.form_cluster(0.21, 0.2, t + 1)
    echo1 = mapper.link_clusters_by_entropy(n1)
    echo2 = mapper.link_clusters_by_entropy(n2)
    assert echo1 == echo2
    n3 = mapper.form_cluster(0.9, 0.1, t + 2)
    echo3 = mapper.link_clusters_by_entropy(n3)
    assert echo3 != echo1

def test_get_echo_score_weights_by_recency(mapper):
    t = time.time()
    n1 = mapper.form_cluster(0.5, 1.0, t)
    mapper.link_clusters_by_entropy(n1)
    n1['timestamp'] = t - 10
    n1['drift'] = 1.0
    n2 = mapper.form_cluster(0.8, 0.9, t)
    mapper.link_clusters_by_entropy(n2)
    score = mapper.get_echo_score(n2)
    assert score > 0 and score < (n1['drift'] + n2['drift']) 