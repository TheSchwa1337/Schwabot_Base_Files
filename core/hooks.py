import os
from core.ncco_manager import NCCOManager
from core.sfsss_router import SFSSSRouter
from core.cluster_mapper import ClusterMapper
from core.drift_shell_engine import DriftShellEngine
from core.ufs_echo_logger import UFSEchoLogger
from core.vault_router import VaultRouter

DEBUG_CLUSTERS = os.getenv("DEBUG_CLUSTERS", "0") == "1"
DEBUG_DRIFTS   = os.getenv("DEBUG_DRIFTS", "0") == "1"
SIMULATE_STRAT = os.getenv("SIMULATE_STRATEGY", "0") == "1"

# Initialize singletons
ncco_manager   = NCCOManager()
sfsss_router   = SFSSSRouter()
cluster_mapper = ClusterMapper()
drift_engine   = DriftShellEngine()
echo_logger    = UFSEchoLogger()
vault_router   = VaultRouter() 