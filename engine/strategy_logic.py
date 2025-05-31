from core.mathlib_v2 import PsiStack, zeta_trigger
from core.mathlib import delta_time
from time_lattice_fork import TimeLatticeFork

# Initialize Time Lattice Fork
lattice_fork = TimeLatticeFork(
    rsi_period=14,
    entropy_threshold=0.5,
    ghost_window=16
)

def evaluate_tick(pkt: dict, rings) -> list:
    mid_price = pkt['mid_price']
    t_now = pkt['ts']
    volume = pkt.get('volume', 1.0)
    prev_mid = getattr(rings, 'prev_mid', mid_price)
    delta_mu = abs(mid_price - prev_mid) / prev_mid if prev_mid else 0.0

    # Process through Time Lattice Fork
    lattice_result = lattice_fork.process_tick(
        price=mid_price,
        volume=volume,
        timestamp=t_now
    )
    
    # Combine signals
    signals = []
    
    # Original PsiStack logic
    if zeta_trigger(delta_mu, band=(0.001, 0.005)):
        psi = PsiStack(rings.get('R3'), rings.get('R5'), rings.get('R8'), rings.get('R10'))
        decision = psi.collapse()
        signals.append(('BUY_FILL', mid_price, decision))
    
    # Time Lattice Fork signals
    lattice_signal = lattice_result['signal']
    if lattice_signal['action'] != 'HOLD' and lattice_signal['confidence'] > 0.6:
        signals.append((
            f"{lattice_signal['action']}_FILL",
            mid_price,
            lattice_signal['confidence']
        ))
    
    return signals 