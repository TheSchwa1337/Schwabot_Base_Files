import logging
from core.bus_events import create_event
from core.cyclic_core import CyclicCore

class AdvancedControlPanel:
    def batch_adjust(self, nccos, signals):
        try:
            # Validate inputs
            if not isinstance(nccos, list) or len(nccos) == 0:
                raise ValueError("nccos must be a non-empty list.")
            
            if not isinstance(signals, dict) or len(signals) == 0:
                raise ValueError("signals must be a non-empty dictionary.")
            
            # Adjust scores based on market signals
            for ncco in nccos:
                if 'score' not in ncco:
                    raise KeyError("Each NCO must have a 'score' attribute.")
                
                score = ncco['score']
                adjusted_score = score * (1 + signals.get(ncco['symbol'], 0))
                ncco['score'] = adjusted_score
            
            return nccos
        
        except ValueError as ve:
            print(f"ValueError: {ve}")
            # Log the error
            logging.error(f"ValueError: {ve}")
        
        except KeyError as ke:
            print(f"KeyError: {ke}")
            # Log the error
            logging.error(f"KeyError: {ke}")
        
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            # Log the error
            logging.error(f"An unexpected error occurred: {e}")

# Example usage:
"""
from ncco_core.control_panel import AdvancedControlPanel

# Initialize the control panel
control_panel = AdvancedControlPanel()

# Example NCCOs and signals
nccos = [
    {'symbol': 'BTC', 'score': 100},
    {'symbol': 'ETH', 'score': 200}
]

signals = {
    'BTC': 0.05,
    'ETH': -0.03
}

# Adjust scores based on market signals
adjusted_nccos = control_panel.batch_adjust(nccos, signals)

# Print the adjusted NCCOs
for ncco in adjusted_nccos:
    print(f"Adjusted score: {ncco['score']}")
""" 

# Initialize the cyclic core
core = CyclicCore() 