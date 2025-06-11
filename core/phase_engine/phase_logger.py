"""
Phase Logger Module
=================

Tracks and visualizes phase transitions for Schwabot's trading system.
Provides logging, persistence, and visualization capabilities for phase changes.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class PhaseTransition:
    """Represents a single phase transition event"""
    def __init__(
        self,
        timestamp: datetime,
        basket_id: str,
        from_phase: str,
        to_phase: str,
        urgency: float,
        metrics: Dict[str, float]
    ):
        self.timestamp = timestamp
        self.basket_id = basket_id
        self.from_phase = from_phase
        self.to_phase = to_phase
        self.urgency = urgency
        self.metrics = metrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert transition to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'basket_id': self.basket_id,
            'from_phase': self.from_phase,
            'to_phase': self.to_phase,
            'urgency': self.urgency,
            'metrics': self.metrics
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhaseTransition':
        """Create transition from dictionary"""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            basket_id=data['basket_id'],
            from_phase=data['from_phase'],
            to_phase=data['to_phase'],
            urgency=data['urgency'],
            metrics=data['metrics']
        )

class PhaseLogger:
    """Logs and visualizes phase transitions"""
    
    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize phase logger"""
        self.log_dir = log_dir or Path('logs/phase_transitions')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.transitions: List[PhaseTransition] = []
        
        # Set up logging
        self.log_file = self.log_dir / 'phase_transitions.log'
        self.handler = logging.FileHandler(self.log_file)
        self.handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(self.handler)

    def log_transition(
        self,
        basket_id: str,
        from_phase: str,
        to_phase: str,
        urgency: float,
        metrics: Dict[str, float]
    ) -> None:
        """Log a phase transition"""
        transition = PhaseTransition(
            timestamp=datetime.now(),
            basket_id=basket_id,
            from_phase=from_phase,
            to_phase=to_phase,
            urgency=urgency,
            metrics=metrics
        )
        
        self.transitions.append(transition)
        
        # Log to file
        logger.info(
            f"Phase transition: {basket_id} {from_phase} -> {to_phase} "
            f"(urgency: {urgency:.2f})"
        )
        
        # Save to JSON
        self._save_transitions()

    def _save_transitions(self) -> None:
        """Save transitions to JSON file"""
        json_file = self.log_dir / 'transitions.json'
        with open(json_file, 'w') as f:
            json.dump(
                [t.to_dict() for t in self.transitions],
                f,
                indent=2
            )

    def load_transitions(self) -> None:
        """Load transitions from JSON file"""
        json_file = self.log_dir / 'transitions.json'
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
                self.transitions = [PhaseTransition.from_dict(t) for t in data]

    def get_transition_dataframe(self) -> pd.DataFrame:
        """Convert transitions to pandas DataFrame"""
        return pd.DataFrame([
            {
                'timestamp': t.timestamp,
                'basket_id': t.basket_id,
                'from_phase': t.from_phase,
                'to_phase': t.to_phase,
                'urgency': t.urgency,
                **t.metrics
            }
            for t in self.transitions
        ])

    def plot_phase_transitions(
        self,
        basket_id: Optional[str] = None,
        metric: str = 'urgency',
        save_path: Optional[Path] = None
    ) -> None:
        """Plot phase transitions over time"""
        df = self.get_transition_dataframe()
        if basket_id:
            df = df[df['basket_id'] == basket_id]
            
        plt.figure(figsize=(12, 6))
        sns.set_style('whitegrid')
        
        # Plot transitions
        for phase in df['to_phase'].unique():
            phase_data = df[df['to_phase'] == phase]
            plt.scatter(
                phase_data['timestamp'],
                phase_data[metric],
                label=phase,
                alpha=0.7
            )
            
        plt.title(f'Phase Transitions - {metric.capitalize()}')
        plt.xlabel('Time')
        plt.ylabel(metric.capitalize())
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_phase_heatmap(
        self,
        save_path: Optional[Path] = None
    ) -> None:
        """Plot phase transition heatmap"""
        df = self.get_transition_dataframe()
        
        # Create transition matrix
        transitions = pd.crosstab(
            df['from_phase'],
            df['to_phase'],
            normalize='index'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            transitions,
            annot=True,
            fmt='.2%',
            cmap='YlOrRd'
        )
        
        plt.title('Phase Transition Probabilities')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def get_phase_statistics(self) -> Dict[str, Any]:
        """Calculate phase transition statistics"""
        df = self.get_transition_dataframe()
        
        stats = {
            'total_transitions': len(df),
            'unique_baskets': df['basket_id'].nunique(),
            'phase_counts': df['to_phase'].value_counts().to_dict(),
            'avg_urgency': df.groupby('to_phase')['urgency'].mean().to_dict(),
            'transition_matrix': pd.crosstab(
                df['from_phase'],
                df['to_phase'],
                normalize='index'
            ).to_dict()
        }
        
        return stats 