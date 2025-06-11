from typing import List, Dict, Callable, Any, TypeVar, Generic
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

T = TypeVar('T')  # Type variable for market states
S = TypeVar('S')  # Type variable for strategies

@dataclass
class MarketObject:
    """Represents a market state object in the category"""
    state: T
    features: Dict[str, Any]
    
@dataclass
class StrategyObject:
    """Represents a strategy object in the category"""
    strategy: S
    parameters: Dict[str, Any]
    
class Morphism(Generic[T, S]):
    """Represents a morphism (transformation) between objects"""
    def __init__(self, f: Callable[[T], S]):
        self.f = f
        
    def compose(self, other: 'Morphism') -> 'Morphism':
        """Compose this morphism with another"""
        return Morphism(lambda x: other.f(self.f(x)))
    
    def __call__(self, x: T) -> S:
        """Apply the morphism to an input"""
        return self.f(x)

class Category(ABC):
    """Abstract base class for categories"""
    @abstractmethod
    def identity(self, obj: Any) -> Morphism:
        """Return the identity morphism for an object"""
        pass
    
    @abstractmethod
    def compose(self, f: Morphism, g: Morphism) -> Morphism:
        """Compose two morphisms"""
        pass

class MarketCategory(Category):
    """Category of market states and transitions"""
    def identity(self, obj: MarketObject) -> Morphism:
        return Morphism(lambda x: x)
    
    def compose(self, f: Morphism, g: Morphism) -> Morphism:
        return f.compose(g)
    
    def transition(self, state1: MarketObject, state2: MarketObject) -> Morphism:
        """Create a morphism representing the transition between states"""
        return Morphism(lambda x: state2.state if x == state1.state else x)

class StrategyCategory(Category):
    """Category of trading strategies and transformations"""
    def identity(self, obj: StrategyObject) -> Morphism:
        return Morphism(lambda x: x)
    
    def compose(self, f: Morphism, g: Morphism) -> Morphism:
        return f.compose(g)
    
    def adapt(self, strategy1: StrategyObject, strategy2: StrategyObject) -> Morphism:
        """Create a morphism representing strategy adaptation"""
        return Morphism(lambda x: strategy2.strategy if x == strategy1.strategy else x)

class StrategyFunctor:
    """
    Functor mapping from MarketCategory to StrategyCategory
    Implements the natural transformation between market states and strategies
    """
    def __init__(self):
        self.market_category = MarketCategory()
        self.strategy_category = StrategyCategory()
        self.mapping: Dict[MarketObject, StrategyObject] = {}
        
    def map_object(self, market_obj: MarketObject) -> StrategyObject:
        """
        Map a market object to a strategy object
        Uses information geometric distance to find nearest known state
        """
        if market_obj in self.mapping:
            return self.mapping[market_obj]
            
        # Find nearest known market state
        nearest = self._find_nearest_state(market_obj)
        if nearest is None:
            return self._create_default_strategy(market_obj)
            
        return self.mapping[nearest]
    
    def map_morphism(
        self,
        market_morphism: Morphism[MarketObject, MarketObject]
    ) -> Morphism[StrategyObject, StrategyObject]:
        """
        Map a market morphism to a strategy morphism
        Ensures natural transformation property
        """
        def strategy_transform(strategy_obj: StrategyObject) -> StrategyObject:
            # Find corresponding market object
            market_obj = next(
                (m for m, s in self.mapping.items() if s == strategy_obj),
                None
            )
            if market_obj is None:
                return strategy_obj
                
            # Apply market morphism
            new_market = market_morphism(market_obj)
            
            # Map to new strategy
            return self.map_object(new_market)
            
        return Morphism(strategy_transform)
    
    def _find_nearest_state(self, market_obj: MarketObject) -> Optional[MarketObject]:
        """Find nearest known market state using information geometric distance"""
        if not self.mapping:
            return None
            
        # Compute distances to all known states
        distances = []
        for known_state in self.mapping.keys():
            dist = self._compute_state_distance(market_obj, known_state)
            distances.append((dist, known_state))
            
        # Return nearest state
        return min(distances, key=lambda x: x[0])[1] if distances else None
    
    def _compute_state_distance(
        self,
        state1: MarketObject,
        state2: MarketObject
    ) -> float:
        """Compute information geometric distance between states"""
        # Implement distance computation based on state features
        # This is a simplified version - in practice, use proper information metric
        feature_diff = np.array([
            abs(state1.features.get(k, 0) - state2.features.get(k, 0))
            for k in set(state1.features) | set(state2.features)
        ])
        return np.sqrt(np.sum(feature_diff ** 2))
    
    def _create_default_strategy(self, market_obj: MarketObject) -> StrategyObject:
        """Create a default strategy for a new market state"""
        # Implement default strategy creation logic
        return StrategyObject(
            strategy=lambda x: 0,  # Default to no action
            parameters={'type': 'default'}
        )
    
    def update_mapping(
        self,
        market_obj: MarketObject,
        strategy_obj: StrategyObject
    ) -> None:
        """Update the functor mapping with a new market-strategy pair"""
        self.mapping[market_obj] = strategy_obj
        
    def get_adjoint(self) -> 'StrategyFunctor':
        """
        Get the adjoint functor (if it exists)
        Maps from StrategyCategory back to MarketCategory
        """
        adjoint = StrategyFunctor()
        adjoint.mapping = {v: k for k, v in self.mapping.items()}
        return adjoint 