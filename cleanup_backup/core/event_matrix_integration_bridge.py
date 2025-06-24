#!/usr/bin/env python3
"""Event-Matrix Integration Bridge - Schwabot Framework.

This module bridges the event impact mapper with matrix controllers and Ferris wheel
systems, ensuring proper event processing and state updates. It maintains the
non-relativistic, profit-focused trading logic while providing seamless integration
between external events and internal matrix controller states.

Key Functions:
- Process events and update matrix controller state
- Calculate event confidence impact on trading decisions
- Update Ferris wheel based on event significance
- Validate event-matrix consistency and reliability
- Maintain event processing history and analytics

Mathematical Foundation:
Matrix_State_t+1 = g(Matrix_State_t, Event_Impact_t, Confidence_t)

Where:
- Matrix_State_t = Current matrix controller state
- Event_Impact_t = Impact of external events
- Confidence_t = Current confidence level
- g() = Matrix state transition function

Flake8 compliant with comprehensive type hints and error handling.
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

# Import core components
try:
    from core.event_impact_mapper import EventImpact, EventImpactMapper
    from core.unified_confidence_matrix import UnifiedConfidenceMatrix
    from core.type_defs import MatrixController, BitLevel, MatrixPhase
except ImportError as e:
    logging.warning(f"Some imports failed: {e}")
    # Fallback type definitions
    from typing import Any as EventImpact
    from typing import Any as MatrixController
    from enum import Enum as BitLevel
    from enum import Enum as MatrixPhase

logger = logging.getLogger(__name__)


class EventProcessingStatus(Enum):
    """Status of event processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    IGNORED = "ignored"


@dataclass
class EventMatrixResult:
    """Result of event-matrix integration processing."""
    event_id: str
    processing_status: EventProcessingStatus
    matrix_state_before: Dict[str, Any]
    matrix_state_after: Dict[str, Any]
    ferris_wheel_position_before: int
    ferris_wheel_position_after: int
    confidence_impact: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class EventProcessingMetrics:
    """Metrics for event processing performance."""
    total_events_processed: int
    successful_events: int
    failed_events: int
    ignored_events: int
    average_processing_time: float
    total_confidence_impact: float
    matrix_state_changes: int
    ferris_wheel_updates: int


class EventMatrixIntegrationBridge:
    """Bridge between event impact mapper and matrix controllers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the event-matrix integration bridge."""
        self.config = config or self._default_config()
        
        # Initialize components
        try:
            self.event_mapper = EventImpactMapper()
            self.confidence_matrix = UnifiedConfidenceMatrix()
        except Exception as e:
            logger.warning(f"Failed to initialize components: {e}")
            self.event_mapper = None
            self.confidence_matrix = None
        
        # Event processing state
        self.processing_queue: List[EventImpact] = []
        self.processing_history: List[EventMatrixResult] = []
        self.current_matrix_state: Dict[str, Any] = {
            'bit_level': '8bit',
            'phase': 'ACCUM',
            'confidence_score': 0.75,
            'fallback_triggered': False,
            'last_update': time.time()
        }
        self.current_ferris_wheel_position = 0
        
        # Performance tracking
        self.metrics = EventProcessingMetrics(
            total_events_processed=0,
            successful_events=0,
            failed_events=0,
            ignored_events=0,
            average_processing_time=0.0,
            total_confidence_impact=0.0,
            matrix_state_changes=0,
            ferris_wheel_updates=0
        )
        
        # Event filtering and prioritization
        self.event_filters = {
            'min_priority': self.config.get('min_event_priority', 3),
            'max_age_hours': self.config.get('max_event_age_hours', 24),
            'required_sources': self.config.get('required_sources', ['news_api', 'market_data']),
            'excluded_tags': self.config.get('excluded_tags', ['spam', 'test'])
        }
        
        logger.info("🌉 Event-Matrix Integration Bridge initialized")
    
    def process_event_with_matrix_impact(self,
                                       event_data: Dict[str, Any],
                                       matrix_controller: Optional[Dict[str, Any]] = None,
                                       ferris_wheel_position: Optional[int] = None) -> EventMatrixResult:
        """Process event and update matrix controller state.
        
        Args:
            event_data: Event data to process
            matrix_controller: Current matrix controller state (optional)
            ferris_wheel_position: Current Ferris wheel position (optional)
            
        Returns:
            EventMatrixResult with processing details
        """
        start_time = time.time()
        
        try:
            # Create event impact object
            event_impact = self._create_event_impact(event_data)
            
            # Validate event
            if not self._validate_event(event_impact):
                return EventMatrixResult(
                    event_id=event_impact.event_id,
                    processing_status=EventProcessingStatus.IGNORED,
                    matrix_state_before=self.current_matrix_state.copy(),
                    matrix_state_after=self.current_matrix_state.copy(),
                    ferris_wheel_position_before=self.current_ferris_wheel_position,
                    ferris_wheel_position_after=self.current_ferris_wheel_position,
                    confidence_impact=0.0,
                    processing_time=time.time() - start_time,
                    metadata={'reason': 'Event validation failed'}
                )
            
            # Store initial states
            matrix_state_before = (matrix_controller or self.current_matrix_state).copy()
            ferris_wheel_position_before = ferris_wheel_position or self.current_ferris_wheel_position
            
            # Calculate event confidence impact
            confidence_impact = self._calculate_event_confidence_impact(event_impact, matrix_state_before)
            
            # Update matrix controller state
            matrix_state_after = self._update_matrix_controller_with_event(
                matrix_state_before, event_impact, confidence_impact
            )
            
            # Update Ferris wheel position
            ferris_wheel_position_after = self._update_ferris_wheel_with_event(
                ferris_wheel_position_before, event_impact
            )
            
            # Update current states
            self.current_matrix_state = matrix_state_after.copy()
            self.current_ferris_wheel_position = ferris_wheel_position_after
            
            # Create result
            processing_time = time.time() - start_time
            result = EventMatrixResult(
                event_id=event_impact.event_id,
                processing_status=EventProcessingStatus.COMPLETED,
                matrix_state_before=matrix_state_before,
                matrix_state_after=matrix_state_after,
                ferris_wheel_position_before=ferris_wheel_position_before,
                ferris_wheel_position_after=ferris_wheel_position_after,
                confidence_impact=confidence_impact,
                processing_time=processing_time,
                metadata={
                    'event_priority': event_impact.priority,
                    'event_source': event_impact.source,
                    'event_tags': event_impact.tags,
                    'sentiment_score': event_impact.sentiment_score,
                    'relevance_score': event_impact.relevance_score
                }
            )
            
            # Update metrics
            self._update_metrics(result)
            
            # Store in history
            self.processing_history.append(result)
            
            # Maintain history size
            if len(self.processing_history) > self.config.get('max_history_size', 1000):
                self.processing_history = self.processing_history[-self.config.get('max_history_size', 1000):]
            
            logger.debug(f"Event {event_impact.event_id} processed successfully "
                        f"(confidence impact: {confidence_impact:.3f})")
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing event: {str(e)}"
            logger.error(error_msg)
            
            # Return error result
            return EventMatrixResult(
                event_id=event_data.get('event_id', 'unknown'),
                processing_status=EventProcessingStatus.FAILED,
                matrix_state_before=self.current_matrix_state.copy(),
                matrix_state_after=self.current_matrix_state.copy(),
                ferris_wheel_position_before=self.current_ferris_wheel_position,
                ferris_wheel_position_after=self.current_ferris_wheel_position,
                confidence_impact=0.0,
                processing_time=time.time() - start_time,
                error_message=error_msg
            )
    
    def calculate_event_confidence_impact(self, event_impact: EventImpact) -> float:
        """Calculate confidence impact of an event."""
        try:
            # Base impact from event priority
            priority_impact = event_impact.priority / 10.0
            
            # Sentiment impact
            sentiment_impact = abs(event_impact.sentiment_score) * 0.3
            
            # Relevance impact
            relevance_impact = event_impact.relevance_score * 0.2
            
            # Time decay factor
            time_diff = time.time() - event_impact.timestamp
            time_decay = np.exp(-time_diff / 3600)  # 1-hour decay
            
            # Source reliability factor
            source_reliability = {
                'news_api': 0.9,
                'market_data': 0.95,
                'social_media': 0.6,
                'unknown': 0.5
            }.get(event_impact.source, 0.7)
            
            # Calculate total impact
            total_impact = (priority_impact * 0.4 + 
                           sentiment_impact * 0.3 + 
                           relevance_impact * 0.2 + 
                           time_decay * 0.1) * source_reliability
            
            return max(0.0, min(1.0, total_impact))
            
        except Exception as e:
            logger.error(f"Error calculating event confidence impact: {e}")
            return 0.0
    
    def update_ferris_wheel_with_event(self, current_position: int, event_impact: EventImpact) -> int:
        """Update Ferris wheel position based on event."""
        try:
            # Calculate event significance
            significance = (event_impact.priority / 10.0 + 
                          abs(event_impact.sentiment_score) + 
                          event_impact.relevance_score) / 3.0
            
            # Determine position change based on significance
            if significance > 0.8:  # High significance
                position_change = 2
            elif significance > 0.6:  # Medium significance
                position_change = 1
            elif significance > 0.4:  # Low significance
                position_change = 0  # No change
            else:  # Very low significance
                position_change = -1  # Reverse direction
            
            # Apply position change
            new_position = (current_position + position_change) % 8
            
            # Ensure position is non-negative
            if new_position < 0:
                new_position = 7
            
            return new_position
            
        except Exception as e:
            logger.error(f"Error updating Ferris wheel with event: {e}")
            return current_position
    
    def validate_event_matrix_consistency(self, event_result: EventMatrixResult) -> bool:
        """Validate event-matrix consistency."""
        try:
            # Check that matrix state changed if event was significant
            if event_result.processing_status == EventProcessingStatus.COMPLETED:
                matrix_changed = (event_result.matrix_state_before != event_result.matrix_state_after)
                
                # For high-impact events, matrix should change
                if event_result.confidence_impact > 0.5:
                    if not matrix_changed:
                        logger.warning(f"High-impact event {event_result.event_id} didn't change matrix state")
                        return False
                
                # Check Ferris wheel consistency
                ferris_changed = (event_result.ferris_wheel_position_before != 
                                event_result.ferris_wheel_position_after)
                
                # For very high-impact events, Ferris wheel should change
                if event_result.confidence_impact > 0.8:
                    if not ferris_changed:
                        logger.warning(f"Very high-impact event {event_result.event_id} didn't change Ferris wheel")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating event-matrix consistency: {e}")
            return False
    
    def get_event_processing_metrics(self) -> Dict[str, Any]:
        """Get event processing metrics."""
        return {
            'total_events_processed': self.metrics.total_events_processed,
            'successful_events': self.metrics.successful_events,
            'failed_events': self.metrics.failed_events,
            'ignored_events': self.metrics.ignored_events,
            'average_processing_time': self.metrics.average_processing_time,
            'total_confidence_impact': self.metrics.total_confidence_impact,
            'matrix_state_changes': self.metrics.matrix_state_changes,
            'ferris_wheel_updates': self.metrics.ferris_wheel_updates,
            'success_rate': (self.metrics.successful_events / 
                           max(self.metrics.total_events_processed, 1)),
            'current_matrix_state': self.current_matrix_state,
            'current_ferris_wheel_position': self.current_ferris_wheel_position,
            'history_size': len(self.processing_history)
        }
    
    def _create_event_impact(self, event_data: Dict[str, Any]) -> EventImpact:
        """Create EventImpact object from event data."""
        try:
            return EventImpact(
                event_id=event_data.get('event_id', f"event_{int(time.time())}"),
                timestamp=event_data.get('timestamp', time.time()),
                source=event_data.get('source', 'unknown'),
                title=event_data.get('title', ''),
                content=event_data.get('content', ''),
                priority=event_data.get('priority', 5),
                tags=event_data.get('tags', []),
                sentiment_score=event_data.get('sentiment_score', 0.0),
                relevance_score=event_data.get('relevance_score', 0.5)
            )
        except Exception as e:
            logger.error(f"Error creating event impact: {e}")
            # Return default event impact
            return EventImpact(
                event_id=f"event_{int(time.time())}",
                timestamp=time.time(),
                source='unknown',
                title='',
                content='',
                priority=5,
                tags=[],
                sentiment_score=0.0,
                relevance_score=0.5
            )
    
    def _validate_event(self, event_impact: EventImpact) -> bool:
        """Validate event for processing."""
        try:
            # Check priority threshold
            if event_impact.priority < self.event_filters['min_priority']:
                return False
            
            # Check age threshold
            event_age_hours = (time.time() - event_impact.timestamp) / 3600
            if event_age_hours > self.event_filters['max_age_hours']:
                return False
            
            # Check source requirements
            if (self.event_filters['required_sources'] and 
                event_impact.source not in self.event_filters['required_sources']):
                return False
            
            # Check excluded tags
            for tag in event_impact.tags:
                if tag in self.event_filters['excluded_tags']:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating event: {e}")
            return False
    
    def _calculate_event_confidence_impact(self, event_impact: EventImpact,
                                          matrix_state: Dict[str, Any]) -> float:
        """Calculate event confidence impact."""
        try:
            # Use confidence matrix if available
            if self.confidence_matrix:
                confidence_result = self.confidence_matrix.calculate_unified_confidence(
                    event_impact=event_impact,
                    matrix_controller_state=matrix_state
                )
                return confidence_result.unified_confidence
            else:
                # Fallback calculation
                return self.calculate_event_confidence_impact(event_impact)
                
        except Exception as e:
            logger.error(f"Error calculating event confidence impact: {e}")
            return 0.0
    
    def _update_matrix_controller_with_event(self, matrix_state: Dict[str, Any],
                                            event_impact: EventImpact,
                                            confidence_impact: float) -> Dict[str, Any]:
        """Update matrix controller state based on event."""
        try:
            updated_state = matrix_state.copy()
            
            # Update confidence score
            current_confidence = matrix_state.get('confidence_score', 0.5)
            new_confidence = current_confidence * 0.7 + confidence_impact * 0.3
            updated_state['confidence_score'] = max(0.0, min(1.0, new_confidence))
            
            # Update phase based on event impact
            if confidence_impact > 0.8:
                updated_state['phase'] = 'CONV'
            elif confidence_impact > 0.6:
                updated_state['phase'] = 'RESON'
            elif confidence_impact > 0.4:
                updated_state['phase'] = 'ACCUM'
            else:
                updated_state['phase'] = 'DISP'
            
            # Update bit level based on event complexity
            if event_impact.priority > 8:
                updated_state['bit_level'] = '16bit'
            elif event_impact.priority > 6:
                updated_state['bit_level'] = '8bit'
            else:
                updated_state['bit_level'] = '4bit'
            
            # Update fallback status
            if confidence_impact < 0.2:
                updated_state['fallback_triggered'] = True
            
            # Update timestamp
            updated_state['last_update'] = time.time()
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Error updating matrix controller with event: {e}")
            return matrix_state
    
    def _update_ferris_wheel_with_event(self, current_position: int,
                                       event_impact: EventImpact) -> int:
        """Update Ferris wheel position based on event."""
        return self.update_ferris_wheel_with_event(current_position, event_impact)
    
    def _update_metrics(self, result: EventMatrixResult) -> None:
        """Update processing metrics."""
        try:
            self.metrics.total_events_processed += 1
            
            if result.processing_status == EventProcessingStatus.COMPLETED:
                self.metrics.successful_events += 1
            elif result.processing_status == EventProcessingStatus.FAILED:
                self.metrics.failed_events += 1
            elif result.processing_status == EventProcessingStatus.IGNORED:
                self.metrics.ignored_events += 1
            
            # Update average processing time
            total_time = self.metrics.average_processing_time * (self.metrics.total_events_processed - 1)
            total_time += result.processing_time
            self.metrics.average_processing_time = total_time / self.metrics.total_events_processed
            
            # Update confidence impact
            self.metrics.total_confidence_impact += result.confidence_impact
            
            # Update state changes
            if result.matrix_state_before != result.matrix_state_after:
                self.metrics.matrix_state_changes += 1
            
            if result.ferris_wheel_position_before != result.ferris_wheel_position_after:
                self.metrics.ferris_wheel_updates += 1
                
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'min_event_priority': 3,
            'max_event_age_hours': 24,
            'required_sources': ['news_api', 'market_data'],
            'excluded_tags': ['spam', 'test'],
            'max_history_size': 1000,
            'processing_timeout': 5.0  # 5 seconds
        }


# Global instance for easy access
event_matrix_bridge = EventMatrixIntegrationBridge()


def process_event_with_matrix_impact(event_data: Dict[str, Any],
                                    matrix_controller: Optional[Dict[str, Any]] = None,
                                    ferris_wheel_position: Optional[int] = None) -> EventMatrixResult:
    """Global function to process event with matrix impact."""
    return event_matrix_bridge.process_event_with_matrix_impact(
        event_data, matrix_controller, ferris_wheel_position
    )


def get_event_processing_metrics() -> Dict[str, Any]:
    """Global function to get event processing metrics."""
    return event_matrix_bridge.get_event_processing_metrics() 
