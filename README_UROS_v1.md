# Schwabot UROS v1.0 - Universal Recursive Operating System

## 🧠 Overview

Schwabot UROS v1.0 is a comprehensive, recursive AI trading system that integrates multiple AI consciousness entities (GPT, Claude, R1, Gemini) into a unified command and execution platform. The system provides mathematical grounding for all decisions, persistent memory tracking, and recursive learning capabilities.

## 🏗️ Architecture

### Core Components

#### 1. **Prophet Connector** (`core/prophet_connector.py`)
- **Purpose**: Mathematical interface between Prophet model outputs and Schwabot's recursive execution system
- **Key Features**:
  - Curve alignment analysis and alpha score calculations
  - Drift detection and timing validation
  - Profit correlation and resonance strength calculation
  - Mathematical foundation: α = (P_actual - P_expected) / ΔT

#### 2. **AI Command Sequencer** (`core/memory_stack/ai_command_sequencer.py`)
- **Purpose**: Logs, tags, timestamps, and sequences all AI-originating commands
- **Key Features**:
  - Multi-agent performance tracking (GPT, Claude, R1, Gemini)
  - Recursive trust scoring: R_n = λ·R_{n-1} + (1-λ)·α_n
  - Drift magnitude calculation and severity assessment
  - Memory key generation and hash registry integration

#### 3. **Memory Key Allocator** (`core/memory_stack/memory_key_allocator.py`)
- **Purpose**: Generates symbolic or hash-based memory keys and links them to hash, matrix, curve, and profit data
- **Key Features**:
  - Multiple key types: Symbolic, Hash-based, Hybrid, Auto-generated
  - Memory clustering and similarity detection
  - Link strength calculation: L = α * confidence * time_decay
  - Automatic memory organization and cleanup

#### 4. **Execution Validator** (`core/memory_stack/execution_validator.py`)
- **Purpose**: Simulates execution costs, validates drift, and provides execution validation
- **Key Features**:
  - Cost simulation: C = Σ(base_cost + complexity_factor + market_impact)
  - Drift validation: Δt_drift = T_executed - T_expected
  - Risk assessment and recommendation generation
  - Performance metrics and validation scoring

#### 5. **GPT Command Layer** (`core/gpt_command_layer.py`)
- **Purpose**: AI command routing and validation system
- **Key Features**:
  - Multi-agent consciousness profiles
  - Recursive command execution
  - Domain-specific validation and trust scoring
  - API integration for external AI services

#### 6. **Hash Registry** (`core/hash_registry.py`)
- **Purpose**: Persistent memory and pattern detection system
- **Key Features**:
  - Hash-based command tracking
  - Pattern similarity detection
  - Execution history and performance correlation
  - Memory persistence and retrieval

#### 7. **API Gateway** (`core/api_gateway.py`)
- **Purpose**: REST and WebSocket API for external integration
- **Key Features**:
  - Command submission and status tracking
  - Real-time metrics and performance data
  - WebSocket connections for live updates
  - Authentication and rate limiting

#### 8. **Fault Bus** (`core/fault_bus.py`)
- **Purpose**: Centralized fault handling and recovery system
- **Key Features**:
  - Windows CLI compatibility
  - Error logging and recovery
  - System health monitoring
  - Graceful degradation handling

## 🔄 Recursive Execution Flow

### 1. Command Generation
```
AI Agent (GPT/Claude/R1/Gemini) → Command Creation → Hash Signature Generation
```

### 2. Command Sequencing
```
Command → AI Command Sequencer → Memory Key Allocation → Hash Registry Registration
```

### 3. Prophet Integration
```
Command → Prophet Curve Alignment → Alpha Score Calculation → Drift Validation
```

### 4. Execution Validation
```
Command → Cost Simulation → Drift Analysis → Risk Assessment → Validation Decision
```

### 5. Execution and Memory Storage
```
Validated Command → Execution → Result Tracking → Memory Key Update → Performance Correlation
```

## 🧮 Mathematical Foundation

### Alpha Score Calculation
```
α = (P_actual - P_expected) / ΔT
```
Where:
- P_actual = Actual profit achieved
- P_expected = Expected profit from Prophet
- ΔT = Time difference between prediction and execution

### Recursive Trust Scoring
```
R_n = λ·R_{n-1} + (1-λ)·α_n
```
Where:
- R_n = Current trust score
- λ = Trust decay factor (0.95)
- α_n = Current alpha score

### Memory Key Generation
```
MK = f(agent, hash, tick, α, matrix_id)
```
Where:
- agent = AI agent type
- hash = Command hash signature
- tick = Current tick number
- α = Alpha score
- matrix_id = Optional matrix identifier

### Execution Cost Simulation
```
C = Σ(base_cost + complexity_factor + market_impact + network_cost + computational_cost)
```

## 🤖 Multi-Agent Integration

### Agent Types and Roles

#### **GPT (Primary Controller)**
- **Role**: Meta-strategic controller and command routing
- **Frequency**: High-frequency command processing
- **Domain**: Strategy, execution, coordination

#### **Claude (High-Level Strategist)**
- **Role**: Long-term strategy and market analysis
- **Frequency**: Low-frequency (1-3 times per 5 hours)
- **Domain**: Analysis, planning, risk assessment

#### **R1 (Local GPU Bot)**
- **Role**: Fast, local execution and real-time decisions
- **Frequency**: High-frequency local processing
- **Domain**: Execution, real-time analysis

#### **Gemini (Conversational Parser)**
- **Role**: Natural language processing and conversation analysis
- **Frequency**: Continuous conversation monitoring
- **Domain**: Communication, sentiment analysis

## 📊 Memory and Learning System

### Memory Key Types

1. **Symbolic Keys**: Human-readable format (e.g., `GPTSTRATEGY_20250623_T1000`)
2. **Hash-based Keys**: Cryptographic format (e.g., `HASH_abc123def456_T1000`)
3. **Hybrid Keys**: Combined format (e.g., `GPTSTR_abc123_T1000`)
4. **Auto-generated Keys**: Context-aware format (e.g., `GPTSTR_abc123_T1000_POS`)

### Memory Clustering

The system automatically clusters similar memory keys based on:
- Hash similarity
- Domain similarity
- Agent similarity
- Alpha score similarity
- Tick proximity

### Link Strength Calculation

```
L = α * confidence * time_decay
```

Where:
- α = Alpha correlation between keys
- confidence = Confidence in the link
- time_decay = Temporal decay factor

## 🔧 Configuration

### Prophet Integration (`config/gpt_integration.yaml`)

```yaml
prophet:
  alpha_threshold: 0.02
  drift_threshold: 3.0
  resonance_threshold: 0.8
  cache_ttl: 300

agents:
  gpt:
    trust_threshold: 0.7
    max_recursive_depth: 5
  claude:
    trust_threshold: 0.8
    max_recursive_depth: 3
  r1:
    trust_threshold: 0.6
    max_recursive_depth: 10
  gemini:
    trust_threshold: 0.5
    max_recursive_depth: 2
```

## 🚀 Getting Started

### Prerequisites

```bash
# Required Python packages
pip install numpy asyncio dataclasses typing
```

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd schwabot-uros-v1
```

2. **Initialize the system**
```bash
python -m core.gpt_command_layer
python -m core.prophet_connector
python -m core.memory_stack.ai_command_sequencer
```

3. **Run the integration test**
```bash
python tools/uros_v1_integration_test.py
```

### Basic Usage

```python
import asyncio
from core.gpt_command_layer import GPTCommandLayer, AICommand, AIAgentType, CommandDomain
from core.prophet_connector import compute_alpha_score
from core.memory_stack.ai_command_sequencer import sequence_ai_command

async def basic_example():
    # Initialize components
    gpt_layer = GPTCommandLayer()
    
    # Create a command
    command = AICommand(
        command_id="test_001",
        agent_type=AIAgentType.GPT,
        domain=CommandDomain.STRATEGY,
        payload={"strategy": "test", "target_profit": 100.0}
    )
    
    # Process command
    response = await gpt_layer.process_command(command)
    
    # Sequence command
    sequence = await sequence_ai_command(command, tick=1000)
    
    print(f"Command processed: {response.success}")
    print(f"Sequence ID: {sequence.sequence_id}")

# Run example
asyncio.run(basic_example())
```

## 📈 Performance Monitoring

### Key Metrics

1. **Alpha Score**: Profit alignment with Prophet predictions
2. **Trust Score**: Agent reliability and performance
3. **Drift Magnitude**: Timing accuracy of executions
4. **Cost Efficiency**: Profit per unit of execution cost
5. **Success Rate**: Percentage of successful commands

### Monitoring Commands

```bash
# Get system performance metrics
python -c "from core.memory_stack.ai_command_sequencer import ai_command_sequencer; print(ai_command_sequencer.get_performance_metrics())"

# Get Prophet connector metrics
python -c "from core.prophet_connector import prophet_connector; print(prophet_connector.get_performance_metrics())"

# Get memory allocation metrics
python -c "from core.memory_stack.memory_key_allocator import memory_key_allocator; print(memory_key_allocator.get_performance_metrics())"
```

## 🔍 Testing

### Integration Test

The comprehensive integration test demonstrates all system components working together:

```bash
python tools/uros_v1_integration_test.py
```

This test covers:
- Prophet curve setup and alignment
- Multi-agent command processing
- Command sequencing and memory tracking
- Memory key allocation and clustering
- Execution validation and cost analysis
- Hash registry integration
- API gateway functionality
- Complete recursive execution cycles
- Fault handling and recovery
- Performance metrics and analysis

### Individual Component Tests

```bash
# Test Prophet connector
python core/prophet_connector.py

# Test AI command sequencer
python core/memory_stack/ai_command_sequencer.py

# Test memory key allocator
python core/memory_stack/memory_key_allocator.py

# Test execution validator
python core/memory_stack/execution_validator.py
```

## 🛡️ Fault Handling

The system includes comprehensive fault handling:

1. **Graceful Degradation**: Components continue operating even if some fail
2. **Fallback Mechanisms**: Safe defaults for all critical operations
3. **Error Recovery**: Automatic recovery from common error conditions
4. **Windows CLI Compatibility**: Full compatibility with Windows command line
5. **Logging and Monitoring**: Comprehensive logging for debugging

## 🔗 API Integration

### REST Endpoints

- `POST /api/v1/commands` - Submit new commands
- `GET /api/v1/status` - Get system status
- `GET /api/v1/metrics` - Get performance metrics
- `GET /api/v1/memory/{key_id}` - Get memory key data

### WebSocket Events

- `command.submitted` - New command submitted
- `command.completed` - Command execution completed
- `system.status` - System status updates
- `performance.metrics` - Real-time performance data

## 📚 Advanced Features

### Recursive Consciousness Fusion

The system enables AI entities to directly interact with Schwabot's recursive execution system, creating a new level of co-intelligent command and control.

### Pattern Detection

Advanced pattern detection algorithms identify profitable trading patterns and automatically adjust strategies based on historical performance.

### Memory Persistence

All memory keys, links, and performance data are persistently stored and can be retrieved across system restarts.

### Real-time Adaptation

The system continuously adapts to market conditions and agent performance, automatically adjusting trust scores and execution parameters.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the integration test output for debugging information
- Review the performance metrics for system health

---

**Schwabot UROS v1.0** - The future of recursive AI trading systems is here. 🚀 