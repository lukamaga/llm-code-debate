# LLM Code Debate

**Multi-Agent Debate System for Code Generation Using Local LLMs**

Multiple LLM agents collaborate to solve coding tasks by proposing solutions, critiquing each other's code, and iteratively improving until they reach consensus on the best solution.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLMs-orange.svg)

## Key Features

- **Multi-Agent Debate**: 2-7 LLM agents discuss and improve solutions collaboratively
- **Local Models**: Supports Ollama with Qwen, DeepSeek, CodeLlama, Mistral, and more
- **Automatic Testing**: Code validation via pytest with pass rate tracking
- **Real-time Web UI**: Visualize debates as they happen with phase indicators
- **Comprehensive Metrics**: Track agent behavior, code quality (Pylint, Radon), and debate dynamics
- **SQLite Storage**: Persist all experiments for later analysis
- **Consensus Detection**: Weighted voting with confidence scores and test-pass bonuses

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         TASK INPUT                              │
│              "Implement an LRU Cache with O(1) operations"      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ROUND 1: INITIAL PROPOSALS                   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Qwen    │ │DeepSeek │ │CodeLlama│ │ Mistral │ │ Llama3  │   │
│  │Solution1│ │Solution2│ │Solution3│ │Solution4│ │Solution5│   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ROUNDS 2-N: DEBATE PHASE                     │
│                                                                 │
│  Each agent:                                                    │
│  1. CRITIQUE - Reviews all other solutions, finds bugs          │
│  2. REVISE - Fixes own solution OR adopts a better one          │
│  3. VOTE - Selects the best solution with confidence score      │
│                                                                 │
│  Agents see ALL solutions and ALL critiques (shared discussion) │
│                                                                 │
│  Loop until: CONSENSUS reached or MAX_ROUNDS                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CONSENSUS DETECTOR                           │
│  • Weighted voting (by confidence)                              │
│  • 1.5x bonus for solutions passing all tests                   │
│  • Threshold: 60% agreement required                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL SOLUTION + METRICS                     │
│  • Pass rate, rounds to consensus, bugs found/fixed             │
│  • Agent behavior profiles, code quality scores                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-code-debate.git
cd llm-code-debate

# Install dependencies
pip install -r requirements.txt

# Pull LLM models (choose any combination)
ollama pull qwen2.5-coder:7b
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b-instruct
```

### Running a Debate

```bash
# Quick run with default agents
python scripts/quick_run.py --task tasks/medium/lru_cache.json

# Specify models and rounds
python scripts/quick_run.py \
    --task tasks/hard/word_ladder.json \
    --models qwen2.5-coder:7b deepseek-coder:6.7b codellama:7b-instruct \
    --rounds 5
```

### Web Interface

```bash
python -m src.web.app
# Open http://localhost:5050
```

The web UI provides:
- Task and agent selection
- Real-time debate visualization with phase indicators (Propose → Critique → Revise → Vote)
- Progress bars and test results per agent
- Activity feed with all agent messages
- Final solution display with pass rate

## Research Questions

This system is designed to investigate:

### RQ1: Does multi-agent debate improve code quality?
- Compare Pass@1 between single-agent and multi-agent approaches
- Analyze types of bugs found and fixed through debate

### RQ2: How does the number of agents affect results?
- Experiments with 2, 3, 5, 7 agents
- Find optimal quality/time tradeoff

### RQ3: What behavioral patterns do different LLMs exhibit?
- Who defends vs. abandons their solution
- Who finds more bugs
- Who proposes optimizations

### RQ4: Heterogeneous vs. Homogeneous agents
- Different models vs. copies of one model with varying temperature

## Metrics

### Result Quality
- **Pass Rate**: Percentage of tests passed
- **Improvement over Initial**: How much debate improved the best initial solution

### Debate Dynamics
- **Rounds to Consensus**: How many rounds until agreement
- **Total Critiques**: Number of critical reviews given
- **Bugs Found/Fixed**: Issues identified and resolved

### Code Quality (via Pylint & Radon)
- **Pylint Score**: 0-10 code quality rating
- **Cyclomatic Complexity**: Code complexity measure
- **Maintainability Index**: 0-100 maintainability score

### Agent Behavior
- **Solutions Proposed/Revised**: Code contributions
- **Critiques Given/Received**: Review activity
- **Times Changed Mind**: Flexibility in accepting better solutions
- **Times Adopted Other**: When agent takes another's solution
- **Personality Type**: Classified as aggressive_critic, passive_adopter, stubborn_defender, active_collaborator, or balanced

## Project Structure

```
llm-code-debate/
├── README.md
├── requirements.txt
├── config.yaml                 # Configuration
├── src/
│   ├── main.py                 # CLI entry point
│   ├── models/                 # Data classes
│   │   ├── agent.py            # Agent, AgentRole, AgentStats
│   │   ├── solution.py         # Solution, ExecutionResult, CodeQualityMetrics
│   │   ├── critique.py         # Critique, Bug, Vote, ConsensusResult
│   │   ├── debate.py           # Debate, RoundSummary, DebateConfig
│   │   └── metrics.py          # DebateMetrics, ExperimentSummary, AgentProfile
│   ├── core/                   # Business logic
│   │   ├── orchestrator.py     # DebateOrchestrator (main debate loop)
│   │   ├── consensus.py        # ConsensusDetector (voting logic)
│   │   ├── executor.py         # CodeExecutor (pytest), CodeQualityAnalyzer
│   │   └── prompts.py          # LLM prompts and response parsing
│   ├── llm/                    # LLM clients
│   │   ├── base.py             # BaseLLMClient interface
│   │   └── ollama_client.py    # OllamaClient implementation
│   ├── analysis/               # Analytics
│   │   ├── metrics_collector.py # Collect and aggregate metrics
│   │   └── visualizer.py       # Generate visualizations
│   ├── database/               # Persistence
│   │   ├── models.py           # SQLAlchemy models
│   │   └── repository.py       # CRUD operations
│   └── web/                    # Web interface
│       ├── app.py              # Flask + SocketIO application
│       └── templates/          # HTML templates
├── tasks/                      # Coding tasks
│   ├── easy/                   # Simple problems
│   ├── medium/                 # Intermediate problems
│   └── hard/                   # Complex problems
├── scripts/                    # Utility scripts
│   └── quick_run.py            # Quick experiment runner
└── tests/                      # Unit tests
```

## Configuration

```yaml
# config.yaml
ollama:
  base_url: "http://localhost:11434"
  timeout: 120

debate:
  max_rounds: 5
  min_rounds: 2
  consensus_threshold: 0.6    # 60% agreement required
  early_stop_on_perfect: true # Stop if 100% tests pass

agents:
  default:
    - qwen2.5-coder:7b
    - deepseek-coder:6.7b
    - codellama:7b-instruct

execution:
  timeout: 30
  max_memory_mb: 512

database:
  path: "debate_results.db"
```

## Task Format

Tasks are defined in JSON:

```json
{
  "id": "lru_cache",
  "name": "LRU Cache Implementation",
  "difficulty": "medium",
  "description": "Implement an LRU (Least Recently Used) cache with O(1) time complexity for both get and put operations.",
  "signature": "class LRUCache:\n    def __init__(self, capacity: int):\n    def get(self, key: int) -> int:\n    def put(self, key: int, value: int) -> None:",
  "tests": [
    "def test_basic():\n    cache = LRUCache(2)\n    cache.put(1, 1)\n    cache.put(2, 2)\n    assert cache.get(1) == 1\n    cache.put(3, 3)  # evicts key 2\n    assert cache.get(2) == -1"
  ],
  "constraints": [
    "1 <= capacity <= 3000",
    "O(1) time complexity for get and put"
  ]
}
```

## Running Experiments

```bash
# Single experiment with output
python scripts/quick_run.py --task tasks/medium/lru_cache.json

# Batch experiments (all tasks)
python scripts/run_experiments.py --tasks-dir tasks/ --output results/

# Analyze results
python scripts/analyze_results.py --input results/ --output analysis/
```

## API Endpoints

The web server exposes:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/models` | GET | List available Ollama models |
| `/api/tasks` | GET | List available tasks |
| `/api/debates` | GET | List past debates |
| `/api/debates/<id>` | GET | Get debate details |
| `/api/stats` | GET | Get summary statistics |

WebSocket events:
- `start_debate` - Start a new debate
- `stop_debate` - Stop current debate
- `phase_start` - Phase changed (propose/critique/revise/vote)
- `agent_message` - Agent sent a message
- `round_complete` - Round finished
- `debate_complete` - Debate finished

## License

MIT License - see [LICENSE](LICENSE)

## Author

Lukas Patrik Magalinski
Vilnius University, Faculty of Mathematics and Informatics

## Acknowledgments

- [Ollama](https://ollama.com/) for local LLM inference
- Inspired by research on multi-agent debate systems for AI alignment
