# 🤖 LLM Code Debate System

**Multi-Agent LLM Debate System for Code Generation**

Система мультиагентных дебатов между языковыми моделями для генерации качественного кода. Несколько LLM обсуждают, критикуют и улучшают решения друг друга до достижения консенсуса.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Ключевые особенности

- **Мульти-агентный дебат**: 2-7 LLM агентов обсуждают решения
- **Локальные модели**: Поддержка Ollama (Qwen, DeepSeek, CodeLlama, Mistral)
- **Автоматическое тестирование**: Валидация кода через pytest
- **Детальная аналитика**: Метрики поведения агентов
- **Веб-интерфейс**: Визуализация дебатов в реальном времени
- **SQLite хранилище**: История всех экспериментов

## 📋 Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                         TASK INPUT                              │
│              "Implement a LRU Cache with O(1) operations"       │
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
│                    ROUND 2-N: DEBATE PHASE                      │
│                                                                 │
│  Each agent:                                                    │
│  1. Critiques other solutions (finds bugs, inefficiencies)      │
│  2. Defends or abandons their solution                          │
│  3. Proposes improvements                                       │
│  4. Votes for best solution                                     │
│                                                                 │
│  Loop until: CONSENSUS or MAX_ROUNDS reached                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CONSENSUS DETECTOR                           │
│  • Majority vote (>50% agree)                                   │
│  • All tests pass                                               │
│  • No new critiques                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL SOLUTION + METRICS                     │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Клонирование и установка
cd llm-code-debate
pip install -r requirements.txt

# Установка Ollama (если не установлен)
curl -fsSL https://ollama.com/install.sh | sh

# Загрузка моделей
ollama pull qwen2.5-coder:7b
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
```

### 2. Запуск дебата

```bash
# Простой запуск
python -m src.main --task tasks/medium/lru_cache.json

# С указанием агентов
python -m src.main --task tasks/hard/graph_traversal.json \
    --agents qwen2.5-coder:7b deepseek-coder:6.7b codellama:7b

# С веб-интерфейсом
python -m src.main --task tasks/medium/lru_cache.json --web
```

### 3. Веб-интерфейс

```bash
python -m src.web.app
# Открыть http://localhost:5000
```

## 📊 Исследовательские вопросы

### RQ1: Улучшает ли мульти-агентный дебат качество кода?
- Сравнение Pass@1 между single-agent и multi-agent
- Анализ типов найденных и исправленных ошибок

### RQ2: Как количество агентов влияет на результат?
- Эксперименты с 2, 3, 5, 7 агентами
- Поиск оптимального соотношения качество/время

### RQ3: Какие паттерны поведения проявляют разные LLM?
- Кто чаще "сдаётся" vs "защищает" своё решение
- Кто находит больше багов
- Кто предлагает оптимизации

### RQ4: Heterogeneous vs Homogeneous агенты
- Разные модели vs копии одной модели с разной temperature

## 📁 Структура проекта

```
llm-code-debate/
├── README.md
├── requirements.txt
├── config.yaml                 # Конфигурация
├── src/
│   ├── main.py                 # CLI точка входа
│   ├── models/                 # Датаклассы
│   │   ├── agent.py            # Agent, AgentRole
│   │   ├── solution.py         # Solution, CodeBlock
│   │   ├── critique.py         # Critique, Bug, Vote
│   │   ├── debate.py           # Debate, Round
│   │   └── metrics.py          # DebateMetrics, AgentStats
│   ├── core/                   # Бизнес-логика
│   │   ├── orchestrator.py     # DebateOrchestrator
│   │   ├── consensus.py        # ConsensusDetector
│   │   ├── executor.py         # CodeExecutor (pytest)
│   │   └── prompts.py          # Промпты для агентов
│   ├── llm/                    # LLM клиенты
│   │   ├── base.py             # BaseLLMClient
│   │   └── ollama_client.py    # OllamaClient
│   ├── analysis/               # Аналитика
│   │   ├── metrics_collector.py
│   │   └── visualizer.py
│   ├── database/               # Хранение
│   │   ├── models.py           # SQLAlchemy модели
│   │   └── repository.py       # CRUD операции
│   └── web/                    # Веб-интерфейс
│       ├── app.py              # Flask приложение
│       ├── templates/
│       └── static/
├── tasks/                      # Тестовые задания
│   ├── easy/
│   ├── medium/
│   └── hard/
└── tests/                      # Тесты
```

## 📈 Метрики

### Качество результата
- **Pass@1**: Доля успешных решений с первой попытки
- **Improvement over best initial**: Улучшение относительно лучшего начального решения

### Динамика дебата
- **Rounds to consensus**: Количество раундов до консенсуса
- **Total critiques**: Общее количество критических замечаний
- **Bugs found/fixed**: Найденные и исправленные баги

### Поведение агентов
- **Solutions proposed**: Количество предложенных решений
- **Critiques given/received**: Критика дана/получена
- **Times changed mind**: Сколько раз агент менял своё решение
- **Times won debate**: Сколько раз решение агента победило

## 🔧 Конфигурация

```yaml
# config.yaml
ollama:
  base_url: "http://localhost:11434"
  timeout: 120

debate:
  max_rounds: 5
  consensus_threshold: 0.6  # 60% агентов должны согласиться
  
agents:
  default:
    - qwen2.5-coder:7b
    - deepseek-coder:6.7b
    - codellama:7b

execution:
  timeout: 30
  max_memory_mb: 512

database:
  path: "debate_results.db"
```

## 📝 Формат задания

```json
{
  "id": "lru_cache",
  "name": "LRU Cache Implementation",
  "difficulty": "medium",
  "description": "Implement an LRU (Least Recently Used) cache...",
  "signature": "class LRUCache:\n    def __init__(self, capacity: int):\n    def get(self, key: int) -> int:\n    def put(self, key: int, value: int) -> None:",
  "tests": [
    "def test_basic():\n    cache = LRUCache(2)\n    cache.put(1, 1)\n    assert cache.get(1) == 1"
  ],
  "constraints": [
    "O(1) time complexity for both get and put",
    "capacity >= 1"
  ]
}
```

## 🧪 Запуск экспериментов

```bash
# Один эксперимент
python -m src.main --task tasks/medium/lru_cache.json --output results/

# Batch эксперименты
python scripts/run_experiments.py --tasks-dir tasks/ --output results/

# Анализ результатов
python scripts/analyze_results.py --input results/ --output analysis/
```

## 📜 Лицензия

MIT License - см. [LICENSE](LICENSE)

## 👤 Автор

Lukaš Patrik Magalinski  
Vilnius University, Faculty of Mathematics and Informatics

Научный руководитель: Prof. Dr. Aistis Raudys
