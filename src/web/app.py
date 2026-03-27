"""
Web interface for the LLM Code Debate System.

Provides:
- Real-time debate visualization
- Task management
- Results dashboard
- Experiment configuration
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from threading import Thread, Event
from typing import Any

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit

from ..core import DebateOrchestrator
from ..database import DebateRepository
from ..llm import MultiModelClient
from ..models import AgentConfig, AgentMessage, DebateConfig, RoundSummary, Task
from ..analysis import MetricsCollector, DebateVisualizer

# Configure logging to show in console
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create Flask app
app = Flask(__name__, 
    template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), 'static')
)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'debate-secret-key-change-in-production')

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global instances
repository: DebateRepository | None = None
llm_client: MultiModelClient | None = None
current_debate_thread: Thread | None = None
debate_stop_event: Event = Event()


def init_app(db_path: str = "debate_results.db", ollama_url: str = "http://localhost:11434"):
    """Initialize the application with dependencies."""
    global repository, llm_client
    repository = DebateRepository(db_path)
    llm_client = MultiModelClient(base_url=ollama_url)

    # Migrate existing database: add is_solo column if missing
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA table_info(debates)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'is_solo' not in columns:
            conn.execute("ALTER TABLE debates ADD COLUMN is_solo BOOLEAN DEFAULT 0")
            conn.commit()
            logger.info("Database migrated: added is_solo column")
        conn.close()
    except Exception as e:
        logger.debug(f"Database migration check: {e}")


# =============================================================================
# Routes
# =============================================================================

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route('/api/models')
def list_models():
    """List available Ollama models."""
    if not llm_client:
        return jsonify({"error": "LLM client not initialized"}), 500

    try:
        import asyncio
        loop = asyncio.new_event_loop()
        client = llm_client.get_client("dummy")
        models = loop.run_until_complete(client.list_models())
        loop.close()
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/tasks', methods=['GET'])
def list_tasks():
    """List available tasks."""
    tasks_dir = Path(__file__).parent.parent.parent / 'tasks'
    tasks = []
    
    for difficulty in ['easy', 'medium', 'hard', 'extreme']:
        diff_dir = tasks_dir / difficulty
        if diff_dir.exists():
            for task_file in diff_dir.glob('*.json'):
                try:
                    with open(task_file) as f:
                        task_data = json.load(f)
                        tasks.append({
                            "id": task_data.get("id"),
                            "name": task_data.get("name"),
                            "difficulty": difficulty,
                            "path": str(task_file),
                        })
                except Exception:
                    pass

    return jsonify({"tasks": tasks})


@app.route('/api/tasks/<task_id>')
def get_task(task_id: str):
    """Get a specific task by ID."""
    tasks_dir = Path(__file__).parent.parent.parent / 'tasks'
    
    for difficulty in ['easy', 'medium', 'hard', 'extreme']:
        diff_dir = tasks_dir / difficulty
        if diff_dir.exists():
            for task_file in diff_dir.glob('*.json'):
                try:
                    with open(task_file) as f:
                        task_data = json.load(f)
                        if task_data.get("id") == task_id:
                            return jsonify(task_data)
                except Exception:
                    pass

    return jsonify({"error": "Task not found"}), 404


@app.route('/api/debates', methods=['GET'])
def list_debates():
    """List all debates."""
    if not repository:
        return jsonify({"error": "Repository not initialized"}), 500
    
    debates = repository.get_all_debates(limit=50)
    return jsonify({
        "debates": [d.to_dict() for d in debates]
    })


@app.route('/api/debates/<debate_id>')
def get_debate(debate_id: str):
    """Get a specific debate."""
    if not repository:
        return jsonify({"error": "Repository not initialized"}), 500
    
    debate = repository.get_debate(debate_id)
    if not debate:
        return jsonify({"error": "Debate not found"}), 404
    
    return jsonify(debate.full_debate_data)


@app.route('/api/stats')
def get_stats():
    """Get summary statistics."""
    if not repository:
        return jsonify({"error": "Repository not initialized"}), 500
    
    stats = repository.get_summary_stats()
    return jsonify(stats)


# =============================================================================
# WebSocket Events
# =============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print("[DEBUG] Client connected via WebSocket")
    logger.info("Client connected")
    emit('connected', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info("Client disconnected")


@socketio.on('start_debate')
def handle_start_debate(data: dict):
    """Start a new debate."""
    global current_debate_thread

    print(f"[DEBUG] start_debate received: {data}")
    logger.info(f"start_debate event received: {data}")

    if not llm_client:
        print("[DEBUG] LLM client not initialized!")
        emit('error', {'message': 'LLM client not initialized'})
        return

    # Parse request
    task_path = data.get('task_path')
    agent_models = data.get('agents', ['qwen2.5-coder:7b', 'deepseek-coder:6.7b', 'codellama:7b-instruct'])
    max_rounds = data.get('max_rounds', 5)

    print(f"[DEBUG] task_path={task_path}, agents={agent_models}, max_rounds={max_rounds}")

    # Load task
    try:
        with open(task_path) as f:
            task_data = json.load(f)
        task = Task.from_dict(task_data)
        print(f"[DEBUG] Task loaded: {task.name}")
    except Exception as e:
        print(f"[DEBUG] Failed to load task: {e}")
        emit('error', {'message': f'Failed to load task: {e}'})
        return

    # Create agent configs
    agent_configs = [
        AgentConfig(name=f"agent_{i+1}", model=model)
        for i, model in enumerate(agent_models)
    ]
    
    # Create config
    config = DebateConfig(max_rounds=max_rounds)
    
    # Callbacks for real-time updates
    def on_round_complete(round_summary: RoundSummary):
        socketio.emit('round_complete', {
            'round_num': round_summary.round_num,
            'best_pass_rate': round_summary.best_pass_rate,
            'avg_pass_rate': round_summary.avg_pass_rate,
            'bugs_found': round_summary.bugs_found,
            'solutions': [s.to_dict() for s in round_summary.solutions],
            'votes': [v.to_dict() for v in round_summary.votes],
        })
    
    def on_message(message: AgentMessage):
        socketio.emit('agent_message', {
            'agent_id': message.agent_id,
            'round_num': message.round_num,
            'message_type': message.message_type,
            'content': message.content[:500],  # Truncate for real-time
        })

    def on_phase(phase: str, round_num: int):
        socketio.emit('phase_start', {
            'phase': phase,
            'round_num': round_num,
        })

    # Create orchestrator
    orchestrator = DebateOrchestrator(
        llm_client=llm_client,
        config=config,
        on_round_complete=on_round_complete,
        on_message=on_message,
        on_phase=on_phase,
    )
    
    # Run debate in background
    debate_stop_event.clear()

    def run_debate():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            socketio.emit('debate_started', {'task': task.name, 'agents': agent_models})

            debate = loop.run_until_complete(
                orchestrator.run_debate(task, agent_configs, stop_event=debate_stop_event)
            )

            # Save to database
            if repository:
                repository.save_debate(debate)

            # Collect metrics
            collector = MetricsCollector()
            metrics = collector.collect_debate_metrics(debate)

            # Build per-round evolution data for charts
            round_history = []
            for r in debate.rounds:
                round_agents = {}
                for sol in r.solutions:
                    pr = sol.pass_rate if sol.execution_result else 0.0
                    round_agents[sol.agent_id] = {
                        'pass_rate': pr,
                        'tests_passed': sol.execution_result.tests_passed if sol.execution_result else 0,
                        'tests_total': sol.execution_result.tests_total if sol.execution_result else 0,
                    }
                round_history.append({
                    'round_num': r.round_num,
                    'agents': round_agents,
                    'best_pass_rate': r.best_pass_rate,
                    'avg_pass_rate': r.avg_pass_rate,
                    'bugs_found': r.bugs_found,
                })

            # Collect critique summaries
            critique_summary = []
            for r in debate.rounds:
                for c in r.critiques:
                    critique_summary.append({
                        'agent_id': c.agent_id,
                        'target_agent_id': c.target_agent_id,
                        'round_num': r.round_num,
                        'correctness_rating': c.correctness_rating,
                        'efficiency_rating': c.efficiency_rating,
                        'readability_rating': c.readability_rating,
                        'bugs_count': len(c.bugs),
                        'improvements_count': len(c.improvements),
                        'would_adopt': c.would_adopt,
                        'ratings_parsed': getattr(c, 'ratings_parsed', True),
                    })

            # Send final result with full metrics
            socketio.emit('debate_complete', {
                'debate_id': debate.id,
                'status': debate.status.value,
                'final_pass_rate': metrics.final_pass_rate,
                'total_rounds': metrics.total_rounds,
                'winning_agent': debate.winning_agent_id,
                'final_solution': debate.final_solution.extract_code_block() if debate.final_solution else None,
                'metrics': metrics.to_dict(),
                'round_history': round_history,
                'critiques': critique_summary,
            })

        except Exception as e:
            logger.exception(f"Debate failed: {e}")
            socketio.emit('debate_error', {'message': str(e)})

        finally:
            loop.close()
            global current_debate_thread
            current_debate_thread = None

    # Start in background thread
    thread = Thread(target=run_debate)
    current_debate_thread = thread
    thread.start()

    emit('debate_starting', {'message': 'Debate is starting...'})


@socketio.on('stop_debate')
def handle_stop_debate():
    """Stop the current debate."""
    if current_debate_thread and current_debate_thread.is_alive():
        debate_stop_event.set()
        emit('debate_stopped', {'message': 'Stopping debate...'})
    else:
        emit('error', {'message': 'No debate running'})


@socketio.on('start_solo')
def handle_start_solo(data: dict):
    """Start a solo run (single agent, no debate)."""
    global current_debate_thread

    logger.info(f"start_solo event received: {data}")

    if not llm_client:
        emit('error', {'message': 'LLM client not initialized'})
        return

    task_path = data.get('task_path')
    agent_model = data.get('agent')

    if not task_path or not agent_model:
        emit('error', {'message': 'Task and agent are required'})
        return

    try:
        with open(task_path) as f:
            task_data = json.load(f)
        task = Task.from_dict(task_data)
    except Exception as e:
        emit('error', {'message': f'Failed to load task: {e}'})
        return

    agent_config = AgentConfig(name="solo_agent", model=agent_model)
    config = DebateConfig(max_rounds=1)

    def on_round_complete(round_summary: RoundSummary):
        socketio.emit('round_complete', {
            'round_num': round_summary.round_num,
            'best_pass_rate': round_summary.best_pass_rate,
            'avg_pass_rate': round_summary.avg_pass_rate,
            'bugs_found': 0,
            'solutions': [s.to_dict() for s in round_summary.solutions],
            'votes': [],
        })

    def on_message(message: AgentMessage):
        socketio.emit('agent_message', {
            'agent_id': message.agent_id,
            'round_num': message.round_num,
            'message_type': message.message_type,
            'content': message.content[:500],
        })

    def on_phase(phase: str, round_num: int):
        socketio.emit('phase_start', {'phase': phase, 'round_num': round_num})

    orchestrator = DebateOrchestrator(
        llm_client=llm_client,
        config=config,
        on_round_complete=on_round_complete,
        on_message=on_message,
        on_phase=on_phase,
    )

    def run_solo():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            socketio.emit('debate_started', {'task': task.name, 'agents': [agent_model], 'is_solo': True})

            debate = loop.run_until_complete(
                orchestrator.run_solo(task, agent_config)
            )

            if repository:
                repository.save_debate(debate)

            collector = MetricsCollector()
            metrics = collector.collect_debate_metrics(debate)

            socketio.emit('debate_complete', {
                'debate_id': debate.id,
                'status': debate.status.value,
                'final_pass_rate': metrics.final_pass_rate,
                'total_rounds': 1,
                'winning_agent': debate.winning_agent_id,
                'final_solution': debate.final_solution.extract_code_block() if debate.final_solution else None,
                'metrics': metrics.to_dict(),
                'round_history': [{
                    'round_num': 1,
                    'agents': {
                        s.agent_id: {
                            'pass_rate': s.pass_rate if s.execution_result else 0.0,
                            'tests_passed': s.execution_result.tests_passed if s.execution_result else 0,
                            'tests_total': s.execution_result.tests_total if s.execution_result else 0,
                        } for s in debate.rounds[0].solutions
                    } if debate.rounds else {},
                    'best_pass_rate': debate.rounds[0].best_pass_rate if debate.rounds else 0,
                    'avg_pass_rate': debate.rounds[0].avg_pass_rate if debate.rounds else 0,
                    'bugs_found': 0,
                }],
                'critiques': [],
                'is_solo': True,
            })

        except Exception as e:
            logger.exception(f"Solo run failed: {e}")
            socketio.emit('debate_error', {'message': str(e)})

        finally:
            loop.close()
            global current_debate_thread
            current_debate_thread = None

    thread = Thread(target=run_solo)
    current_debate_thread = thread
    thread.start()

    emit('debate_starting', {'message': 'Solo run starting...'})


@socketio.on('start_batch')
def handle_start_batch(data: dict):
    """Start a batch experiment: multiple tasks with given agents."""
    global current_debate_thread

    logger.info(f"start_batch event received: {data}")

    if not llm_client:
        emit('error', {'message': 'LLM client not initialized'})
        return

    task_paths = data.get('task_paths', [])
    agent_models = data.get('agents', [])
    max_rounds = data.get('max_rounds', 5)
    mode = data.get('mode', 'debate')  # 'debate' or 'solo'

    if not task_paths:
        emit('error', {'message': 'Select at least one task'})
        return
    if not agent_models:
        emit('error', {'message': 'Select at least one agent'})
        return

    config = DebateConfig(max_rounds=max_rounds)

    def run_batch():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        results = []
        total = len(task_paths)

        try:
            for idx, task_path in enumerate(task_paths):
                if debate_stop_event.is_set():
                    socketio.emit('batch_progress', {
                        'current': idx, 'total': total,
                        'status': 'stopped', 'task_name': 'Batch stopped by user',
                    })
                    break

                try:
                    with open(task_path) as f:
                        task_data = json.load(f)
                    task = Task.from_dict(task_data)
                except Exception as e:
                    logger.warning(f"Batch: failed to load {task_path}: {e}")
                    socketio.emit('batch_progress', {
                        'current': idx + 1, 'total': total,
                        'task_name': task_path, 'status': 'error', 'error': str(e),
                    })
                    continue

                orchestrator = DebateOrchestrator(llm_client=llm_client, config=config)

                try:
                    if mode == 'solo':
                        agent_config = AgentConfig(name="solo_agent", model=agent_models[0])
                        debate = loop.run_until_complete(orchestrator.run_solo(task, agent_config))
                    else:
                        agent_configs = [
                            AgentConfig(name=f"agent_{i+1}", model=m)
                            for i, m in enumerate(agent_models)
                        ]
                        debate = loop.run_until_complete(
                            orchestrator.run_debate(task, agent_configs)
                        )

                    if repository:
                        repository.save_debate(debate)

                    collector = MetricsCollector()
                    metrics = collector.collect_debate_metrics(debate)

                    results.append({
                        'task_id': task.id,
                        'task_name': task.name,
                        'pass_rate': metrics.final_pass_rate,
                        'status': debate.status.value,
                        'rounds': metrics.total_rounds,
                    })

                    socketio.emit('batch_progress', {
                        'current': idx + 1, 'total': total,
                        'task_name': task.name,
                        'pass_rate': metrics.final_pass_rate,
                        'status': 'done',
                    })

                except Exception as e:
                    logger.exception(f"Batch task {task.name} failed: {e}")
                    socketio.emit('batch_progress', {
                        'current': idx + 1, 'total': total,
                        'task_name': task.name, 'status': 'error', 'error': str(e),
                    })

            socketio.emit('batch_complete', {
                'total': total,
                'results': results,
                'mode': mode,
            })

        except Exception as e:
            logger.exception(f"Batch failed: {e}")
            socketio.emit('debate_error', {'message': str(e)})

        finally:
            loop.close()
            global current_debate_thread
            current_debate_thread = None

    thread = Thread(target=run_batch)
    current_debate_thread = thread
    thread.start()

    emit('debate_starting', {'message': f'Batch starting: {len(task_paths)} tasks...'})


@app.route('/api/clear-data', methods=['POST'])
def api_clear_data():
    """Clear all debate data from the database."""
    if not repository:
        return jsonify({"error": "No database"}), 404

    session = repository._get_session()
    try:
        from ..database.models import AgentStatRecord, DebateRecord, RoundRecord
        session.query(AgentStatRecord).delete()
        session.query(RoundRecord).delete()
        session.query(DebateRecord).delete()
        session.commit()
        logger.info("All debate data cleared by user")
        return jsonify({"status": "ok", "message": "All data cleared"})
    except Exception as e:
        session.rollback()
        logger.exception(f"Failed to clear data: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()


@app.route('/api/comparison')
def api_comparison():
    """Get comparison data between solo and debate runs."""
    if not repository:
        return jsonify({"tasks": [], "aggregate": {}})
    return jsonify(repository.get_comparison_data())


def _extract_metrics_from_record(record) -> dict:
    """Extract extended metrics from full_debate_data JSON."""
    data = record.full_debate_data or {}
    rounds = data.get("rounds", [])
    agents = data.get("agents", [])

    # Collect all solutions across rounds
    all_solutions = []
    for rd in rounds:
        for sol in rd.get("solutions", []):
            all_solutions.append(sol)

    n = len(all_solutions)
    c = sum(1 for s in all_solutions
            if s.get("execution_result", {}).get("pass_rate", 0) >= 100.0)

    # Pass@k
    from ..analysis.metrics_collector import compute_pass_at_k
    pass_at_1 = compute_pass_at_k(n, c, 1) if n >= 1 else 0.0
    pass_at_3 = compute_pass_at_k(n, c, 3) if n >= 3 else 0.0

    # Improvement over initial (round 1)
    initial_rates = []
    if rounds:
        for sol in rounds[0].get("solutions", []):
            er = sol.get("execution_result", {})
            initial_rates.append(er.get("pass_rate", 0.0))

    final_rate = record.final_pass_rate or 0.0
    best_initial = max(initial_rates) if initial_rates else 0.0
    avg_initial = (sum(initial_rates) / len(initial_rates)) if initial_rates else 0.0
    imp_best = (final_rate - best_initial) if initial_rates else 0.0
    imp_avg = (final_rate - avg_initial) if initial_rates else 0.0

    # Critique stats
    all_critiques = []
    for rd in rounds:
        all_critiques.extend(rd.get("critiques", []))

    total_critiques = len(all_critiques)
    total_bugs = sum(len(cr.get("bugs", [])) for cr in all_critiques)

    # Bug fix rate: compare round 1 best vs final
    bugs_fixed = max(0, int(final_rate - best_initial)) if initial_rates else 0
    bug_fix_rate = (bugs_fixed / total_bugs) if total_bugs > 0 else 0.0

    # Ratings (only from parsed critiques)
    parsed_critiques = [cr for cr in all_critiques if cr.get("ratings_parsed", False)]
    avg_corr = (sum(cr.get("correctness_rating", 0) for cr in parsed_critiques) / len(parsed_critiques)) if parsed_critiques else 0.0
    avg_eff = (sum(cr.get("efficiency_rating", 0) for cr in parsed_critiques) / len(parsed_critiques)) if parsed_critiques else 0.0
    avg_read = (sum(cr.get("readability_rating", 0) for cr in parsed_critiques) / len(parsed_critiques)) if parsed_critiques else 0.0

    # Quality metrics from final solution
    final_sol = data.get("final_solution", {})
    final_qm = final_sol.get("quality_metrics", {}) if final_sol else {}
    final_pylint = final_qm.get("pylint_score", 0.0)
    final_complexity = final_qm.get("cyclomatic_complexity", 0.0)

    # Initial quality (average from round 1)
    initial_pylints = []
    initial_complexities = []
    if rounds:
        for sol in rounds[0].get("solutions", []):
            qm = sol.get("quality_metrics", {})
            if qm:
                initial_pylints.append(qm.get("pylint_score", 0.0))
                initial_complexities.append(qm.get("cyclomatic_complexity", 0.0))
    init_avg_pylint = (sum(initial_pylints) / len(initial_pylints)) if initial_pylints else 0.0
    init_avg_complexity = (sum(initial_complexities) / len(initial_complexities)) if initial_complexities else 0.0

    # LLM time from agent stats
    total_llm_time = sum(a.get("stats", a).get("total_generation_time", 0.0) for a in agents)

    # Consensus
    final_consensus = data.get("final_consensus", {})
    consensus_ratio = final_consensus.get("consensus_ratio", record.consensus_ratio or 0.0) if final_consensus else (record.consensus_ratio or 0.0)

    # Rounds to consensus
    rounds_to_consensus = 0
    for rd in rounds:
        cr = rd.get("consensus_result", {})
        if cr and cr.get("reached"):
            rounds_to_consensus = rd.get("round_num", 0)
            break

    total_rounds = record.total_rounds or len(rounds)
    duration = record.duration_seconds or 0.0
    avg_round_dur = (duration / total_rounds) if total_rounds > 0 else 0.0

    return {
        "pass_at_1": round(pass_at_1, 4),
        "pass_at_3": round(pass_at_3, 4),
        "improvement_over_best_initial": round(imp_best, 2),
        "improvement_over_avg_initial": round(imp_avg, 2),
        "rounds_to_consensus": rounds_to_consensus,
        "total_critiques": total_critiques,
        "total_bugs_found": total_bugs,
        "total_bugs_fixed": bugs_fixed,
        "bug_fix_rate": round(bug_fix_rate, 4),
        "avg_correctness_rating": round(avg_corr, 2),
        "avg_efficiency_rating": round(avg_eff, 2),
        "avg_readability_rating": round(avg_read, 2),
        "initial_avg_pylint": round(init_avg_pylint, 2),
        "final_pylint": round(final_pylint, 2),
        "initial_avg_complexity": round(init_avg_complexity, 2),
        "final_complexity": round(final_complexity, 2),
        "avg_round_duration": round(avg_round_dur, 2),
        "total_llm_time": round(total_llm_time, 2),
        "all_solutions_count": n,
        "passing_solutions_count": c,
    }


@app.route('/api/export/csv')
def api_export_csv():
    """Export all results as CSV with 35+ metrics columns."""
    import csv
    import io

    if not repository:
        return "No data", 404

    session = repository._get_session()
    try:
        from ..database.models import DebateRecord
        records = session.query(DebateRecord).order_by(DebateRecord.start_time.desc()).all()

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            'debate_id', 'task_id', 'task_name', 'difficulty', 'mode',
            'agent_models', 'num_agents',
            'final_pass_rate', 'tests_passed', 'tests_total',
            'pass_at_1', 'pass_at_3',
            'improvement_over_best_initial', 'improvement_over_avg_initial',
            'total_rounds', 'consensus_reached', 'consensus_ratio', 'rounds_to_consensus',
            'total_critiques', 'total_bugs_found', 'total_bugs_fixed', 'bug_fix_rate',
            'avg_correctness_rating', 'avg_efficiency_rating', 'avg_readability_rating',
            'initial_avg_pylint', 'final_pylint',
            'initial_avg_complexity', 'final_complexity',
            'duration_seconds', 'avg_round_duration', 'total_llm_time',
            'all_solutions_count', 'passing_solutions_count',
            'status', 'winning_agent',
        ])

        for r in records:
            mode = 'solo' if r.is_solo else 'debate'
            models = ','.join(r.agent_models) if r.agent_models else ''
            num_agents = len(r.agent_models) if r.agent_models else 0
            ext = _extract_metrics_from_record(r)
            writer.writerow([
                r.id, r.task_id, r.task_name, r.task_difficulty, mode,
                models, num_agents,
                r.final_pass_rate, r.tests_passed, r.tests_total,
                ext['pass_at_1'], ext['pass_at_3'],
                ext['improvement_over_best_initial'], ext['improvement_over_avg_initial'],
                r.total_rounds, r.consensus_reached, r.consensus_ratio, ext['rounds_to_consensus'],
                ext['total_critiques'], ext['total_bugs_found'], ext['total_bugs_fixed'], ext['bug_fix_rate'],
                ext['avg_correctness_rating'], ext['avg_efficiency_rating'], ext['avg_readability_rating'],
                ext['initial_avg_pylint'], ext['final_pylint'],
                ext['initial_avg_complexity'], ext['final_complexity'],
                r.duration_seconds, ext['avg_round_duration'], ext['total_llm_time'],
                ext['all_solutions_count'], ext['passing_solutions_count'],
                r.status, r.winning_agent_id,
            ])

        from flask import Response
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=debate_results.csv'},
        )
    finally:
        session.close()


# =============================================================================
# Main
# =============================================================================

def run_server(host: str = "0.0.0.0", port: int = 5050, debug: bool = False):
    """Run the web server."""
    init_app()
    socketio.run(app, host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server(debug=True)
