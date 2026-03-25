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
    agent_models = data.get('agents', ['qwen2.5-coder:7b', 'deepseek-coder:6.7b', 'codellama:7b'])
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

            # Send final result
            socketio.emit('debate_complete', {
                'debate_id': debate.id,
                'status': debate.status.value,
                'final_pass_rate': metrics.final_pass_rate,
                'total_rounds': metrics.total_rounds,
                'winning_agent': debate.winning_agent_id,
                'final_solution': debate.final_solution.extract_code_block() if debate.final_solution else None,
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


# =============================================================================
# Main
# =============================================================================

def run_server(host: str = "0.0.0.0", port: int = 5050, debug: bool = False):
    """Run the web server."""
    init_app()
    socketio.run(app, host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server(debug=True)
