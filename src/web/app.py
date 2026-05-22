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

from flask import Flask, jsonify, render_template, request, send_from_directory, abort
from flask_socketio import SocketIO, emit

from ..core import DebateOrchestrator
from ..database import DebateRepository
from ..llm import MultiModelClient
from ..models import AgentConfig, AgentMessage, AgentRole, DebateConfig, RoundSummary, Task
from ..analysis import MetricsCollector, DebateVisualizer

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

app = Flask(__name__, 
    template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), 'static')
)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'debate-secret-key-change-in-production')

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

repository: DebateRepository | None = None
llm_client: MultiModelClient | None = None
current_debate_thread: Thread | None = None
_debate_lock = __import__('threading').Lock()
debate_stop_event: Event = Event()

_last_finished_state: dict | None = None
_LAST_STATE_TTL_SEC = 60 * 60

TRANSCRIPTS_DIR = Path("transcripts")


def _save_transcript(debate) -> None:
    try:
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        visualizer = DebateVisualizer(output_dir=TRANSCRIPTS_DIR)
        path = visualizer.generate_full_transcript(debate)
        logger.info(f"Transcript saved: {path}")
    except Exception as e:
        logger.warning(f"Failed to save transcript for {getattr(debate, 'id', '?')}: {e}")


def _remember_finished(event_name: str, payload: dict) -> None:
    global _last_finished_state
    import time as _time
    _last_finished_state = {
        "event": event_name,
        "payload": payload,
        "timestamp": _time.time(),
        "debate_id": payload.get("debate_id") or "",
    }


def _try_save_debate(debate) -> bool:
    if not repository:
        return False
    try:
        repository.save_debate(debate)
        return True
    except Exception as e:
        logger.error(
            "Failed to save debate %s to DB — results will still be emitted to UI: %s",
            getattr(debate, "id", "?"), e,
        )
        return False


def init_app(db_path: str = "debate_results.db", ollama_url: str = "http://localhost:11434"):
    global repository, llm_client
    repository = DebateRepository(db_path)
    llm_client = MultiModelClient(base_url=ollama_url)

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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/health')
def health_check():
    return jsonify({"status": "ok"})


@app.route('/api/models')
def list_models():
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
    if not repository:
        return jsonify({"error": "Repository not initialized"}), 500
    
    debates = repository.get_all_debates(limit=50)
    return jsonify({
        "debates": [d.to_dict() for d in debates]
    })


@app.route('/api/debates/<debate_id>')
def get_debate(debate_id: str):
    if not repository:
        return jsonify({"error": "Repository not initialized"}), 500
    
    debate = repository.get_debate(debate_id)
    if not debate:
        return jsonify({"error": "Debate not found"}), 404
    
    return jsonify(debate.full_debate_data)


@app.route('/api/stats')
def get_stats():
    if not repository:
        return jsonify({"error": "Repository not initialized"}), 500

    stats = repository.get_summary_stats()
    return jsonify(stats)


def _parse_transcript_filename(name: str) -> dict:
    stem = name[:-4] if name.endswith('.txt') else name
    parts = stem.split('_')
    meta = {
        'difficulty': '-',
        'task_id': '-',
        'mode': '-',
        'debate_id': '-',
        'timestamp': '-',
    }
    if len(parts) >= 5:
        meta['difficulty'] = parts[0]
        meta['timestamp'] = parts[-2] + '_' + parts[-1]
        if parts[-3] in ('solo', 'debate'):
            meta['mode'] = parts[-3]
            meta['debate_id'] = parts[-3] + '_' + parts[-4] if parts[-3] == 'solo' else parts[-4]
        for i, p in enumerate(parts):
            if p in ('solo', 'debate'):
                meta['mode'] = p
                meta['task_id'] = '_'.join(parts[1:i])
                meta['debate_id'] = '_'.join(parts[i+1:-2]) if len(parts) > i + 3 else parts[i+1] if len(parts) > i + 1 else '-'
                break
    return meta


def _scan_transcript_metadata(path: Path) -> dict:
    info = {'task_name': '-', 'status': '-', 'pass_rate': None, 'rounds': None}
    try:
        with path.open('r', encoding='utf-8') as f:
            head = [next(f, '') for _ in range(40)]
        for line in head:
            line = line.strip()
            if line.startswith('DEBATE TRANSCRIPT'):
                after = line.split('|', 1)[-1].strip() if '|' in line else ''
                if '(' in after:
                    info['task_name'] = after.rsplit('(', 1)[0].strip()
                else:
                    info['task_name'] = after
            elif line.startswith('Status:'):
                info['status'] = line.split(':', 1)[1].strip()
            elif line.startswith('Rounds run:'):
                try:
                    info['rounds'] = int(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
        if info['task_name'] != '-':
            with path.open('r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('Winner:'):
                        if '%' in line:
                            pct = line.rsplit(',', 1)[-1].strip().rstrip(')')
                            pct = pct.replace('%', '')
                            try:
                                info['pass_rate'] = float(pct) / 100.0
                            except ValueError:
                                pass
                        break
    except Exception:
        pass
    return info


@app.route('/api/transcripts', methods=['GET'])
def list_transcripts():
    if not TRANSCRIPTS_DIR.exists():
        return jsonify({'transcripts': []})

    items = []
    for p in sorted(TRANSCRIPTS_DIR.glob('*.txt'), key=lambda x: x.stat().st_mtime, reverse=True):
        meta = _parse_transcript_filename(p.name)
        scanned = _scan_transcript_metadata(p)
        stat = p.stat()
        items.append({
            'filename': p.name,
            'size': stat.st_size,
            'mtime': stat.st_mtime,
            'difficulty': meta['difficulty'],
            'task_id': meta['task_id'],
            'task_name': scanned['task_name'],
            'mode': meta['mode'],
            'debate_id': meta['debate_id'],
            'timestamp': meta['timestamp'],
            'status': scanned['status'],
            'pass_rate': scanned['pass_rate'],
            'rounds': scanned['rounds'],
        })
    return jsonify({'transcripts': items})


def _safe_transcript_path(filename: str) -> Path | None:
    if '/' in filename or '\\' in filename or filename.startswith('.'):
        return None
    if not filename.endswith('.txt'):
        return None
    path = (TRANSCRIPTS_DIR / filename).resolve()
    try:
        path.relative_to(TRANSCRIPTS_DIR.resolve())
    except ValueError:
        return None
    if not path.exists() or not path.is_file():
        return None
    return path


@app.route('/api/transcripts/<path:filename>', methods=['GET'])
def get_transcript(filename: str):
    path = _safe_transcript_path(filename)
    if path is None:
        abort(404)
    try:
        content = path.read_text(encoding='utf-8')
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    return content, 200, {'Content-Type': 'text/plain; charset=utf-8'}


@app.route('/api/transcripts/<path:filename>/download', methods=['GET'])
def download_transcript(filename: str):
    path = _safe_transcript_path(filename)
    if path is None:
        abort(404)
    return send_from_directory(
        TRANSCRIPTS_DIR.resolve(),
        path.name,
        as_attachment=True,
        mimetype='text/plain',
    )


@socketio.on('connect')
def handle_connect():
    print("[DEBUG] Client connected via WebSocket")
    logger.info("Client connected")
    emit('connected', {'status': 'connected'})

    if _last_finished_state is not None:
        import time as _time
        age = _time.time() - _last_finished_state["timestamp"]
        if age <= _LAST_STATE_TTL_SEC:
            logger.info(
                "Replaying %s for reconnected client (debate_id=%s, age=%.0fs)",
                _last_finished_state["event"],
                _last_finished_state["debate_id"],
                age,
            )
            emit(_last_finished_state["event"], _last_finished_state["payload"])


@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")


@socketio.on('start_debate')
def handle_start_debate(data: dict):
    global current_debate_thread

    print(f"[DEBUG] start_debate received: {data}")
    logger.info(f"start_debate event received: {data}")

    if not llm_client:
        print("[DEBUG] LLM client not initialized!")
        emit('error', {'message': 'LLM client not initialized'})
        return

    task_path = data.get('task_path')
    agent_models = data.get('agents', ['qwen2.5-coder:7b', 'deepseek-coder:6.7b', 'codellama:7b-instruct'])
    max_rounds = data.get('max_rounds', 5)
    show_all_solutions = data.get('show_all_solutions', False)
    revision_strategy = data.get('revision_strategy', 'uniform')
    agent_strategies = data.get('agent_strategies', {})
    adaptive_temperature = data.get('adaptive_temperature', False)
    critique_history = data.get('critique_history', False)
    judge_model = (data.get('judge') or '').strip() or None
    if judge_model and judge_model.lower() == 'none':
        judge_model = None

    print(f"[DEBUG] task_path={task_path}, agents={agent_models}, judge={judge_model}, max_rounds={max_rounds}, show_all={show_all_solutions}, strategy={revision_strategy}, adaptive_temp={adaptive_temperature}, critique_hist={critique_history}")

    try:
        with open(task_path) as f:
            task_data = json.load(f)
        task = Task.from_dict(task_data)
        print(f"[DEBUG] Task loaded: {task.name}")
    except Exception as e:
        print(f"[DEBUG] Failed to load task: {e}")
        emit('error', {'message': f'Failed to load task: {e}'})
        return

    agent_configs = [
        AgentConfig(name=f"agent_{i+1}", model=model)
        for i, model in enumerate(agent_models)
    ]
    if judge_model:
        agent_configs.append(AgentConfig(
            name="judge",
            model=judge_model,
            role=AgentRole.JUDGE,
        ))
    
    config = DebateConfig(
        max_rounds=max_rounds, min_rounds=max_rounds,
        revision_show_all_solutions=show_all_solutions,
        revision_strategy=revision_strategy,
        agent_strategies=agent_strategies,
        adaptive_temperature=adaptive_temperature,
        critique_history=critique_history,
    )
    
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
            'content': message.content[:2000],
        })

    def on_phase(phase: str, round_num: int):
        socketio.emit('phase_start', {
            'phase': phase,
            'round_num': round_num,
        })

    orchestrator = DebateOrchestrator(
        llm_client=llm_client,
        config=config,
        on_round_complete=on_round_complete,
        on_message=on_message,
        on_phase=on_phase,
    )
    
    debate_stop_event.clear()

    def run_debate():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            global _last_finished_state
            _last_finished_state = None

            socketio.emit('debate_started', {
                'task': task.name,
                'agents': agent_models,
                'judge': judge_model,
            })

            debate = loop.run_until_complete(
                orchestrator.run_debate(task, agent_configs, stop_event=debate_stop_event)
            )

            _try_save_debate(debate)

            _save_transcript(debate)

            collector = MetricsCollector()
            metrics = collector.collect_debate_metrics(debate)

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

            complete_payload = {
                'debate_id': debate.id,
                'status': debate.status.value,
                'final_pass_rate': metrics.final_pass_rate,
                'total_rounds': metrics.total_rounds,
                'winning_agent': debate.winning_agent_id,
                'final_solution': debate.final_solution.extract_code_block() if debate.final_solution else None,
                'metrics': metrics.to_dict(),
                'round_history': round_history,
                'critiques': critique_summary,
            }
            _remember_finished('debate_complete', complete_payload)
            socketio.emit('debate_complete', complete_payload)

        except Exception as e:
            logger.exception(f"Debate failed: {e}")
            err_payload = {'message': str(e)}
            _remember_finished('debate_error', err_payload)
            socketio.emit('debate_error', err_payload)

        finally:
            loop.close()
            global current_debate_thread
            with _debate_lock:
                current_debate_thread = None

    with _debate_lock:
        if current_debate_thread and current_debate_thread.is_alive():
            emit('error', {'message': 'A debate is already running'})
            return
        thread = Thread(target=run_debate)
        current_debate_thread = thread
        thread.start()

    emit('debate_starting', {'message': 'Debate is starting...'})


@socketio.on('stop_debate')
def handle_stop_debate():
    if current_debate_thread and current_debate_thread.is_alive():
        debate_stop_event.set()
        emit('debate_stopped', {'message': 'Stopping debate...'})
    else:
        emit('error', {'message': 'No debate running'})


@socketio.on('start_solo')
def handle_start_solo(data: dict):
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
            'content': message.content[:2000],
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
            global _last_finished_state
            _last_finished_state = None

            socketio.emit('debate_started', {'task': task.name, 'agents': [agent_model], 'is_solo': True})

            debate = loop.run_until_complete(
                orchestrator.run_solo(task, agent_config)
            )

            _try_save_debate(debate)

            _save_transcript(debate)

            collector = MetricsCollector()
            metrics = collector.collect_debate_metrics(debate)

            solo_payload = {
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
            }
            _remember_finished('debate_complete', solo_payload)
            socketio.emit('debate_complete', solo_payload)

        except Exception as e:
            logger.exception(f"Solo run failed: {e}")
            err_payload = {'message': str(e)}
            _remember_finished('debate_error', err_payload)
            socketio.emit('debate_error', err_payload)

        finally:
            loop.close()
            global current_debate_thread
            with _debate_lock:
                current_debate_thread = None

    with _debate_lock:
        if current_debate_thread and current_debate_thread.is_alive():
            emit('error', {'message': 'A debate is already running'})
            return
        thread = Thread(target=run_solo)
        current_debate_thread = thread
        thread.start()

    emit('debate_starting', {'message': 'Solo run starting...'})


@socketio.on('start_batch')
def handle_start_batch(data: dict):
    global current_debate_thread

    logger.info(f"start_batch event received: {data}")

    if not llm_client:
        emit('error', {'message': 'LLM client not initialized'})
        return

    task_paths = data.get('task_paths', [])
    agent_models = data.get('agents', [])
    max_rounds = data.get('max_rounds', 5)
    mode = data.get('mode', 'debate')
    show_all_solutions = data.get('show_all_solutions', False)
    revision_strategy = data.get('revision_strategy', 'uniform')
    agent_strategies = data.get('agent_strategies', {})
    adaptive_temperature = data.get('adaptive_temperature', False)
    critique_history = data.get('critique_history', False)
    judge_model = (data.get('judge') or '').strip() or None
    if judge_model and judge_model.lower() == 'none':
        judge_model = None

    if not task_paths:
        emit('error', {'message': 'Select at least one task'})
        return
    if not agent_models:
        emit('error', {'message': 'Select at least one agent'})
        return

    config = DebateConfig(
        max_rounds=max_rounds,
        revision_show_all_solutions=show_all_solutions,
        revision_strategy=revision_strategy,
        agent_strategies=agent_strategies,
        adaptive_temperature=adaptive_temperature,
        critique_history=critique_history,
    )

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
                        if judge_model:
                            agent_configs.append(AgentConfig(
                                name="judge",
                                model=judge_model,
                                role=AgentRole.JUDGE,
                            ))
                        debate = loop.run_until_complete(
                            orchestrator.run_debate(task, agent_configs)
                        )

                    _try_save_debate(debate)

                    _save_transcript(debate)

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
            with _debate_lock:
                current_debate_thread = None

    with _debate_lock:
        if current_debate_thread and current_debate_thread.is_alive():
            emit('error', {'message': 'A debate is already running'})
            return
        thread = Thread(target=run_batch)
        current_debate_thread = thread
        thread.start()

    emit('debate_starting', {'message': f'Batch starting: {len(task_paths)} tasks...'})


@app.route('/api/clear-data', methods=['POST'])
def api_clear_data():
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
    if not repository:
        return jsonify({"tasks": [], "aggregate": {}})
    return jsonify(repository.get_comparison_data())


def _extract_metrics_from_record(record) -> dict:
    data = record.full_debate_data or {}
    rounds = data.get("rounds", [])
    agents = data.get("agents", [])

    all_solutions = []
    for rd in rounds:
        for sol in rd.get("solutions", []):
            all_solutions.append(sol)

    n = len(all_solutions)
    c = sum(1 for s in all_solutions
            if s.get("execution_result", {}).get("status") == "passed")

    from ..analysis.metrics_collector import compute_pass_at_k
    pass_at_1 = compute_pass_at_k(n, c, 1) if n >= 1 else 0.0
    pass_at_3 = compute_pass_at_k(n, c, 3) if n >= 3 else 0.0

    initial_rates = []
    if rounds:
        for sol in rounds[0].get("solutions", []):
            er = sol.get("execution_result", {})
            initial_rates.append(er.get("pass_rate", 0.0))

    final_rate = record.final_pass_rate or 0.0
    best_initial = max(initial_rates) if initial_rates else 0.0
    avg_initial = (sum(initial_rates) / len(initial_rates)) if initial_rates else 0.0
    if best_initial > 0:
        imp_best = (final_rate - best_initial) / best_initial
    elif final_rate > 0:
        imp_best = final_rate
    else:
        imp_best = 0.0
    if avg_initial > 0:
        imp_avg = (final_rate - avg_initial) / avg_initial
    elif final_rate > 0:
        imp_avg = final_rate
    else:
        imp_avg = 0.0

    all_critiques = []
    for rd in rounds:
        all_critiques.extend(rd.get("critiques", []))

    total_critiques = len(all_critiques)
    total_bugs = sum(len(cr.get("bugs", [])) for cr in all_critiques)
    total_improvements = sum(len(cr.get("improvements", [])) for cr in all_critiques)

    bugs_fixed = 0
    if len(rounds) >= 2:
        first_sols = {s.get("agent_id"): s for s in rounds[0].get("solutions", [])}
        last_sols = {s.get("agent_id"): s for s in rounds[-1].get("solutions", [])}
        for agent_id, sol1 in first_sols.items():
            sol2 = last_sols.get(agent_id)
            if sol1 and sol2:
                t1 = sol1.get("execution_result", {}).get("tests_passed", 0)
                t2 = sol2.get("execution_result", {}).get("tests_passed", 0)
                if t2 > t1:
                    bugs_fixed += (t2 - t1)
    bug_fix_rate = (bugs_fixed / total_bugs) if total_bugs > 0 else 0.0

    parsed_critiques = [cr for cr in all_critiques if cr.get("ratings_parsed", False)]
    avg_corr = (sum(cr.get("correctness_rating", 0) for cr in parsed_critiques) / len(parsed_critiques)) if parsed_critiques else 0.0
    avg_eff = (sum(cr.get("efficiency_rating", 0) for cr in parsed_critiques) / len(parsed_critiques)) if parsed_critiques else 0.0
    avg_read = (sum(cr.get("readability_rating", 0) for cr in parsed_critiques) / len(parsed_critiques)) if parsed_critiques else 0.0

    final_sol = data.get("final_solution", {})
    final_qm = final_sol.get("quality_metrics", {}) if final_sol else {}
    final_pylint = final_qm.get("pylint_score", 0.0)
    final_complexity = final_qm.get("cyclomatic_complexity", 0.0)

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

    total_llm_time = sum(a.get("stats", a).get("total_generation_time", 0.0) for a in agents)
    total_exec_time = sum(
        s.get("execution_result", {}).get("execution_time", 0.0)
        for s in all_solutions
    )

    final_consensus = data.get("final_consensus", {})
    consensus_ratio = final_consensus.get("consensus_ratio", record.consensus_ratio or 0.0) if final_consensus else (record.consensus_ratio or 0.0)

    rounds_to_consensus = 0
    for rd in rounds:
        cr = rd.get("consensus_result", {})
        if cr and cr.get("reached"):
            rounds_to_consensus = rd.get("round_num", 0)
            break

    best_round = 1
    best_round_pass_rate = 0.0
    for rd in rounds:
        rd_best = rd.get("best_pass_rate", 0.0) or 0.0
        if rd_best > best_round_pass_rate:
            best_round_pass_rate = rd_best
            best_round = rd.get("round_num", 0)
    peak_after_debate = best_round > 1

    total_rounds = record.total_rounds or len(rounds)
    duration = record.duration_seconds or 0.0
    avg_round_dur = (duration / total_rounds) if total_rounds > 0 else 0.0

    agent_behavior = []
    for a in agents:
        stats = a.get("stats", a)
        agent_behavior.append({
            "agent_id": a.get("id", stats.get("agent_id", "")),
            "model": a.get("model", stats.get("model", "")),
            "critiques_given": stats.get("critiques_given", 0),
            "bugs_found": stats.get("bugs_found", 0),
            "times_changed_mind": stats.get("times_changed_mind", 0),
            "times_defended": stats.get("times_defended", 0),
            "times_adopted_other": stats.get("times_adopted_other", 0),
            "times_won_debate": stats.get("times_won_debate", 0),
        })

    most_bugs_found_by = ""
    most_active_agent = ""
    most_successful_agent = ""
    if agent_behavior:
        by_bugs = max(agent_behavior, key=lambda a: a["bugs_found"])
        if by_bugs["bugs_found"] > 0:
            most_bugs_found_by = by_bugs["agent_id"]
        by_critiques = max(agent_behavior, key=lambda a: a["critiques_given"])
        most_active_agent = by_critiques["agent_id"]
        winners = [a for a in agent_behavior if a["times_won_debate"] > 0]
        if winners:
            most_successful_agent = winners[0]["agent_id"]

    return {
        "pass_at_1": round(pass_at_1, 4),
        "pass_at_3": round(pass_at_3, 4),
        "improvement_over_best_initial": round(imp_best, 4),
        "improvement_over_avg_initial": round(imp_avg, 4),
        "rounds_to_consensus": rounds_to_consensus,
        "total_critiques": total_critiques,
        "total_bugs_found": total_bugs,
        "total_bugs_fixed": bugs_fixed,
        "bug_fix_rate": round(bug_fix_rate, 4),
        "total_improvements_suggested": total_improvements,
        "avg_correctness_rating": round(avg_corr, 2),
        "avg_efficiency_rating": round(avg_eff, 2),
        "avg_readability_rating": round(avg_read, 2),
        "initial_avg_pylint": round(init_avg_pylint, 2),
        "final_pylint": round(final_pylint, 2),
        "initial_avg_complexity": round(init_avg_complexity, 2),
        "final_complexity": round(final_complexity, 2),
        "avg_round_duration": round(avg_round_dur, 2),
        "total_llm_time": round(total_llm_time, 2),
        "total_execution_time": round(total_exec_time, 2),
        "all_solutions_count": n,
        "passing_solutions_count": c,
        "most_active_agent": most_active_agent,
        "most_successful_agent": most_successful_agent,
        "most_bugs_found_by": most_bugs_found_by,
        "agent_behavior": agent_behavior,
        "best_round": best_round,
        "best_round_pass_rate": round(best_round_pass_rate, 4),
        "peak_after_debate": peak_after_debate,
    }


@app.route('/api/export/csv')
def api_export_csv():
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
            'total_improvements_suggested',
            'avg_correctness_rating', 'avg_efficiency_rating', 'avg_readability_rating',
            'initial_avg_pylint', 'final_pylint',
            'initial_avg_complexity', 'final_complexity',
            'duration_seconds', 'avg_round_duration', 'total_llm_time', 'total_execution_time',
            'all_solutions_count', 'passing_solutions_count',
            'most_active_agent', 'most_successful_agent', 'most_bugs_found_by',
            'status', 'winning_agent',
            'best_round', 'best_round_pass_rate', 'peak_after_debate',
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
                ext['total_improvements_suggested'],
                ext['avg_correctness_rating'], ext['avg_efficiency_rating'], ext['avg_readability_rating'],
                ext['initial_avg_pylint'], ext['final_pylint'],
                ext['initial_avg_complexity'], ext['final_complexity'],
                r.duration_seconds, ext['avg_round_duration'], ext['total_llm_time'], ext['total_execution_time'],
                ext['all_solutions_count'], ext['passing_solutions_count'],
                ext['most_active_agent'], ext['most_successful_agent'], ext['most_bugs_found_by'],
                r.status, r.winning_agent_id,
                ext['best_round'], ext['best_round_pass_rate'], ext['peak_after_debate'],
            ])

        from flask import Response
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=debate_results.csv'},
        )
    finally:
        session.close()


def run_server(host: str = "0.0.0.0", port: int = 5050, debug: bool = False):
    init_app()
    socketio.run(app, host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server(debug=True)
