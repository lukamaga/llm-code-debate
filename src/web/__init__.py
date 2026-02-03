"""
Web interface for the LLM Code Debate System.
"""
from .app import app, init_app, run_server, socketio

__all__ = ["app", "init_app", "run_server", "socketio"]
