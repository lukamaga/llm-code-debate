from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from ..models import AgentStats, Debate, DebateMetrics, Task
from .models import (
    AgentStatRecord,
    DebateRecord,
    ExperimentRecord,
    RoundRecord,
    TaskRecord,
    create_database,
)


class DebateRepository:
    
    def __init__(self, db_path: str = "debate_results.db"):
        self.SessionMaker = create_database(db_path)
    
    def _get_session(self) -> Session:
        return self.SessionMaker()
    
    def save_debate(self, debate: Debate) -> DebateRecord:
        session = self._get_session()
        
        try:
            record = DebateRecord(
                id=debate.id,
                task_id=debate.task.id,
                task_name=debate.task.name,
                task_difficulty=debate.task.difficulty,
                status=debate.status.value,
                error_message=debate.error_message,
                num_agents=len(debate.agents),
                max_rounds=debate.max_rounds,
                consensus_threshold=debate.consensus_threshold,
                agent_models=[a.model for a in debate.agents],
                final_pass_rate=debate.final_solution.pass_rate if debate.final_solution else 0.0,
                tests_passed=debate.final_solution.execution_result.tests_passed if debate.final_solution and debate.final_solution.execution_result else 0,
                tests_total=debate.final_solution.execution_result.tests_total if debate.final_solution and debate.final_solution.execution_result else 0,
                winning_agent_id=debate.winning_agent_id,
                consensus_reached=debate.final_consensus.reached if debate.final_consensus else False,
                consensus_ratio=debate.final_consensus.consensus_ratio if debate.final_consensus else 0.0,
                total_rounds=debate.total_rounds,
                duration_seconds=debate.duration_seconds,
                start_time=debate.start_time,
                end_time=debate.end_time,
                is_solo=debate.id.startswith("solo_"),
                full_debate_data=debate.to_dict(),
            )
            
            for round_summary in debate.rounds:
                round_record = RoundRecord(
                    debate_id=debate.id,
                    round_num=round_summary.round_num,
                    best_pass_rate=round_summary.best_pass_rate,
                    avg_pass_rate=round_summary.avg_pass_rate,
                    bugs_found=round_summary.bugs_found,
                    improvements_suggested=round_summary.improvements_suggested,
                    duration_seconds=round_summary.duration_seconds,
                    solutions_data=[s.to_dict() for s in round_summary.solutions],
                    critiques_data=[c.to_dict() for c in round_summary.critiques],
                    votes_data=[v.to_dict() for v in round_summary.votes],
                    consensus_data=round_summary.consensus_result.to_dict() if round_summary.consensus_result else None,
                )
                record.rounds.append(round_record)
            
            for agent in debate.agents:
                stat_record = AgentStatRecord(
                    debate_id=debate.id,
                    agent_id=agent.id,
                    model=agent.model,
                    role=agent.role.value,
                    solutions_proposed=agent.stats.solutions_proposed,
                    solutions_revised=agent.stats.solutions_revised,
                    critiques_given=agent.stats.critiques_given,
                    bugs_found=agent.stats.bugs_found,
                    times_changed_mind=agent.stats.times_changed_mind,
                    times_defended=agent.stats.times_defended,
                    times_won_debate=agent.stats.times_won_debate,
                    total_generation_time=agent.stats.total_generation_time,
                )
                record.agent_stats.append(stat_record)
            
            session.add(record)
            session.commit()
            
            return record
            
        finally:
            session.close()
    
    def get_debate(self, debate_id: str) -> DebateRecord | None:
        session = self._get_session()
        try:
            return session.query(DebateRecord).filter_by(id=debate_id).first()
        finally:
            session.close()
    
    def get_debates_by_task(self, task_id: str) -> list[DebateRecord]:
        session = self._get_session()
        try:
            return session.query(DebateRecord).filter_by(task_id=task_id).all()
        finally:
            session.close()
    
    def get_debates_by_difficulty(self, difficulty: str) -> list[DebateRecord]:
        session = self._get_session()
        try:
            return session.query(DebateRecord).filter_by(task_difficulty=difficulty).all()
        finally:
            session.close()
    
    def get_all_debates(self, limit: int = 100) -> list[DebateRecord]:
        session = self._get_session()
        try:
            return session.query(DebateRecord).order_by(DebateRecord.start_time.desc()).limit(limit).all()
        finally:
            session.close()
    
    def get_agent_stats_by_model(self, model: str) -> list[AgentStatRecord]:
        session = self._get_session()
        try:
            return session.query(AgentStatRecord).filter_by(model=model).all()
        finally:
            session.close()
    
    def save_task(self, task: Task) -> TaskRecord:
        session = self._get_session()
        try:
            existing = session.query(TaskRecord).filter_by(id=task.id).first()
            if existing:
                return existing
            
            record = TaskRecord(
                id=task.id,
                name=task.name,
                difficulty=task.difficulty,
                description=task.description,
                signature=task.signature,
                tests=task.tests,
                constraints=task.constraints,
                tags=task.tags,
            )
            session.add(record)
            session.commit()
            return record
        finally:
            session.close()
    
    def get_summary_stats(self) -> dict[str, Any]:
        session = self._get_session()
        try:
            debates = session.query(DebateRecord).all()
            
            if not debates:
                return {
                    "total_debates": 0,
                    "overall_pass_rate": 0.0,
                    "consensus_rate": 0.0,
                    "avg_rounds": 0.0,
                }
            
            total = len(debates)
            pass_rates = [d.final_pass_rate or 0.0 for d in debates]
            consensus_count = sum(1 for d in debates if d.consensus_reached)
            rounds = [d.total_rounds or 0 for d in debates]
            
            return {
                "total_debates": total,
                "overall_pass_rate": sum(pass_rates) / total,
                "consensus_rate": consensus_count / total,
                "avg_rounds": sum(rounds) / total,
                "by_difficulty": self._get_stats_by_difficulty(debates),
            }
        finally:
            session.close()
    
    def _get_stats_by_difficulty(self, debates: list[DebateRecord]) -> dict[str, dict]:
        by_diff: dict[str, list[DebateRecord]] = {}
        for d in debates:
            diff = d.task_difficulty or "unknown"
            if diff not in by_diff:
                by_diff[diff] = []
            by_diff[diff].append(d)
        
        result = {}
        for diff, dlist in by_diff.items():
            pass_rates = [d.final_pass_rate or 0.0 for d in dlist]
            result[diff] = {
                "count": len(dlist),
                "avg_pass_rate": sum(pass_rates) / len(dlist),
                "consensus_rate": sum(1 for d in dlist if d.consensus_reached) / len(dlist),
            }
        
        return result
    
    def create_experiment(
        self,
        experiment_id: str,
        name: str,
        description: str = "",
        config: dict | None = None,
    ) -> ExperimentRecord:
        session = self._get_session()
        try:
            record = ExperimentRecord(
                id=experiment_id,
                name=name,
                description=description,
                config=config or {},
                debate_ids=[],
            )
            session.add(record)
            session.commit()
            return record
        finally:
            session.close()
    
    def add_debate_to_experiment(
        self,
        experiment_id: str,
        debate_id: str,
    ) -> None:
        session = self._get_session()
        try:
            experiment = session.query(ExperimentRecord).filter_by(id=experiment_id).first()
            if experiment:
                debate_ids = experiment.debate_ids or []
                debate_ids.append(debate_id)
                experiment.debate_ids = debate_ids
                experiment.total_debates = len(debate_ids)
                session.commit()
        finally:
            session.close()

    def get_comparison_data(self) -> dict[str, Any]:
        session = self._get_session()
        try:
            all_records = session.query(DebateRecord).all()

            solo_by_task: dict[str, list[float]] = {}
            debate_by_task: dict[str, list[float]] = {}
            task_names: dict[str, str] = {}
            task_diffs: dict[str, str] = {}

            for d in all_records:
                tid = d.task_id
                pr = d.final_pass_rate or 0.0
                task_names[tid] = d.task_name or tid
                task_diffs[tid] = d.task_difficulty or "unknown"

                if d.is_solo:
                    solo_by_task.setdefault(tid, []).append(pr)
                else:
                    debate_by_task.setdefault(tid, []).append(pr)

            tasks = []
            all_task_ids = set(solo_by_task) | set(debate_by_task)
            for tid in sorted(all_task_ids):
                solo_rates = solo_by_task.get(tid, [])
                debate_rates = debate_by_task.get(tid, [])
                solo_avg = sum(solo_rates) / len(solo_rates) if solo_rates else None
                debate_avg = sum(debate_rates) / len(debate_rates) if debate_rates else None

                improvement = None
                if solo_avg is not None and debate_avg is not None:
                    improvement = debate_avg - solo_avg

                tasks.append({
                    "task_id": tid,
                    "task_name": task_names.get(tid, tid),
                    "difficulty": task_diffs.get(tid, "unknown"),
                    "solo_pass_rate": solo_avg,
                    "debate_pass_rate": debate_avg,
                    "solo_count": len(solo_rates),
                    "debate_count": len(debate_rates),
                    "improvement": improvement,
                })

            improvements = [t["improvement"] for t in tasks if t["improvement"] is not None]
            return {
                "tasks": tasks,
                "aggregate": {
                    "avg_improvement": sum(improvements) / len(improvements) if improvements else 0,
                    "improved_count": sum(1 for i in improvements if i > 0),
                    "same_count": sum(1 for i in improvements if i == 0),
                    "worse_count": sum(1 for i in improvements if i < 0),
                    "total_compared": len(improvements),
                },
            }
        finally:
            session.close()
