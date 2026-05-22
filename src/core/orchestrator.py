from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from threading import Event
from typing import Any, Callable

from ..llm import LLMRequest, MultiModelClient
from ..models import (
    Agent,
    AgentConfig,
    AgentMessage,
    AgentRole,
    Bug,
    BugSeverity,
    ConsensusResult,
    Critique,
    Debate,
    DebateConfig,
    DebateStatus,
    Improvement,
    ImprovementType,
    RoundSummary,
    Solution,
    Task,
    Vote,
    VoteType,
)
from .consensus import ConsensusConfig, ConsensusDetector
from .executor import CodeExecutor, CodeQualityAnalyzer
from .prompts import (
    SYSTEM_PROMPT_CODER,
    SYSTEM_PROMPT_CRITIC,
    SYSTEM_PROMPT_JUDGE,
    STRATEGY_ORDER,
    build_critique_prompt,
    build_proposal_prompt,
    build_multi_file_proposal_prompt,
    build_chunked_file_proposal_prompt,
    build_chunked_file_revision_prompt,
    build_revision_prompt,
    build_voting_prompt,
    extract_code_from_response,
    extract_multi_file_code_from_response,
    parse_critique_response,
    parse_vote_response,
)

logger = logging.getLogger(__name__)


_SYNTAX_BREAKAGE_TOKENS = (
    "syntaxerror",
    "indentationerror",
    "importerror",
    "modulenotfounderror",
    "nameerror",
)

_BROKEN_STATUSES = frozenset({"syntax_error", "runtime_error", "timeout"})


def _status_value(result: "Any") -> str:
    status = getattr(result, "status", None)
    if status is None:
        return ""
    return getattr(status, "value", None) or str(status)


_CONTROL_CLAUSE_PREFIXES = (
    "if ", "elif ", "else:", "else ", "try:", "try ", "except",
    "finally:", "finally ", "for ", "while ", "with ",
)


def _looks_truncated(code: str) -> bool:
    if not code:
        return False
    stripped = code.rstrip()
    if not stripped:
        return False

    opens = stripped.count("{") + stripped.count("[") + stripped.count("(")
    closes = stripped.count("}") + stripped.count("]") + stripped.count(")")
    bracket_delta = opens - closes
    if bracket_delta >= 2:
        return True

    last = stripped[-1]

    if (last.isalnum() or last == "_") and bracket_delta >= 1:
        return True

    lines = stripped.splitlines()
    if len(lines) >= 2:
        last_line = lines[-1].strip()
        if last_line.endswith(":") and any(
            last_line.startswith(p) for p in _CONTROL_CLAUSE_PREFIXES
        ):
            return True

    return False


def _is_placeholder_body(body: "list") -> bool:
    import ast as _ast

    if not body:
        return False

    stmts = body
    if (
        isinstance(stmts[0], _ast.Expr)
        and isinstance(stmts[0].value, _ast.Constant)
        and isinstance(stmts[0].value.value, str)
    ):
        stmts = stmts[1:]

    if len(stmts) != 1:
        return False

    s = stmts[0]
    if isinstance(s, _ast.Pass):
        return True
    if (
        isinstance(s, _ast.Expr)
        and isinstance(s.value, _ast.Constant)
        and s.value.value is Ellipsis
    ):
        return True
    if isinstance(s, _ast.Raise) and s.exc is not None:
        exc = s.exc
        name_node = exc.func if isinstance(exc, _ast.Call) else exc
        if isinstance(name_node, _ast.Name) and name_node.id == "NotImplementedError":
            return True
    return False


def _looks_lazy(code: str) -> bool:
    if not code or not code.strip():
        return False

    import ast as _ast

    if "# FILE:" in code:
        parts = []
        current: list[str] = []
        for line in code.splitlines():
            if line.strip().startswith("# FILE:"):
                if current:
                    parts.append("\n".join(current))
                current = []
            else:
                current.append(line)
        if current:
            parts.append("\n".join(current))
        return any(_looks_lazy(part) for part in parts if part.strip())

    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return False

    total_defs = 0
    placeholder_defs = 0
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            total_defs += 1
            if _is_placeholder_body(node.body):
                placeholder_defs += 1

    if total_defs < 4:
        return False

    return (placeholder_defs / total_defs) >= 0.25


def should_revert_revision(
    prev_best: "Solution | None",
    revised_result: "Any",
    revised_code: str | None = None,
) -> tuple[bool, str]:
    if prev_best is None or prev_best.execution_result is None:
        return (False, "no_prev_best")

    prev_passed = prev_best.execution_result.tests_passed
    regression_size = prev_passed - revised_result.tests_passed

    status_val = _status_value(revised_result).lower()
    if status_val in _BROKEN_STATUSES and revised_result.tests_passed == 0:
        return (True, "syntax_broken")

    if revised_result.tests_total <= 1 and (revised_result.error_message or "").strip():
        err_lower = (revised_result.error_message or "").lower()
        if any(tok in err_lower for tok in _SYNTAX_BREAKAGE_TOKENS):
            return (True, "syntax_broken")

    if (
        revised_code is not None
        and _looks_truncated(revised_code)
        and (regression_size >= 1 or revised_result.tests_passed * 2 < prev_passed)
    ):
        return (True, "truncated_output")

    if (
        revised_code is not None
        and _looks_lazy(revised_code)
        and regression_size >= 0
    ):
        return (True, "lazy_skeleton")

    if regression_size >= 2:
        return (True, f"regressed_{regression_size}_tests")

    if regression_size == 1:
        return (False, "regressed_by_one_kept")

    if revised_code is not None and prev_best.code:
        prev_norm = (prev_best.code or "").strip()
        new_norm = (revised_code or "").strip()
        if prev_norm and prev_norm == new_norm:
            return (False, "no_change_revision")

    return (False, "no_regression")


class DebateOrchestrator:
    
    def __init__(
        self,
        llm_client: MultiModelClient,
        config: DebateConfig | None = None,
        on_round_complete: Callable[[RoundSummary], None] | None = None,
        on_message: Callable[[AgentMessage], None] | None = None,
        on_phase: Callable[[str, int], None] | None = None,
    ):
        self.llm_client = llm_client
        self.config = config or DebateConfig()
        self.executor = CodeExecutor(timeout=self.config.execution_timeout)
        self.quality_analyzer = CodeQualityAnalyzer()
        self.consensus_detector = ConsensusDetector(
            ConsensusConfig(threshold=self.config.consensus_threshold)
        )
        self.on_round_complete = on_round_complete
        self.on_message = on_message
        self.on_phase = on_phase
    
    async def run_debate(
        self,
        task: Task,
        agent_configs: list[AgentConfig],
        stop_event: Event | None = None,
    ) -> Debate:
        debate_id = str(uuid.uuid4())[:8]
        agents = self._create_agents(agent_configs)
        
        debate = Debate(
            id=debate_id,
            task=task,
            agents=agents,
            max_rounds=self.config.max_rounds,
            consensus_threshold=self.config.consensus_threshold,
        )
        debate.status = DebateStatus.RUNNING
        
        logger.info(f"Starting debate {debate_id} on task '{task.name}' with {len(agents)} agents")
        
        try:
            round1 = await self._run_proposal_round(debate)
            debate.add_round(round1)
            
            if self.on_round_complete:
                self.on_round_complete(round1)
            
            if self.config.early_stop_on_perfect:
                perfect = self.consensus_detector.check_perfect_solution(round1.solutions)
                if perfect:
                    debate.finalize(
                        status=DebateStatus.EARLY_STOP,
                        final_solution=perfect,
                        consensus=ConsensusResult(
                            reached=True,
                            winning_solution_id=perfect.id,
                            winning_agent_id=perfect.agent_id,
                            consensus_ratio=1.0,
                            round_num=1,
                        ),
                    )
                    logger.info(f"Early stop: Perfect solution found in round 1")
                    return debate
            
            for round_num in range(2, self.config.max_rounds + 1):
                if stop_event and stop_event.is_set():
                    best = self._find_best_solution(debate.get_latest_solutions())
                    debate.finalize(
                        status=DebateStatus.EARLY_STOP,
                        final_solution=best,
                    )
                    logger.info("Debate stopped by user")
                    return debate

                round_summary = await self._run_debate_round(debate, round_num)
                debate.add_round(round_summary)
                
                if self.on_round_complete:
                    self.on_round_complete(round_summary)
                
                if (round_summary.consensus_result and round_summary.consensus_result.reached
                        and round_num >= self.config.min_rounds):
                    winning_solution = self._get_solution_by_id(
                        round_summary.solutions,
                        round_summary.consensus_result.winning_solution_id
                    )
                    if not winning_solution:
                        winning_solution = self._find_best_solution(round_summary.solutions)
                    best_ever = self._find_best_solution_across_rounds(debate)
                    if best_ever and best_ever.pass_rate > winning_solution.pass_rate:
                        logger.warning(
                            "Consensus winner (%.0f%%) is worse than best historical (%.0f%%), using historical",
                            winning_solution.pass_rate * 100, best_ever.pass_rate * 100,
                        )
                        winning_solution = best_ever
                    debate.finalize(
                        status=DebateStatus.CONSENSUS_REACHED,
                        final_solution=winning_solution,
                        consensus=round_summary.consensus_result,
                    )
                    logger.info(f"Consensus reached in round {round_num}")
                    return debate
                elif round_summary.consensus_result and round_summary.consensus_result.reached:
                    logger.info(
                        f"Consensus reached in round {round_num} but min_rounds={self.config.min_rounds}, continuing"
                    )

                should_stop, reason = self.consensus_detector.should_early_stop(
                    round_summary.solutions,
                    round_summary.votes,
                    round_num,
                    self.config.min_rounds,
                )
                if should_stop:
                    best = self._find_best_solution_across_rounds(debate)
                    debate.finalize(
                        status=DebateStatus.EARLY_STOP,
                        final_solution=best,
                    )
                    logger.info(f"Early stop: {reason}")
                    return debate

            best_solution = self._find_best_solution_across_rounds(debate)
            debate.finalize(
                status=DebateStatus.MAX_ROUNDS_REACHED,
                final_solution=best_solution,
            )
            logger.info(f"Max rounds ({self.config.max_rounds}) reached")
            
        except Exception as e:
            logger.exception(f"Debate failed with error: {e}")
            debate.finalize(
                status=DebateStatus.ERROR,
                error_message=str(e),
            )
        
        return debate

    async def run_solo(
        self,
        task: Task,
        agent_config: AgentConfig,
    ) -> Debate:
        debate_id = f"solo_{str(uuid.uuid4())[:8]}"
        agents = self._create_agents([agent_config])
        agent = agents[0]

        debate = Debate(
            id=debate_id,
            task=task,
            agents=agents,
            max_rounds=1,
            consensus_threshold=1.0,
        )
        debate.status = DebateStatus.RUNNING

        logger.info(f"Starting solo run {debate_id} on task '{task.name}' with {agent_config.model}")

        if self.on_phase:
            self.on_phase("propose", 1)

        try:
            solution = await self._get_proposal(agent, task)

            round_summary = RoundSummary(round_num=1, solutions=[], critiques=[], votes=[])

            if solution:
                result = await self.executor.execute(solution, task)
                solution.execution_result = result

                try:
                    quality = await self.quality_analyzer.analyze(
                        "\n\n".join(solution.extract_code_files().values()) if solution.code_files else solution.extract_code_block()
                    )
                    solution.quality_metrics = quality
                except Exception as e:
                    logger.warning(f"Quality analysis failed: {e}")

                round_summary.solutions.append(solution)
                logger.info(
                    f"Solo agent {solution.agent_id}: "
                    f"{result.tests_passed}/{result.tests_total} tests passed"
                )

                if self.on_message:
                    self.on_message(AgentMessage(
                        agent_id=agent.id,
                        round_num=1,
                        message_type="proposal",
                        content=solution.code[:500],
                    ))

            round_summary.end_time = datetime.now()
            round_summary.compute_stats()
            debate.add_round(round_summary)

            if self.on_round_complete:
                self.on_round_complete(round_summary)

            debate.finalize(
                status=DebateStatus.EARLY_STOP,
                final_solution=solution,
                consensus=ConsensusResult(
                    reached=True,
                    winning_solution_id=solution.id if solution else "",
                    winning_agent_id=agent.id,
                    consensus_ratio=1.0,
                    round_num=1,
                ) if solution else None,
            )

        except Exception as e:
            logger.exception(f"Solo run failed: {e}")
            debate.finalize(status=DebateStatus.ERROR, error_message=str(e))

        return debate

    def _create_agents(self, configs: list[AgentConfig]) -> list[Agent]:
        agents = []
        for i, config in enumerate(configs):
            agent_id = f"agent_{i+1}_{config.model.split(':')[0]}"
            agents.append(Agent(id=agent_id, config=config))
        return agents
    
    async def _run_proposal_round(self, debate: Debate) -> RoundSummary:
        logger.info("Running proposal round (Round 1)")
        if self.on_phase:
            self.on_phase("propose", 1)
        round_summary = RoundSummary(round_num=1, solutions=[], critiques=[], votes=[])
        
        tasks = []
        for agent in debate.agents:
            if agent.role != AgentRole.JUDGE:
                tasks.append(self._get_proposal(agent, debate.task))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        solutions = [r for r in results if isinstance(r, Solution)]

        for solution in solutions:
            if solution:
                result = await self.executor.execute(solution, debate.task)
                solution.execution_result = result
                
                try:
                    quality = await self.quality_analyzer.analyze(
                        "\n\n".join(solution.extract_code_files().values()) if solution.code_files else solution.extract_code_block()
                    )
                    solution.quality_metrics = quality
                except Exception as e:
                    logger.warning(f"Quality analysis failed: {e}")
                
                round_summary.solutions.append(solution)
                logger.info(
                    f"Agent {solution.agent_id}: "
                    f"{result.tests_passed}/{result.tests_total} tests passed"
                )
        
        round_summary.end_time = datetime.now()
        return round_summary
    
    async def _run_debate_round(self, debate: Debate, round_num: int) -> RoundSummary:
        logger.info(f"Running debate round {round_num}")

        round_summary = RoundSummary(
            round_num=round_num,
            solutions=[],
            critiques=[],
            votes=[],
        )

        current_solutions = debate.get_latest_solutions()

        prev_critique_summary = ""
        if self.config.critique_history and len(debate.rounds) >= 1:
            prev_round = debate.rounds[-1]
            summary_parts = []
            for crit in prev_round.critiques:
                bugs_str = "; ".join(b.description[:60] for b in crit.bugs[:3])
                if bugs_str:
                    summary_parts.append(
                        f"- {crit.agent_id} → {crit.target_agent_id}: {bugs_str}"
                    )
            if summary_parts:
                prev_critique_summary = "\n".join(summary_parts)

        if self.on_phase:
            self.on_phase("critique", round_num)
        critique_tasks = []
        for agent in debate.agents:
            if agent.role != AgentRole.PROPOSER:
                critique_tasks.append(
                    self._get_critique(
                        agent, debate.task, current_solutions, round_num,
                        previous_critique_summary=prev_critique_summary,
                    )
                )

        critique_results = await asyncio.gather(*critique_tasks, return_exceptions=True)
        for result in critique_results:
            if isinstance(result, list):
                round_summary.critiques.extend(result)

        critiques_by_agent: dict[str, list[Critique]] = {}
        for critique in round_summary.critiques:
            target = critique.target_agent_id
            if target not in critiques_by_agent:
                critiques_by_agent[target] = []
            critiques_by_agent[target].append(critique)

        if self.on_phase:
            self.on_phase("revise", round_num)

        best_historical: dict[str, Solution] = {}
        for rd in debate.rounds:
            for sol in rd.solutions:
                prev = best_historical.get(sol.agent_id)
                if prev is None or sol.pass_rate > prev.pass_rate:
                    best_historical[sol.agent_id] = sol

        effective_temp = self.config.temperature_revision
        if self.config.adaptive_temperature and len(debate.rounds) >= 2:
            prev_best = debate.rounds[-1].best_pass_rate
            prev_prev_best = debate.rounds[-2].best_pass_rate
            if prev_best <= prev_prev_best:
                stagnant_rounds = 0
                for i in range(len(debate.rounds) - 1, 0, -1):
                    if debate.rounds[i].best_pass_rate <= debate.rounds[i - 1].best_pass_rate:
                        stagnant_rounds += 1
                    else:
                        break
                temp_boost = stagnant_rounds * 0.15
                effective_temp = min(0.9, self.config.temperature_revision + temp_boost)
                logger.info(
                    "Adaptive temperature: %d stagnant rounds, "
                    "revision temp %.2f → %.2f",
                    stagnant_rounds, self.config.temperature_revision, effective_temp,
                )

        agent_strategy_map: dict[str, str] = {}
        if self.config.revision_strategy == "diverse":
            non_judge_agents = [a for a in debate.agents if a.role != AgentRole.JUDGE]
            for i, agent in enumerate(non_judge_agents):
                if agent.id in self.config.agent_strategies:
                    agent_strategy_map[agent.id] = self.config.agent_strategies[agent.id]
                else:
                    agent_strategy_map[agent.id] = STRATEGY_ORDER[i % len(STRATEGY_ORDER)]
            logger.info("Diverse strategies assigned: %s", agent_strategy_map)

        revision_tasks = []
        for agent in debate.agents:
            if agent.role != AgentRole.JUDGE:
                own_solution = best_historical.get(agent.id)
                if not own_solution:
                    own_solution = next(
                        (s for s in current_solutions if s.agent_id == agent.id),
                        None
                    )
                if not own_solution and current_solutions:
                    own_solution = self._find_best_solution(current_solutions)
                    logger.info(
                        f"Agent {agent.id} has no solution, adopting from {own_solution.agent_id}"
                    )
                if own_solution and own_solution != next(
                    (s for s in current_solutions if s.agent_id == agent.id), None
                ):
                    logger.info(
                        f"Agent {agent.id}: using historical best (round {own_solution.round_num}, "
                        f"{own_solution.pass_rate:.0%}) instead of latest round"
                    )
                if own_solution:
                    agent_critiques = critiques_by_agent.get(agent.id, [])
                    revision_tasks.append(
                        self._get_revision(
                            agent, debate.task, own_solution, agent_critiques, round_num,
                            all_solutions=current_solutions,
                            all_critiques=round_summary.critiques,
                            strategy=agent_strategy_map.get(agent.id, ""),
                            temperature_override=effective_temp,
                            previous_critiques_summary=prev_critique_summary,
                        )
                    )
        
        revision_results = await asyncio.gather(*revision_tasks, return_exceptions=True)
        revised_solutions = [r for r in revision_results if isinstance(r, Solution)]

        for solution in revised_solutions:
            if solution:
                result = await self.executor.execute(solution, debate.task)
                solution.execution_result = result

                prev_best = best_historical.get(solution.agent_id)
                revert_flag, revert_reason = should_revert_revision(
                    prev_best, result, revised_code=solution.code,
                )
                prev_passed = (
                    prev_best.execution_result.tests_passed
                    if prev_best and prev_best.execution_result else 0
                )
                prev_total = (
                    prev_best.execution_result.tests_total
                    if prev_best and prev_best.execution_result else 0
                )
                if revert_flag:
                    logger.warning(
                        "Agent %s revision reverted (%s): %d/%d → %d/%d",
                        solution.agent_id, revert_reason,
                        prev_passed, prev_total,
                        result.tests_passed, result.tests_total,
                    )
                    solution = prev_best
                elif revert_reason == "regressed_by_one_kept":
                    logger.info(
                        "Agent %s revision regressed by 1 test (%d/%d → %d/%d), "
                        "keeping for exploration",
                        solution.agent_id,
                        prev_passed, prev_total,
                        result.tests_passed, result.tests_total,
                    )
                elif revert_reason == "no_change_revision":
                    logger.warning(
                        "Agent %s submitted IDENTICAL code as revision (no change "
                        "from prev best %d/%d) — model rejected critique or stuck",
                        solution.agent_id, prev_passed, prev_total,
                    )

                try:
                    quality = await self.quality_analyzer.analyze(
                        "\n\n".join(solution.extract_code_files().values()) if solution.code_files else solution.extract_code_block()
                    )
                    solution.quality_metrics = quality
                except Exception:
                    pass

                round_summary.solutions.append(solution)
                logger.info(
                    f"Agent {solution.agent_id} (revised): "
                    f"{solution.execution_result.tests_passed}/{solution.execution_result.tests_total} tests passed"
                )
        
        if self.on_phase:
            self.on_phase("vote", round_num)
        vote_tasks = []
        for agent in debate.agents:
            vote_tasks.append(
                self._get_vote(agent, debate.task, round_summary.solutions, round_num)
            )
        
        vote_results = await asyncio.gather(*vote_tasks, return_exceptions=True)
        round_summary.votes = [v for v in vote_results if isinstance(v, Vote)]

        sol_map = {s.id: s for s in round_summary.solutions}
        for vote in round_summary.votes:
            if vote.voted_solution_id and not vote.parse_failed:
                sol = sol_map.get(vote.voted_solution_id)
                if sol:
                    sol.votes_received += 1
                if vote.voted_agent_id:
                    for ag in debate.agents:
                        if ag.id == vote.voted_agent_id:
                            ag.stats.final_votes_received += 1
                            break

        round_summary.consensus_result = self.consensus_detector.detect(
            round_summary.votes,
            round_summary.solutions,
            debate.agents,
            round_num,
        )
        
        round_summary.end_time = datetime.now()
        round_summary.compute_stats()
        return round_summary

    @staticmethod
    def _is_truncated(response) -> bool:
        return response.finish_reason == "length"

    async def _get_proposal(self, agent: Agent, task: Task) -> Solution | None:
        try:
            import time

            if task.is_multi_file:
                code_files: dict[str, str] = {}
                total_gen_time = 0.0
                was_truncated = False

                for filename in task.required_files:
                    prompt = build_chunked_file_proposal_prompt(
                        task, filename, already_generated=code_files or None,
                    )
                    request = LLMRequest(
                        prompt=prompt,
                        system_prompt=SYSTEM_PROMPT_CODER,
                        temperature=self.config.temperature_initial,
                        max_tokens=agent.config.max_tokens,
                    )

                    start = time.time()
                    response = await self.llm_client.generate(agent.model, request)
                    gen_time = time.time() - start
                    total_gen_time += gen_time

                    if self._is_truncated(response):
                        was_truncated = True
                        logger.warning(
                            "Agent %s proposal for %s TRUNCATED "
                            "(finish_reason=length, %d tokens, %d chars)",
                            agent.id, filename,
                            response.tokens_used, len(response.content),
                        )

                    file_code = extract_code_from_response(response.content)
                    if not file_code:
                        logger.warning(
                            "Agent %s proposal: no code extracted for %s "
                            "(response: %d chars)",
                            agent.id, filename, len(response.content),
                        )
                    code_files[filename] = file_code

                agent.stats.total_generation_time += total_gen_time
                combined = "\n\n".join(
                    f"# FILE: {fn}\n{fc}" for fn, fc in code_files.items()
                )
                solution = Solution(
                    id=f"sol_{agent.id}_r1",
                    agent_id=agent.id,
                    round_num=1,
                    code=combined,
                    code_files=code_files,
                    generation_time=total_gen_time,
                    was_truncated=was_truncated,
                )
            else:
                prompt = build_proposal_prompt(task)
                request = LLMRequest(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT_CODER,
                    temperature=self.config.temperature_initial,
                    max_tokens=agent.config.max_tokens,
                )

                start = time.time()
                response = await self.llm_client.generate(agent.model, request)
                generation_time = time.time() - start
                agent.stats.total_generation_time += generation_time

                was_truncated = self._is_truncated(response)
                if was_truncated:
                    logger.warning(
                        "Agent %s proposal TRUNCATED "
                        "(finish_reason=length, %d tokens, %d chars)",
                        agent.id, response.tokens_used, len(response.content),
                    )

                code = extract_code_from_response(response.content)
                if not code:
                    logger.warning(
                        "Agent %s proposal: no code extracted "
                        "(response: %d chars)",
                        agent.id, len(response.content),
                    )
                solution = Solution(
                    id=f"sol_{agent.id}_r1",
                    agent_id=agent.id,
                    round_num=1,
                    code=code,
                    generation_time=generation_time,
                    was_truncated=was_truncated,
                )

            message = AgentMessage(
                agent_id=agent.id,
                round_num=1,
                message_type="proposal",
                content=solution.code,
            )
            agent.add_message(message)

            if self.on_message:
                self.on_message(message)

            return solution

        except Exception as e:
            logger.error(f"Agent {agent.id} failed to propose: {e}")
            return None
    
    async def _get_critique(
        self,
        agent: Agent,
        task: Task,
        solutions: list[Solution],
        round_num: int = 0,
        previous_critique_summary: str = "",
    ) -> list[Critique]:
        prompt = build_critique_prompt(
            task, solutions, agent.id,
            previous_critique_summary=previous_critique_summary,
        )

        system_prompt = SYSTEM_PROMPT_JUDGE if agent.role == AgentRole.JUDGE else SYSTEM_PROMPT_CRITIC

        request = LLMRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=self.config.temperature_critique,
            max_tokens=agent.config.max_tokens,
        )

        try:
            response = await self.llm_client.generate(agent.model, request)

            if self._is_truncated(response):
                logger.warning(
                    "Agent %s critique TRUNCATED (finish_reason=length, %d tokens)",
                    agent.id, response.tokens_used,
                )

            parsed = parse_critique_response(response.content)

            if not solutions:
                return []

            other_solutions = [s for s in solutions if s.agent_id != agent.id]
            if not other_solutions:
                other_solutions = solutions

            parsed_critiques = parsed.get("critiques", [])
            pc_by_num = {pc.get("solution_num"): pc for pc in parsed_critiques if pc.get("solution_num")}
            result_critiques: list[Critique] = []
            total_bugs = 0

            for i, target_sol in enumerate(other_solutions):
                sol_num = i + 1
                pc = pc_by_num.get(sol_num)
                if pc is None:
                    pc = parsed_critiques[i] if i < len(parsed_critiques) else {}
                bugs = pc.get("bugs", [])
                total_bugs += len(bugs)

                ratings_parsed = bool(pc) and pc.get("ratings_parsed", True)
                if not ratings_parsed:
                    logger.warning(
                        "Ratings not parsed for agent %s critique of %s, using defaults",
                        agent.id, target_sol.agent_id,
                    )

                critique = Critique(
                    id=f"crit_{agent.id}_{target_sol.id}",
                    agent_id=agent.id,
                    solution_id=target_sol.id,
                    target_agent_id=target_sol.agent_id,
                    round_num=round_num if round_num > 0 else target_sol.round_num + 1,
                    overall_assessment=response.content,
                    bugs=[
                        Bug(description=b, severity=BugSeverity.MINOR)
                        for b in bugs
                    ],
                    correctness_rating=pc.get("correctness_rating", 5),
                    efficiency_rating=pc.get("efficiency_rating", 5),
                    readability_rating=pc.get("readability_rating", 5),
                    would_adopt=pc.get("would_adopt", False),
                    ratings_parsed=ratings_parsed,
                )
                result_critiques.append(critique)

            agent.stats.critiques_given += len(result_critiques)

            all_bug_descs = []
            for c in result_critiques:
                all_bug_descs.extend(b.description for b in c.bugs)

            message = AgentMessage(
                agent_id=agent.id,
                round_num=result_critiques[0].round_num if result_critiques else 1,
                message_type="critique",
                content=response.content,
                metadata={"bugs": all_bug_descs},
            )
            agent.add_message(message)

            if self.on_message:
                self.on_message(message)

            return result_critiques

        except Exception as e:
            logger.error(f"Agent {agent.id} failed to critique: {e}")
            return []
    
    async def _get_revision(
        self,
        agent: Agent,
        task: Task,
        own_solution: Solution,
        critiques: list[Critique],
        round_num: int,
        all_solutions: list[Solution] | None = None,
        all_critiques: list[Critique] | None = None,
        strategy: str = "",
        temperature_override: float | None = None,
        previous_critiques_summary: str = "",
    ) -> Solution | None:
        from .prompts import _format_test_feedback

        revision_temp = (
            temperature_override
            if temperature_override is not None
            else self.config.temperature_revision
        )

        try:
            import time

            if task.is_multi_file:
                code_files: dict[str, str] = {}
                total_gen_time = 0.0
                was_truncated = False

                test_feedback = _format_test_feedback(own_solution.execution_result)

                for filename in task.required_files:
                    prompt = build_chunked_file_revision_prompt(
                        task, filename, own_solution, critiques,
                        already_revised=code_files or None,
                        test_feedback=test_feedback,
                        strategy=strategy,
                    )
                    request = LLMRequest(
                        prompt=prompt,
                        system_prompt=SYSTEM_PROMPT_CODER,
                        temperature=revision_temp,
                        max_tokens=agent.config.max_tokens,
                    )

                    start = time.time()
                    response = await self.llm_client.generate(agent.model, request)
                    gen_time = time.time() - start
                    total_gen_time += gen_time

                    if self._is_truncated(response):
                        was_truncated = True
                        logger.warning(
                            "Agent %s revision for %s TRUNCATED "
                            "(finish_reason=length, %d tokens, %d chars)",
                            agent.id, filename,
                            response.tokens_used, len(response.content),
                        )

                    file_code = extract_code_from_response(response.content)
                    if not file_code:
                        logger.warning(
                            "Agent %s revision: no code extracted for %s",
                            agent.id, filename,
                        )
                    code_files[filename] = file_code

                agent.stats.total_generation_time += total_gen_time
                combined = "\n\n".join(
                    f"# FILE: {fn}\n{fc}" for fn, fc in code_files.items()
                )

                old_files = own_solution.extract_code_files()
                if code_files == old_files:
                    agent.stats.times_defended += 1
                else:
                    adopted = False
                    if all_solutions:
                        for other_sol in all_solutions:
                            if other_sol.agent_id != agent.id and other_sol.code_files:
                                if code_files == other_sol.extract_code_files():
                                    agent.stats.times_adopted_other += 1
                                    adopted = True
                                    break
                    if not adopted:
                        agent.stats.times_changed_mind += 1

                solution = Solution(
                    id=f"sol_{agent.id}_r{round_num}",
                    agent_id=agent.id,
                    round_num=round_num,
                    code=combined,
                    code_files=code_files,
                    is_revision=True,
                    parent_solution_id=own_solution.id,
                    generation_time=total_gen_time,
                    was_truncated=was_truncated,
                )
            else:
                prompt = build_revision_prompt(
                    task, own_solution, critiques,
                    all_solutions=all_solutions,
                    all_critiques=all_critiques,
                    show_all_solutions=self.config.revision_show_all_solutions,
                    strategy=strategy,
                    previous_critiques_summary=previous_critiques_summary,
                )
                request = LLMRequest(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT_CODER,
                    temperature=revision_temp,
                    max_tokens=agent.config.max_tokens,
                )

                start = time.time()
                response = await self.llm_client.generate(agent.model, request)
                generation_time = time.time() - start
                agent.stats.total_generation_time += generation_time

                was_truncated = self._is_truncated(response)
                if was_truncated:
                    logger.warning(
                        "Agent %s revision TRUNCATED "
                        "(finish_reason=length, %d tokens, %d chars)",
                        agent.id, response.tokens_used, len(response.content),
                    )

                code = extract_code_from_response(response.content)

                old_code = own_solution.extract_code_block()
                if code.strip() == old_code.strip():
                    agent.stats.times_defended += 1
                else:
                    adopted = False
                    if all_solutions:
                        for other_sol in all_solutions:
                            if other_sol.agent_id != agent.id:
                                if code.strip() == other_sol.extract_code_block().strip():
                                    agent.stats.times_adopted_other += 1
                                    adopted = True
                                    break
                    if not adopted:
                        agent.stats.times_changed_mind += 1

                solution = Solution(
                    id=f"sol_{agent.id}_r{round_num}",
                    agent_id=agent.id,
                    round_num=round_num,
                    code=code,
                    is_revision=True,
                    parent_solution_id=own_solution.id,
                    generation_time=generation_time,
                    was_truncated=was_truncated,
                )

            message = AgentMessage(
                agent_id=agent.id,
                round_num=round_num,
                message_type="revision",
                content=solution.code,
            )
            agent.add_message(message)

            if self.on_message:
                self.on_message(message)

            return solution

        except Exception as e:
            logger.error(f"Agent {agent.id} failed to revise: {e}")
            return None
    
    async def _get_vote(
        self,
        agent: Agent,
        task: Task,
        solutions: list[Solution],
        round_num: int,
    ) -> Vote | None:
        prompt = build_voting_prompt(task, solutions, agent.id)
        
        request = LLMRequest(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT_JUDGE,
            temperature=0.1,
        )
        
        try:
            response = await self.llm_client.generate(agent.model, request)
            parsed = parse_vote_response(response.content)
            
            vote_type = VoteType(parsed["vote_type"])
            voted_solution_id = None
            voted_agent_id = None
            
            if vote_type == VoteType.ADOPT and parsed.get("voted_solution"):
                sol_index = parsed["voted_solution"] - 1
                if 0 <= sol_index < len(solutions):
                    voted_solution_id = solutions[sol_index].id
                    voted_agent_id = solutions[sol_index].agent_id
            elif vote_type == VoteType.DEFEND:
                own_sol = next((s for s in solutions if s.agent_id == agent.id), None)
                if own_sol:
                    voted_solution_id = own_sol.id
                    voted_agent_id = agent.id

            if agent.role == AgentRole.JUDGE and vote_type == VoteType.DEFEND:
                logger.warning(
                    "Judge %s emitted DEFEND (round %d, no own solution) — rewriting to ABSTAIN",
                    agent.id, round_num,
                )
                vote_type = VoteType.ABSTAIN
                voted_solution_id = None
                voted_agent_id = None

            if voted_agent_id == agent.id and vote_type == VoteType.ADOPT:
                logger.warning(
                    "Agent %s attempted self-vote (round %d) — rewriting to ABSTAIN",
                    agent.id, round_num,
                )
                vote_type = VoteType.ABSTAIN
                voted_solution_id = None
                voted_agent_id = None

            parse_failed = (vote_type == VoteType.ADOPT and voted_solution_id is None)
            if parse_failed:
                logger.warning(
                    "Agent %s vote parse failed (round %d): %s",
                    agent.id, round_num, response.content[:200],
                )

            vote = Vote(
                id=f"vote_{agent.id}_r{round_num}",
                agent_id=agent.id,
                round_num=round_num,
                vote_type=vote_type,
                voted_solution_id=voted_solution_id,
                voted_agent_id=voted_agent_id,
                confidence=parsed.get("confidence", 0.5),
                reasoning=parsed.get("reasoning", ""),
                raw_response=response.content,
                parse_failed=parse_failed,
            )
            
            message = AgentMessage(
                agent_id=agent.id,
                round_num=round_num,
                message_type="vote",
                content=response.content,
            )
            agent.add_message(message)
            
            if self.on_message:
                self.on_message(message)
            
            return vote
            
        except Exception as e:
            logger.error(f"Agent {agent.id} failed to vote: {e}")
            return None
    
    def _get_solution_by_id(
        self,
        solutions: list[Solution],
        solution_id: str | None,
    ) -> Solution | None:
        if not solution_id:
            return None
        for sol in solutions:
            if sol.id == solution_id:
                return sol
        return None
    
    def _find_best_solution(self, solutions: list[Solution]) -> Solution | None:
        if not solutions:
            return None

        def score(s: Solution) -> tuple:
            pass_rate = s.pass_rate

            pylint_score = 0.0
            complexity_penalty = 0.0
            maintainability = 0.0

            if s.quality_metrics:
                pylint_score = s.quality_metrics.pylint_score / 10.0
                complexity_penalty = max(0, 1 - (s.quality_metrics.cyclomatic_complexity / 20.0))
                maintainability = s.quality_metrics.maintainability_index / 100.0

            votes = s.votes_received

            quality_score = (pylint_score + complexity_penalty + maintainability) / 3

            return (pass_rate, quality_score, votes)

        return max(solutions, key=score)

    def _find_best_solution_across_rounds(self, debate: Debate) -> Solution | None:
        all_solutions = []
        for rd in debate.rounds:
            all_solutions.extend(rd.solutions)
        if not all_solutions:
            return None
        best = self._find_best_solution(all_solutions)
        if best:
            logger.info(
                "Best solution across all rounds: %s (round %d, pass_rate=%.0f%%)",
                best.agent_id, best.round_num, best.pass_rate * 100,
            )
        return best
