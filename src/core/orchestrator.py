"""
Debate orchestrator - coordinates the entire debate process.
"""
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
    build_critique_prompt,
    build_proposal_prompt,
    build_multi_file_proposal_prompt,
    build_revision_prompt,
    build_voting_prompt,
    extract_code_from_response,
    extract_multi_file_code_from_response,
    parse_critique_response,
    parse_vote_response,
)

logger = logging.getLogger(__name__)


class DebateOrchestrator:
    """
    Orchestrates debates between multiple LLM agents.
    
    Manages the entire debate lifecycle:
    1. Initialize agents
    2. Round 1: Collect initial proposals
    3. Rounds 2-N: Critique → Revise → Vote loop
    4. Detect consensus or reach max rounds
    5. Finalize and report results
    """
    
    def __init__(
        self,
        llm_client: MultiModelClient,
        config: DebateConfig | None = None,
        on_round_complete: Callable[[RoundSummary], None] | None = None,
        on_message: Callable[[AgentMessage], None] | None = None,
        on_phase: Callable[[str, int], None] | None = None,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            llm_client: Client for LLM inference.
            config: Debate configuration.
            on_round_complete: Callback when a round completes.
            on_message: Callback when an agent sends a message.
        """
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
        """
        Run a complete debate on a task.
        
        Args:
            task: The coding task to solve.
            agent_configs: Configuration for each agent.
            
        Returns:
            Debate object with all results.
        """
        # Initialize debate
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
            # Round 1: Initial proposals
            round1 = await self._run_proposal_round(debate)
            debate.add_round(round1)
            
            if self.on_round_complete:
                self.on_round_complete(round1)
            
            # Check for early stop (perfect solution in round 1)
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
            
            # Rounds 2-N: Debate loop
            for round_num in range(2, self.config.max_rounds + 1):
                # Check if stop was requested
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
                
                # Check consensus
                if round_summary.consensus_result and round_summary.consensus_result.reached:
                    winning_solution = self._get_solution_by_id(
                        round_summary.solutions,
                        round_summary.consensus_result.winning_solution_id
                    )
                    if not winning_solution:
                        winning_solution = self._find_best_solution(round_summary.solutions)
                    debate.finalize(
                        status=DebateStatus.CONSENSUS_REACHED,
                        final_solution=winning_solution,
                        consensus=round_summary.consensus_result,
                    )
                    logger.info(f"Consensus reached in round {round_num}")
                    return debate
                
                # Check early stop conditions
                should_stop, reason = self.consensus_detector.should_early_stop(
                    round_summary.solutions,
                    round_summary.votes,
                    round_num,
                    self.config.min_rounds,
                )
                if should_stop:
                    # Find best solution
                    best = self._find_best_solution(round_summary.solutions)
                    debate.finalize(
                        status=DebateStatus.EARLY_STOP,
                        final_solution=best,
                    )
                    logger.info(f"Early stop: {reason}")
                    return debate
            
            # Max rounds reached
            final_round = debate.rounds[-1]
            best_solution = self._find_best_solution(final_round.solutions)
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
        """
        Run a single agent solving a task with no debate (baseline mode).

        One agent, one round, no critique/revision/voting.
        Returns a Debate object for compatibility with DB and metrics.
        """
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
        """Create agent instances from configurations."""
        agents = []
        for i, config in enumerate(configs):
            agent_id = f"agent_{i+1}_{config.model.split(':')[0]}"
            agents.append(Agent(id=agent_id, config=config))
        return agents
    
    async def _run_proposal_round(self, debate: Debate) -> RoundSummary:
        """Run the initial proposal round."""
        logger.info("Running proposal round (Round 1)")
        if self.on_phase:
            self.on_phase("propose", 1)
        round_summary = RoundSummary(round_num=1, solutions=[], critiques=[], votes=[])
        
        # Collect proposals from all agents concurrently
        tasks = []
        for agent in debate.agents:
            if agent.role != AgentRole.JUDGE:  # Judges don't propose
                tasks.append(self._get_proposal(agent, debate.task))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        solutions = [r for r in results if isinstance(r, Solution)]

        # Test each solution
        for solution in solutions:
            if solution:
                result = await self.executor.execute(solution, debate.task)
                solution.execution_result = result
                
                # Analyze quality
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
        """Run a debate round (critique → revise → vote)."""
        logger.info(f"Running debate round {round_num}")
        
        round_summary = RoundSummary(
            round_num=round_num,
            solutions=[],
            critiques=[],
            votes=[],
        )
        
        # Get current solutions (from previous round)
        current_solutions = debate.get_latest_solutions()
        
        # Phase 1: Critique
        if self.on_phase:
            self.on_phase("critique", round_num)
        critique_tasks = []
        for agent in debate.agents:
            if agent.role != AgentRole.PROPOSER:  # Proposers don't critique
                critique_tasks.append(
                    self._get_critique(agent, debate.task, current_solutions)
                )
        
        critique_results = await asyncio.gather(*critique_tasks, return_exceptions=True)
        # Each result is a list[Critique] or an Exception; flatten valid ones
        for result in critique_results:
            if isinstance(result, list):
                round_summary.critiques.extend(result)
        
        # Group critiques by target agent
        critiques_by_agent: dict[str, list[Critique]] = {}
        for critique in round_summary.critiques:
            target = critique.target_agent_id
            if target not in critiques_by_agent:
                critiques_by_agent[target] = []
            critiques_by_agent[target].append(critique)
        
        # Phase 2: Revise based on critiques
        if self.on_phase:
            self.on_phase("revise", round_num)
        revision_tasks = []
        for agent in debate.agents:
            if agent.role != AgentRole.JUDGE:
                own_solution = next(
                    (s for s in current_solutions if s.agent_id == agent.id),
                    None
                )
                # If agent has no solution, adopt the best available one
                if not own_solution and current_solutions:
                    own_solution = self._find_best_solution(current_solutions)
                    logger.info(
                        f"Agent {agent.id} has no solution, adopting from {own_solution.agent_id}"
                    )
                if own_solution:
                    agent_critiques = critiques_by_agent.get(agent.id, [])
                    revision_tasks.append(
                        self._get_revision(
                            agent, debate.task, own_solution, agent_critiques, round_num,
                            all_solutions=current_solutions,
                            all_critiques=round_summary.critiques,
                        )
                    )
        
        revision_results = await asyncio.gather(*revision_tasks, return_exceptions=True)
        revised_solutions = [r for r in revision_results if isinstance(r, Solution)]

        # Test revised solutions
        for solution in revised_solutions:
            if solution:
                result = await self.executor.execute(solution, debate.task)
                solution.execution_result = result
                
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
                    f"{result.tests_passed}/{result.tests_total} tests passed"
                )
        
        # Phase 3: Vote
        if self.on_phase:
            self.on_phase("vote", round_num)
        vote_tasks = []
        for agent in debate.agents:
            vote_tasks.append(
                self._get_vote(agent, debate.task, round_summary.solutions, round_num)
            )
        
        vote_results = await asyncio.gather(*vote_tasks, return_exceptions=True)
        round_summary.votes = [v for v in vote_results if isinstance(v, Vote)]
        
        # Detect consensus
        round_summary.consensus_result = self.consensus_detector.detect(
            round_summary.votes,
            round_summary.solutions,
            debate.agents,
            round_num,
        )
        
        round_summary.end_time = datetime.now()
        return round_summary
    
    async def _get_proposal(self, agent: Agent, task: Task) -> Solution | None:
        """Get initial proposal from an agent."""
        if task.is_multi_file:
            prompt = build_multi_file_proposal_prompt(task)
        else:
            prompt = build_proposal_prompt(task)

        request = LLMRequest(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT_CODER,
            temperature=self.config.temperature_initial,
        )

        try:
            import time
            start = time.time()
            response = await self.llm_client.generate(agent.model, request)
            generation_time = time.time() - start

            if task.is_multi_file:
                code_files = extract_multi_file_code_from_response(
                    response.content, task.required_files
                )
                combined = "\n\n".join(
                    f"# FILE: {fn}\n{fc}" for fn, fc in code_files.items()
                )
                solution = Solution(
                    id=f"sol_{agent.id}_r1",
                    agent_id=agent.id,
                    round_num=1,
                    code=combined,
                    code_files=code_files,
                    generation_time=generation_time,
                )
            else:
                code = extract_code_from_response(response.content)
                solution = Solution(
                    id=f"sol_{agent.id}_r1",
                    agent_id=agent.id,
                    round_num=1,
                    code=code,
                    generation_time=generation_time,
                )

            # Record message
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
    ) -> list[Critique]:
        """Get critiques from an agent for all other solutions.

        Returns one Critique object per non-own solution. The LLM receives
        all solutions in a single prompt and returns critiques for each.
        The parsed response is matched to solutions by index.
        """
        prompt = build_critique_prompt(task, solutions, agent.id)

        system_prompt = SYSTEM_PROMPT_JUDGE if agent.role == AgentRole.JUDGE else SYSTEM_PROMPT_CRITIC

        request = LLMRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=self.config.temperature_critique,
        )

        try:
            response = await self.llm_client.generate(agent.model, request)
            parsed = parse_critique_response(response.content)

            if not solutions:
                return []

            # Get solutions to critique (all except own)
            other_solutions = [s for s in solutions if s.agent_id != agent.id]
            if not other_solutions:
                other_solutions = solutions  # Edge case: critique own if only one

            parsed_critiques = parsed.get("critiques", [])
            result_critiques: list[Critique] = []
            total_bugs = 0

            for i, target_sol in enumerate(other_solutions):
                # Match parsed critique by index if available, else use defaults
                pc = parsed_critiques[i] if i < len(parsed_critiques) else {}
                bugs = pc.get("bugs", [])
                total_bugs += len(bugs)

                ratings_parsed = pc.get("ratings_parsed", True)
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
                    round_num=target_sol.round_num + 1,
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

            # Update stats
            agent.stats.critiques_given += len(result_critiques)
            agent.stats.bugs_found += total_bugs

            # Record message
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
    ) -> Solution | None:
        """Get revision from an agent based on critiques.

        The agent sees all solutions and all critiques, so it can adopt
        another agent's solution if it's clearly better.
        """
        prompt = build_revision_prompt(
            task, own_solution, critiques,
            all_solutions=all_solutions,
            all_critiques=all_critiques,
        )

        request = LLMRequest(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT_CODER,
            temperature=self.config.temperature_revision,
        )

        try:
            import time
            start = time.time()
            response = await self.llm_client.generate(agent.model, request)
            generation_time = time.time() - start

            if task.is_multi_file:
                code_files = extract_multi_file_code_from_response(
                    response.content, task.required_files
                )
                combined = "\n\n".join(
                    f"# FILE: {fn}\n{fc}" for fn, fc in code_files.items()
                )

                # Detect: defended, changed mind, or adopted
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
                    generation_time=generation_time,
                )
            else:
                code = extract_code_from_response(response.content)

                # Detect: defended, changed mind, or adopted another's solution
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
                )

            # Record message
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
        """Get vote from an agent."""
        prompt = build_voting_prompt(task, solutions, agent.id)
        
        request = LLMRequest(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT_JUDGE,
            temperature=0.1,  # Low temperature for deterministic voting
        )
        
        try:
            response = await self.llm_client.generate(agent.model, request)
            parsed = parse_vote_response(response.content)
            
            # Map vote to solution
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
            
            # Record message
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
        """Find solution by ID."""
        if not solution_id:
            return None
        for sol in solutions:
            if sol.id == solution_id:
                return sol
        return None
    
    def _find_best_solution(self, solutions: list[Solution]) -> Solution | None:
        """Find the best solution based on test results, code quality, and votes."""
        if not solutions:
            return None

        def score(s: Solution) -> tuple:
            # 1. Test pass rate (most important)
            pass_rate = s.pass_rate

            # 2. Code quality metrics
            pylint_score = 0.0
            complexity_penalty = 0.0
            maintainability = 0.0

            if s.quality_metrics:
                pylint_score = s.quality_metrics.pylint_score / 10.0  # Normalize to 0-1
                # Lower complexity is better, invert and normalize
                complexity_penalty = max(0, 1 - (s.quality_metrics.cyclomatic_complexity / 20.0))
                maintainability = s.quality_metrics.maintainability_index / 100.0  # Normalize to 0-1

            # 3. Votes received
            votes = s.votes_received

            # Combined score: tests (50%) + quality (30%) + votes (20%)
            quality_score = (pylint_score + complexity_penalty + maintainability) / 3

            # Return tuple for comparison (higher is better for all)
            return (pass_rate, quality_score, votes)

        return max(solutions, key=score)
