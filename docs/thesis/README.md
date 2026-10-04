# Bachelor's thesis

**The Impact of Multi-Agent Debate on the Code Generation Effectiveness of Local Large Language Models (LLMs) Across Varying Task Complexities**

*Daugiaagentinių debatų įtaka lokalių didelių kalbos modelių (LLM) kodo generavimo efektyvumui skirtingo sudėtingumo užduotyse*

- **Author:** Lukaš Patrik Magalinski
- **Supervisor:** Prof. Dr. Aistis Raudys
- **Institution:** Vilnius University, Faculty of Mathematics and Informatics, study programme Informatics
- **Year:** 2026, Vilnius
- **Full text:** [Magalinski_2026_BSc_thesis_LT.pdf](Magalinski_2026_BSc_thesis_LT.pdf) (90 pages)

The thesis is written in Lithuanian and includes an English summary ("Reziume (Summary)"). The abstract below is a condensed version of that summary. Numbers are as printed in the thesis, with decimal points; where the summary's wording is imprecise, the precise definition is used and marked.

## Abstract

Open-weight models with 7-9 billion parameters that can run locally are an attractive alternative to closed commercial models, but on complex code-generation tasks they often produce incorrect, incomplete or poorly structured code. Multi-agent debate, in which several model instances iteratively propose, critique and revise each other's solutions, is a promising way to close this gap, but its behaviour with smaller open-weight models on multi-file code-generation tasks had been largely uninvestigated.

This work designs, implements and empirically evaluates a modular four-phase multi-agent debate system tailored to 7-9 billion parameter models. The system (about 27 thousand lines in total, in six layered packages) integrates an anti-regression policy, adaptive temperature, critique history, chunked multi-file generation and solution-visibility levels suited to the limited context window of small models. A custom benchmark of 90 unique tasks across four difficulty levels was developed, including 40 multi-file extreme-level tasks designed for this work.

Eleven SLURM experiments on the VU MIF high-performance computing cluster produced 960 run records (600 debates and 360 solo runs) across three comparison axes: proposer pool composition (7B vs 9B class), judge type (none, code-specialised, reasoning-specialised) and debate vs solo. A separate A/B ablation of the adaptive-temperature mechanism added four SLURM jobs and 240 paired ON/OFF comparisons, with a pooled effect of +3.99 percentage points (p = 0.010) and the largest difference in the Pool B no-judge configuration on extreme tasks (+10.6 percentage points, p = 0.029 before multiple-comparison correction). In total the experimental base comprises 15 SLURM jobs and 1200 unique run records.

Multi-agent debate outperformed the baseline (solo runs on `tasks/`, round-1 proposals on `tasks2/`) by 27.3 percentage points (mean pass rate 0.824 vs 0.551; paired Student's t-test t = 17.27, p < 10⁻³⁴; 108 wins, 12 ties and 0 losses over 120 task-set pairs). In the 282 debates in which no candidate in any round passed all tests, the final solution passed at least one test in 254 cases (90.1%; the summary calls these "initially failed cases"). The hypothesis that a 9B pool would outperform a 7B pool was not confirmed. The judge benefit reported for textual debates (+28 percentage points) did not transfer to code: with a code-specialised judge, Pool B (the pool with the higher solo mean) reached 0.722 versus 0.761 without a judge (−3.9 percentage points; one run per configuration). Adding a judge increased the number of bug statements in critiques, but hardly changed, and in some configurations lowered, the number of additionally passed tests (the thesis's "bugs fixed"); the latter correlated with the relative improvement at Pearson r = +0.446 (n = 600, p < 10⁻³⁰). A reasoning-specialised judge added 55-58% computation time without commensurate gains, static code quality (Pylint, Radon) was preserved within 4%, and the anti-regression policy, which returns the best solution found across rounds, ensured that no final result fell below the best first-round proposal in any of the 600 debates.

**Keywords:** multi-agent debate, large language models, open-weight models, code generation, LLM-as-Judge, multi-file programming tasks.

## Notes

- All result tables of the thesis are reproduced in English in [../RESULTS.md](../RESULTS.md), with the operational definitions of each metric.
- A post-hoc re-analysis of the same data, including clarifications to several statements of the thesis, is in [../RESULTS.md#post-hoc-re-analysis-october-2026](../RESULTS.md#post-hoc-re-analysis-october-2026).
