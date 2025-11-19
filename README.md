# awesome-ai-eval

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

Curated tools, frameworks, benchmarks, and observability platforms for evaluating LLMs, RAG systems, and agents and mitigate hallucination.

---

## Contents

- [1. Evaluators & Test Harnesses](#1-evaluators--test-harnesses)
- [2. RAG & Retrieval Evaluation](#2-rag--retrieval-evaluation)
- [3. Benchmarks & Leaderboards](#3-benchmarks--leaderboards)
- [4. Observability & Experimentation Platforms](#4-observability--experimentation-platforms)
- [5. Prompt & Prompt-System Evaluation](#5-prompt--prompt-system-evaluation)
- [6. Datasets, Papers & Methodology](#6-datasets-papers--methodology)
- [7. Starter Kits & Example Apps](#7-starter-kits--example-apps)
- [8. Related Awesome Lists & Collections](#8-related-awesome-lists--collections)

---

## 1. Evaluators & Test Harnesses

General-purpose frameworks you wire into your CI/CD, eval jobs, or offline experiments. Most support “LLM-as-a-judge” and custom metrics.

### 1.1 Core evaluation frameworks

- [DeepEval](https://github.com/confident-ai/deepeval) – Python framework for unit-testing LLM apps with 50+ research-backed metrics (hallucinations, coherence, toxicity, etc.). Integrates tightly with Confident AI’s hosted platform.
- [Ragas](https://github.com/explodinggradients/ragas) – End-to-end evaluation framework focused on LLM apps and RAG pipelines; supports LangChain, LlamaIndex, and custom graphs.
- [TruLens](https://github.com/truera/trulens) – Stack-agnostic eval and tracing for LLM apps and agents; “feedback functions” let you define custom quality metrics.
- [OpenAI Evals](https://github.com/openai/evals) – Official framework and registry for benchmarking LLMs and LLM-based systems; used internally by OpenAI and the community.
- [Promptfoo](https://github.com/promptfoo/promptfoo) – Local-first CLI and dashboard for evaluating prompts, RAG flows, and agents; integrates with GitHub Actions and CI.
- [MLflow](https://github.com/mlflow/mlflow) – Experiment-tracking framework with support for generative/LLM evaluation via the Eval API and custom evaluators.
- [Prompt Flow](https://github.com/microsoft/promptflow) – Microsoft’s flow-based LLM dev toolkit; includes evaluation flows and metrics for both local and Azure AI Studio workflows.
- [ZenML](https://github.com/zenml-io/zenml) – LLMOps platform that bakes evaluation into pipelines; supports retrieval, generation, and end-to-end app evaluation.
- [W&B Weave Evaluations](https://wandb.ai/site/evaluations/) – Evaluation framework on top of Weights & Biases for LLM apps, with datasets, scoring functions, and dashboards.
- [LLMbench (tooling)](https://www.deepchecks.com/llm-evaluation/best-tools/) – Emerging benchmarking tools branded as LLMbench for standardized tests across models and prompts (multiple OSS + commercial variants exist).

### 1.2 Application- and agent-level harnesses

- [LangSmith](https://smith.langchain.com/) – Hosted platform for tracing, evaluation, and dataset-based regression testing for LangChain apps and agents.
- [Braintrust](https://www.braintrust.dev/) – Evaluation and observability layer with trace-driven agent evals, GitHub Actions integration, and CI-style regression tests.
- [Deepchecks LLM Evaluation](https://www.deepchecks.com/) – End-to-end evaluation platform with SDK, dashboards, and built-in metrics; integrates with popular LLM stacks.
- [Opik](https://github.com/comet-ml/opik) – Open-source evaluation + observability platform for LLM apps; supports traces, datasets, metrics, and dashboards.
- [Langfuse](https://github.com/langfuse/langfuse) – OSS LLM engineering platform with tracing, evals, prompt management, and metrics for any framework.
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) – AI observability and evaluation toolkit built on OpenTelemetry; supports LLMs, RAG, and agents.
- [OpenLIT](https://github.com/openlit/openlit) – OpenTelemetry-native observability and evaluation for LLM apps with built-in hallucination/bias/toxicity metrics.
- [Maxim AI](https://www.getmaxim.ai/) – Hosted eval and observability platform focusing on simulation, LLM-as-a-judge, and agent-centric workflows.
- [Humanloop](https://humanloop.com/) – Production platform for prompt management, evals, and data collection with human feedback loops.
- [Galileo GenAI Studio](https://www.rungalileo.io/product/genai-studio) – Evaluation and error analysis for LLM outputs, with labeling and data curation features.

---

## 2. RAG & Retrieval Evaluation

Frameworks, datasets, and tools specifically targeting retrieval-augmented generation (RAG).

### 2.1 RAG evaluation frameworks

- [Ragas](https://github.com/explodinggradients/ragas) – Core RAG eval toolkit: context precision/recall, answer faithfulness, answer relevance, and pipeline-level metrics.
- [RAGEval](https://github.com/OpenBMB/RAGEval) – Scenario-specific RAG evaluation framework that auto-generates documents, questions, and metrics (Completeness/Hallucination/Irrelevance).
- [Open RAG Eval](https://github.com/vectara/open-rag-eval) – Vectara’s open framework for comparing RAG pipelines with standard metrics and ready-made datasets.
- [R-Eval](https://github.com/THU-KEG/R-Eval) – Toolkit for robust RAG evaluation, aligned with the “Evaluation of Retrieval-Augmented Generation” survey.
- [EvalScope RAG](https://evalscope.readthedocs.io/en/latest/blog/RAG/RAG_Evaluation.html) – RAG evaluation guide and tools built on top of Ragas and related work.
- [LlamaIndex Evaluation](https://docs.llamaindex.ai/en/stable/understanding/evaluating/index.html) – Evaluation modules for LlamaIndex pipelines (retrieval, query engines, and agents).

### 2.2 Retrieval & search quality

- [BEIR](https://github.com/beir-cellar/beir) – Large-scale evaluation benchmark for retrieval models across diverse tasks and domains.
- [MTEB](https://github.com/embeddings-benchmark/mteb) – Massive text embeddings benchmark with retrieval and semantic similarity tasks useful for RAG retrievers.
- [ColBERT and ColBERTv2](https://github.com/stanford-futuredata/ColBERT) – Late-interaction dense retrieval framework with evaluation scripts for IR benchmarks.

### 2.3 RAG-specific datasets & surveys

- [Awesome-RAG-Evaluation](https://github.com/YHPeter/Awesome-RAG-Evaluation) – Curated survey repo for RAG evaluation benchmarks and metrics.
- [RAG Evaluation Survey](https://arxiv.org/abs/2405.07437) – Comprehensive survey of RAG evaluation frameworks and metrics.
- [RAG evaluation papers in Awesome-RAG](https://github.com/liunian-Jay/Awesome-RAG) – Section collecting RAG eval papers such as RAGEval, MEMERAG, and others.

---

## 3. Benchmarks & Leaderboards

Reusable benchmarks and leaderboards for foundation models, code, safety, and multi-agent systems.

### 3.1 General LLM benchmarks

- [OpenAI Evals registry](https://github.com/openai/evals/tree/main/evals/elsuite) – Collection of community‐contributed evals across reasoning, QA, safety, etc.
- [BigBench / BIG-bench](https://github.com/google/BIG-bench) – Large, diverse benchmark suite for language models.
- [HELM](https://crfm.stanford.edu/helm/latest/) – Holistic Evaluation of Language Models: multi-metric benchmark across many tasks and scenarios.
- [LLM-Perf / lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) – Standard harness for evaluating autoregressive LMs on classic and new benchmarks.

### 3.2 Agent & tool-use benchmarks

- [AgentBench](https://github.com/THUDM/AgentBench) – Comprehensive benchmark for evaluating LLMs as agents across 8+ simulated environments.
- [GAIA](https://github.com/GAIA-benchmark/GAIA) – “General AI Assistants” benchmark: tool-use and reasoning tasks that require web access and multi-step planning.
- [MetaTool](https://github.com/meta-llama/MetaTool) – Tool-use benchmark and eval harness for Llama-based tool-calling models.
- [LLM Agents Benchmark Collections](https://llmbench.ai/) – Aggregated benchmarks focused on agent safety, reliability, and decision-making.

### 3.3 Safety, robustness & bias

- [Holistic Safety Evaluation (OpenAI/Anthropic style)](https://github.com/centerforaisafety/harmbench) – Benchmarks targeting harmful content, jailbreaks, and safety alignment.
- [ToxiGen](https://github.com/microsoft/ToxiGen) – Benchmark and dataset for measuring toxicity in language models.
- [BBQ (Bias Benchmark for QA)](https://github.com/nyu-mll/BBQ) – Benchmark for social bias in QA tasks.
- [Safety Benchmarks via Azure Content Safety & Bedrock Guardrails docs](https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/observability) – Evaluation workflows for safety and risk.

### 3.4 Code & math benchmarks

- [HumanEval](https://github.com/openai/human-eval) – Classic code generation benchmark for function-synthesis tasks.
- [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) – “Mostly Basic Programming Problems” benchmark.
- [LeetCode-style eval collections](https://github.com/alopatenko/LLMEvaluation) – Index of code-oriented LLM evaluation benchmarks such as LLMeBench and LM-PUB-QUIZ.

---

## 4. Observability & Experimentation Platforms

Platforms that combine **tracing, evaluation, analytics, and guardrails**. Many provide hosted and self-hosted options.

### 4.1 Open-source platforms

- [Langfuse](https://github.com/langfuse/langfuse) – OSS LLM engineering platform (traces, evals, prompt management, metrics) for any framework.
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) – AI observability and evaluation toolkit built on OpenTelemetry; supports training + inference + RAG workloads.
- [Opik](https://github.com/comet-ml/opik) – Open-source evaluation & observability platform for LLM applications, with dashboards and agent optimization tools.
- [OpenLIT](https://github.com/openlit/openlit) – OpenTelemetry-native observability + evaluation library for LLMs, vector DBs, and GPUs.
- [OpenLLMetry](https://github.com/traceloop/openllmetry) – OpenTelemetry instrumentation for LLM apps (traces + metrics), suitable as a base for custom eval flows.
- [VoltAgent / VoltOps](https://github.com/VoltAgent/voltagent) – TypeScript agent framework with built-in VoltOps observability platform for tracing and debugging agents.
- [PostHog LLM Observability](https://posthog.com/blog/llm-observability) – Analytics and event tracking extended to LLM apps; supports custom LLM eval metrics via events.
- [Zeno](https://zenoml.com/) – Data-centric evaluation/debugging tool for ML and LLM applications; especially strong at slicing outputs and spotting failure clusters.

### 4.2 Hosted & commercial eval/obs platforms

- [Confident AI](https://www.confident-ai.com/) – Full LLM evaluation platform built around DeepEval; provides regression suites, guardrails, and production monitoring.
- [Braintrust](https://www.braintrust.dev/) – Evaluation and observability layer with CI integration, remote agent evals, and trace-driven debugging.
- [LangSmith](https://smith.langchain.com/) – Hosted tracing + evals for LangChain apps, with datasets, annotation UIs, and CI hooks.
- [Humanloop](https://humanloop.com/) – Managed platform for prompt management, A/B testing, and evaluation with human feedback workflows.
- [Maxim AI](https://www.getmaxim.ai/) – End-to-end evaluation and observability infra with simulation, evals, and monitoring tailored to agentic applications.
- [Deepchecks LLM Evaluation](https://www.deepchecks.com/solutions/llm-evaluation/) – Production eval suite including datasets, metrics, dashboards, and safety/risk tools.
- [Galileo GenAI](https://www.rungalileo.io/product/genai-studio) – Evaluation and data debugging for LLM applications with labeling tools.
- [Datadog AI Observability](https://www.datadoghq.com/product/ai-observability/) – LLM monitoring and evaluation metrics integrated with broader infra monitoring.

### 4.3 Cloud provider evaluation services

- [Amazon Bedrock Evaluations](https://aws.amazon.com/bedrock/evaluations/) – Evaluate foundation models and RAG workflows using predefined and custom metrics, including safety via Guardrails.
- [Amazon Bedrock Guardrails](https://aws.amazon.com/bedrock/guardrails/) – Configurable safeguards that evaluate prompts and responses for safety and policy compliance.
- [Azure AI Foundry Evaluations](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/evaluate-generative-ai-app) – Cloud eval service with safety and quality metrics, integrated with Prompt Flow and agents.
- [Azure AI Foundry Risk & Safety Evaluations](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-ai-assisted-risk-and-safety-evaluations-in-azure-ai-foundry/4098595) – Automated safety evals to test generative apps before deployment.
- [Vertex AI Gen AI Evaluation](https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview) – Adaptive rubric-based evaluation, including model migration and third-party model evaluation.

---

## 5. Prompt & Prompt-System Evaluation

Tools dedicated to **prompt quality**, regression testing, and adversarial prompt attacks.

- [Promptfoo](https://github.com/promptfoo/promptfoo) – Prompt testing and red-teaming toolkit with GitHub Actions, code scanning, and dashboards.
- [ChainForge](https://github.com/ianarawjo/ChainForge) – Visual environment for comparing prompts, models, and responses; great for manual prompt experiments and hypothesis testing.
- [Prompt Flow](https://github.com/microsoft/promptflow) – Flow-based dev tool with explicit “evaluation flows” for systematic prompt testing and batch evals.
- [OpenAI Evals prompt tests](https://github.com/openai/evals) – Includes numerous prompt-level benchmarks and examples for classification, extraction, and reasoning tasks.
- [Guardrails AI](https://github.com/ShreyaR/guardrails) – Validation and correction framework for LLM outputs; can be wired into evaluation pipelines to enforce and measure contract adherence.
- [Prompt injection / jailbreak test suites](https://github.com/centerforaisafety/harmbench) – Datasets and tools for evaluating prompt injection and jailbreak robustness.

---

## 6. Datasets, Papers & Methodology

Resources to **design** better evals: metrics, methodologies, and surveys.

- [RAG Evaluation Survey: Framework, Metrics, and Methods](https://arxiv.org/abs/2405.07437) – Deep dive into RAG eval, with taxonomy and open problems.
- [RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework](https://arxiv.org/abs/2408.01262) – Framework for generating domain-specific RAG eval datasets and metrics.
- [HELM](https://crfm.stanford.edu/helm/latest/) – Methodology and benchmark for holistic evaluation across many axes (accuracy, calibration, robustness, fairness, etc.).
- [LLM Evaluation Tools Roundups](https://www.deepchecks.com/llm-evaluation/best-tools/) – Regularly updated overviews of LLM eval and observability tools across OSS and SaaS.
- [ZenML Evaluation Playbook](https://www.zenml.io/blog/the-evaluation-playbook-making-llms-production-ready) – Practical guide to integrating evaluation into LLM pipelines and RAG systems.
- [W&B Evaluations Docs](https://docs.wandb.ai/weave/guides/core-types/evaluations) – Evaluation-driven development workflow patterns for LLM apps.
- [Evaluation Guides for Cloud Platforms](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/evaluate-generative-ai-app) – Azure, Vertex AI, and Bedrock docs with concrete evaluation recipes and best practices.
- [EvalScope Blogs](https://evalscope.readthedocs.io/en/latest/blog/RAG/RAG_Evaluation.html) – Hands-on guides for RAG evaluation using Ragas and related tools.

---

## 7. Starter Kits & Example Apps

Ready-to-run repos that demonstrate **evaluation in context** for RAG, agents, and apps.

- [Arize Phoenix AI Chatbot](https://github.com/Arize-ai/phoenix-ai-chatbot) – Next.js chatbot instrumented with Phoenix for full tracing and evaluation.
- [Deepchecks QA-over-CSV Demo](https://github.com/deepchecks/qa-over-csv) – Example agent app wired to Deepchecks LLM evaluation SDK.
- [Azure LLM Evaluation Samples](https://github.com/Azure-Samples/llm-evaluation) – Azure AI Foundry examples: evaluation projects, Prompt Flow integration, and safety evals.
- [Promptfoo Example Repos](https://github.com/promptfoo/promptfoo/tree/main/examples) – Example test suites for prompts, RAG, and retrieval quality.
- [OpenAI Evals Demo Evals](https://github.com/withmartian/demo-evals) – Example OpenAI evals showing how to structure custom benchmarks.
- [ZenML LLMOps Examples](https://github.com/zenml-io/zenml-projects) – End-to-end pipelines including evaluation for RAG, agents, and classic ML.

---

## 8. Related Awesome Lists & Collections

Other curated lists that overlap; useful for deep dives into niches:

- [Awesome-RAG-Evaluation](https://github.com/YHPeter/Awesome-RAG-Evaluation) – Benchmarks, datasets, and metrics specifically for RAG systems.
- [Awesome-RAG](https://github.com/noworneverev/Awesome-RAG) – Broad RAG list, including a dedicated RAG evaluation section.
- [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM) – General LLM list; several sections on evaluation, benchmarks, and leaderboards.
- [Awesome ChainForge](https://github.com/loloMD/awesome_chainforge) – Ecosystem list around the ChainForge prompt-testing tool.
- [Awesome GenAI / LLMOps Lists](https://github.com/eugeneyan/genai-notes) – Broader collections with subsections on eval, tracing, and LLMOps.

---

## Contributing

Contributions are welcome!

- Add tools that are **actively maintained** and have clear documentation or examples.
- Prefer entries that:
  - Are used in production or referenced in recent surveys/blogs.
  - Provide *concrete* evaluation functionality (metrics, harnesses, dashboards), not just generic libraries.
- Avoid duplicates: a project should appear in the **single most relevant section**.

Please follow the [Awesome List guidelines](https://github.com/sindresorhus/awesome/blob/main/pull_request_template.md) and keep entries in alphabetical order within each subsection.

---

## Why this list?

Most AI apps now look like some combination of:

- RAG pipelines (LangChain, LlamaIndex, custom graphs)
- Tool-using agents (LangGraph, crewAI-style, custom orchestrators)
- Cloud-hosted models (OpenAI, Anthropic, Gemini, Bedrock, Vertex, Azure)

This list is meant to be the **fastest path** from “I know I should evaluate my system” to “I have a concrete tool wired into my stack and CI/CD.”
