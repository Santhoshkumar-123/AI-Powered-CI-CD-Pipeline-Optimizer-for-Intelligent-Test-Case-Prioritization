# Requirements Document

## Introduction

Semanti-Q is an AI-powered test case prioritization system for regression testing in CI/CD pipelines. It combines semantic code embeddings with a dual-branch Deep Q-Network (DQN) to learn optimal test execution orderings that maximize early fault detection (measured by APFD). The current implementation has several critical gaps that prevent it from producing reproducible, scientifically valid results required for an IEEE paper. This document specifies the improvements needed across data generation, the RL reward function, baseline comparisons, model persistence, neural architecture novelty, and frontend enhancements. Note: the core frontend files (`api_client.py`, `components.py`, `config.py`) are already implemented; the gaps are in correctness, completeness, and new feature surfaces.

## Glossary

- **System**: The Semanti-Q backend service (Backend/main.py) and frontend dashboard (Frontend/).
- **Data_Generator**: The `generate_hybrid_dataset_final` function responsible for producing synthetic test execution records across multiple drift phases.
- **Drift_Phase**: A named period of code change activity targeting a specific system component (Backend, Frontend, Database, API, Mobile).
- **Semantic_Encoder**: The component that converts test metadata and commit context into a fixed-length semantic vector.
- **SemantiQ_Agent**: The dual-branch DQN PyTorch model that assigns priority scores to test cases.
- **Attention_Module**: The cross-attention fusion layer inside SemantiQ_Agent that combines statistical and semantic branch outputs.
- **Reward_Function**: The function that computes the RL training signal for each test ordering decision.
- **APFD**: Average Percentage of Faults Detected — the primary evaluation metric, defined as `1 - (sum of first-fault-detection positions) / (n * m) + 1/(2n)`.
- **APFD_Delta**: The change in APFD score caused by placing a specific test at a specific position in the execution order.
- **Baseline_Prioritizer**: Any of the three comparison strategies: Random, History-Based, or Coverage-Based.
- **Checkpoint**: A serialized PyTorch model file saved to disk that can be reloaded to restore trained weights.
- **SHAP_Explainer**: A SHAP TreeExplainer or DeepExplainer instance that computes feature-level attribution scores for the SemantiQ_Agent's output.
- **Drift_Detector**: A rolling-window variance monitor that signals concept drift in the incoming test result stream.
- **API_Client**: The `Frontend/api_client.py` module that wraps all HTTP calls from the Streamlit frontend to the FastAPI backend.
- **Dashboard**: The Streamlit application (`Frontend/app_enhanced.py`) that visualizes test prioritization results.

---

## Requirements

### Requirement 1: Five-Phase Drift Data Generation

**User Story:** As a researcher, I want the synthetic dataset to contain all five drift phases (Backend, Frontend, Database, API, Mobile), so that the paper's claim of evaluating across five system components is accurate and reproducible.

#### Acceptance Criteria

1. THE Data_Generator SHALL produce records with exactly five distinct `drift_phase` values: "Backend", "Frontend", "Database", "API", and "Mobile".
2. WHEN generating 100 build cycles, THE Data_Generator SHALL assign cycles 0–19 to the "Backend" drift phase, cycles 20–39 to "Frontend", cycles 40–59 to "Database", cycles 60–79 to "API", and cycles 80–99 to "Mobile".
3. THE Data_Generator SHALL assign a distinct `commit_focus` range to each drift phase such that each phase activates a different subset of the 10-dimensional semantic space (e.g., Backend uses dimensions 0–1, Frontend 2–3, Database 4–5, API 6–7, Mobile 8–9).
4. WHEN a drift phase is active, THE Data_Generator SHALL set the `semantic_vector` spike at the dimension corresponding to that phase's `commit_focus` range with a magnitude of at least 1.0.
5. IF the generated DataFrame does not contain all five drift phase values, THEN THE Data_Generator SHALL raise a `ValueError` with a descriptive message.

---

### Requirement 2: Context-Aware Semantic Encoding

**User Story:** As a researcher, I want semantic vectors to reflect actual code-drift context rather than random noise, so that the "semantic embedding" claim in the paper is scientifically defensible.

#### Acceptance Criteria

1. THE Semantic_Encoder SHALL compute each test's semantic vector as a function of the test's `focus_area`, the current cycle's `commit_focus`, and the test's `sensitivity_vector`, not as pure random noise.
2. WHEN two tests share the same `focus_area` and are executed in the same drift phase, THE Semantic_Encoder SHALL produce semantic vectors with a cosine similarity of at least 0.7.
3. WHEN two tests have different `focus_area` values that are at least 3 dimensions apart, THE Semantic_Encoder SHALL produce semantic vectors with a cosine similarity of at most 0.3.
4. THE Semantic_Encoder SHALL normalize each semantic vector to unit length before storing it in the dataset.
5. FOR ALL valid semantic vectors produced by THE Semantic_Encoder, the L2 norm SHALL equal 1.0 within a tolerance of 1e-5 (normalization invariant).

---

### Requirement 3: APFD-Delta Reward Shaping

**User Story:** As a researcher, I want the RL reward function to be based on APFD improvement rather than a static priority lookup, so that the agent genuinely learns to maximize early fault detection.

#### Acceptance Criteria

1. THE Reward_Function SHALL compute the reward for each test in a cycle as the APFD_Delta: the change in APFD score that results from placing that test at its assigned execution position compared to a random baseline ordering.
2. WHEN a failing test is placed in the top 10% of the execution order, THE Reward_Function SHALL return a positive reward greater than 0.
3. WHEN a passing test is placed in any position, THE Reward_Function SHALL return a reward of 0.0 or a small negative penalty not exceeding -0.1.
4. WHEN a failing test is placed in the bottom 50% of the execution order, THE Reward_Function SHALL return a negative reward less than 0.
5. THE Reward_Function SHALL clip all reward values to the range [-1.0, 1.0] to ensure training stability.
6. FOR ALL cycle orderings, the sum of APFD_Delta rewards across all tests in a cycle SHALL equal the difference between the cycle's APFD and the APFD of a uniform random ordering (reward additivity property).

---

### Requirement 4: Baseline Prioritization Implementations

**User Story:** As a researcher, I want Random, History-Based, and Coverage-Based baselines to be fully implemented and evaluated, so that the APFD comparison table in the paper reflects real computed values.

#### Acceptance Criteria

1. THE System SHALL implement a Random baseline that shuffles the test execution order uniformly at random using a fixed seed for reproducibility.
2. THE System SHALL implement a History-Based baseline that sorts tests in descending order of `fail_count_rolling` (most recently failing tests first).
3. THE System SHALL implement a Coverage-Based baseline that sorts tests in descending order of the product `complexity_score * code_churn` (highest estimated coverage impact first).
4. WHEN evaluated on the streaming test set (cycles 20–99), THE System SHALL compute and store the APFD score for each baseline on every cycle.
5. THE System SHALL expose the computed baseline APFD scores via the `/api/baselines/apfd` endpoint, returning a JSON object with keys "random", "history_based", "coverage_based", and "semanti_q", each mapping to a list of per-cycle APFD values.
6. WHEN the Random baseline is evaluated over 80 cycles, its mean APFD SHALL fall within the range [0.45, 0.60], consistent with theoretical expectation for random ordering.
7. THE History-Based baseline mean APFD SHALL be greater than the Random baseline mean APFD when evaluated on the same dataset.

---

### Requirement 5: APFD Calculation Correctness

**User Story:** As a researcher, I want the APFD formula and execution order assignment to be correct, so that reported metrics are scientifically valid.

#### Acceptance Criteria

1. THE System SHALL assign `execution_order` values as consecutive integers starting from 1, with no gaps or duplicates within a single cycle.
2. WHEN all failing tests are placed at positions 1 through m (the first m positions), THE System's APFD calculation SHALL return a value greater than or equal to `1 - m/(2*n)`.
3. WHEN all failing tests are placed at positions (n-m+1) through n (the last m positions), THE System's APFD calculation SHALL return a value less than or equal to `1/(2*n)`.
4. IF a cycle contains zero failing tests, THEN THE System SHALL return an APFD of 1.0 for that cycle.
5. FOR ALL valid test orderings, THE System's APFD calculation SHALL return a value in the range [0.0, 1.0].

---

### Requirement 6: Model Persistence via Checkpoints

**User Story:** As a developer, I want the trained SemantiQ_Agent weights to be saved to disk and reloaded on startup, so that the backend does not retrain from scratch on every restart.

#### Acceptance Criteria

1. WHEN training completes, THE System SHALL save the SemantiQ_Agent's state dictionary to a file at `Backend/checkpoints/semanti_q_agent.pt` using `torch.save`.
2. WHEN the backend starts and a checkpoint file exists at `Backend/checkpoints/semanti_q_agent.pt`, THE System SHALL load the checkpoint using `torch.load` and skip the training phase.
3. WHEN the backend starts and no checkpoint file exists, THE System SHALL run the full training pipeline and then save the resulting checkpoint.
4. FOR ALL valid checkpoint files, loading the checkpoint and running inference SHALL produce identical output to the original trained model on the same input (round-trip property).
5. THE System SHALL expose a `/api/model/checkpoint-status` endpoint that returns whether a checkpoint exists, the file size in bytes, and the timestamp of the last save.

---

### Requirement 7: Attention Mechanism in Fusion Layer

**User Story:** As a researcher, I want the SemantiQ_Agent to use a cross-attention fusion layer instead of simple concatenation, so that the architecture novelty claim is technically substantiated.

#### Acceptance Criteria

1. THE Attention_Module SHALL implement scaled dot-product attention over the statistical and semantic branch outputs, using the semantic branch output as the query and the statistical branch output as the key and value.
2. THE Attention_Module SHALL apply a softmax function to the attention scores before computing the weighted sum, ensuring attention weights sum to 1.0 for each sample.
3. WHEN the Attention_Module processes a batch of inputs, THE Attention_Module SHALL produce an output tensor of shape `(batch_size, fusion_dim)` where `fusion_dim` equals the statistical branch output dimension.
4. FOR ALL input batches, the attention weights produced by THE Attention_Module SHALL sum to 1.0 per sample within a tolerance of 1e-5 (softmax invariant).
5. THE System SHALL expose the attention weights for the most recent inference batch via the `/api/ai/attention-weights` endpoint for use in explainability visualizations.

---

### Requirement 8: Drift-Aware Adaptive Learning Rate

**User Story:** As a researcher, I want the agent's learning rate to adapt when concept drift is detected in the test result stream, so that the system can recover quickly from distribution shifts.

#### Acceptance Criteria

1. THE Drift_Detector SHALL monitor the rolling mean of `test_result` over a window of 10 consecutive cycles and signal drift when the absolute change in rolling mean exceeds 0.05.
2. WHEN THE Drift_Detector signals drift, THE System SHALL multiply the current optimizer learning rate by a factor of 2.0, up to a maximum learning rate of 0.01.
3. WHEN no drift is detected for 5 consecutive cycles after a drift event, THE System SHALL decay the learning rate back toward the base rate of 0.001 using exponential decay with factor 0.9 per cycle.
4. THE System SHALL log each drift detection event with the cycle ID, the rolling mean before and after, and the new learning rate.
5. THE System SHALL expose drift detection history via the `/api/model/drift-history` endpoint, returning a list of drift events with cycle ID, old LR, and new LR.

---

### Requirement 9: SHAP Explainability for Test Prioritization

**User Story:** As a researcher, I want SHAP feature attribution scores for each test prioritization decision, so that the paper can include an explainability analysis section.

#### Acceptance Criteria

1. THE SHAP_Explainer SHALL compute per-feature SHAP values for the SemantiQ_Agent's output score for each test in the most recent inference batch.
2. FOR ALL tests in an inference batch, the sum of SHAP values across all features SHALL equal the model output minus the expected model output (SHAP additivity property), within a tolerance of 1e-3.
3. THE System SHALL expose the top-5 most influential features (by mean absolute SHAP value) for the most recent cycle via the `/api/ai/shap-summary` endpoint.
4. WHEN THE SHAP_Explainer is called on a batch of fewer than 10 samples, THE System SHALL return a descriptive error rather than silently producing incorrect values.
5. THE System SHALL cache SHAP values per cycle and recompute them only when new simulation data is available, to avoid redundant computation.

---

### Requirement 10: End-to-End Frontend Operability

**User Story:** As a developer, I want the Streamlit frontend to start and display real data from the backend without errors, so that the system can be demonstrated end-to-end.

#### Acceptance Criteria

1. THE Dashboard SHALL successfully start via `streamlit run Frontend/app_enhanced.py` without raising any `ImportError` or `AttributeError` at startup.
2. WHEN the backend is running and has completed its simulation, THE Dashboard SHALL display non-zero values for total tests, tests prioritized, critical failures, and time saved.
3. WHEN the backend is not reachable, THE Dashboard SHALL display a clear error message and a retry button rather than crashing with an unhandled exception.
4. THE Dashboard SHALL render the five drift phase checkboxes (Backend, Frontend, Database, API, Mobile) in the sidebar filter panel, consistent with the five drift phases produced by the Data_Generator.
5. WHEN a user selects a drift phase filter and clicks "Apply Filters", THE Dashboard SHALL re-fetch data from the backend with the updated `drift_phase` query parameter and refresh all visualizations.
6. THE `render_sidebar_filters` function in `Frontend/components.py` SHALL NOT reference `session_state.filters` directly; it SHALL accept filter defaults as explicit parameters to avoid `AttributeError` when called from `app_enhanced.py`.
7. THE Dashboard SHALL expose a new "📊 Baselines" tab that renders the per-cycle APFD comparison chart for Random, History-Based, Coverage-Based, and Semanti-Q, sourced from the `/api/baselines/apfd` endpoint.
8. THE Dashboard SHALL expose a new "🔍 Explainability" tab that renders the SHAP feature importance bar chart and the attention weight heatmap for the most recent inference cycle.
