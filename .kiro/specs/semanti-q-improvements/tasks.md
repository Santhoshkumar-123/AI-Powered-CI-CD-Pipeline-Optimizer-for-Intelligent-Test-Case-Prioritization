# Implementation Plan: Semanti-Q Improvements

## Overview

Implement ten targeted improvements to the Semanti-Q test prioritization system in dependency order:
data generation → semantic encoding → APFD correctness → reward shaping → baselines →
model persistence → cross-attention fusion → drift-aware LR → SHAP explainability → frontend.

All backend changes go into `Backend/main.py`. Frontend changes go into
`Frontend/app_enhanced.py`, `Frontend/components.py`, and `Frontend/api_client.py`.

---

## Tasks

- [x] 1. Five-phase drift data generation
  - [x] 1.1 Rewrite `generate_hybrid_dataset_final` in `Backend/main.py` to produce exactly five drift phases
    - Replace the two-phase (Backend/Frontend) logic with a deterministic five-phase mapping:
      cycles 0–19 → Backend (dims 0–1), 20–39 → Frontend (2–3), 40–59 → Database (4–5),
      60–79 → API (6–7), 80–99 → Mobile (8–9)
    - Set `commit_focus = phase_start_dim + (cycle % 2)` for deterministic spike placement
    - After building the DataFrame, validate `set(df['drift_phase'].unique()) == {"Backend","Frontend","Database","API","Mobile"}` and raise `ValueError` if not
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 1.2 Write property test for five-phase coverage (Property 1)
    - **Property 1: Five-phase drift coverage invariant**
    - **Validates: Requirements 1.1, 1.5**
    - Use `@settings(max_examples=100)` and `st.integers(min_value=100, max_value=600)` for n_tests
    - Assert `set(df['drift_phase'].unique()) == {"Backend","Frontend","Database","API","Mobile"}`

  - [ ]* 1.3 Write property test for cycle-to-phase assignment (Property 2)
    - **Property 2: Cycle-to-phase assignment invariant**
    - **Validates: Requirements 1.2**
    - For each row, assert `drift_phase == PHASES[cycle_id // 20]`

  - [ ]* 1.4 Write property test for semantic spike alignment (Property 3)
    - **Property 3: Semantic spike alignment**
    - **Validates: Requirements 1.3, 1.4**
    - For each row, assert `argmax(semantic_vector)` falls within the two-dim range for that phase

- [x] 2. Context-aware semantic encoding
  - [x] 2.1 Replace the random-noise semantic vector construction in `generate_hybrid_dataset_final`
    - Implement: `base = sensitivity_vector * dot(sensitivity_vector, commit_focus_vector)` where
      `commit_focus_vector` is a zero vector with 1.2 at the active `commit_focus` dimension
    - Add `noise = np.random.normal(0, 0.05, 10)`, then `raw = base + noise`
    - L2-normalize: `semantic_vector = raw / np.linalg.norm(raw)`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ]* 2.2 Write property test for semantic vector normalization (Property 4)
    - **Property 4: Semantic vector normalization invariant**
    - **Validates: Requirements 2.4, 2.5**
    - For every row in the generated dataset, assert `abs(np.linalg.norm(vec) - 1.0) < 1e-5`

  - [ ]* 2.3 Write property test for same-phase cosine similarity (Property 5)
    - **Property 5: Same-phase cosine similarity**
    - **Validates: Requirements 2.2**
    - Sample pairs of tests with the same `focus_area` and `drift_phase`; assert cosine similarity ≥ 0.7

  - [ ]* 2.4 Write property test for different-phase cosine dissimilarity (Property 6)
    - **Property 6: Different-phase cosine dissimilarity**
    - **Validates: Requirements 2.3**
    - Sample pairs of tests whose `focus_area` dimensions differ by ≥ 3; assert cosine similarity ≤ 0.3

- [x] 3. APFD calculation correctness fix
  - [x] 3.1 Fix `execution_order` assignment in `run_online_simulation` in `Backend/main.py`
    - Replace any existing assignment with `prioritized_data['execution_order'] = range(1, len(prioritized_data) + 1)`
    - Ensure `calculate_apfd` returns `1.0` immediately when `m == 0` (no failures)
    - _Requirements: 5.1, 5.4_

  - [ ]* 3.2 Write property test for execution order permutation (Property 14)
    - **Property 14: Execution order is a valid permutation**
    - **Validates: Requirements 5.1**
    - For any prioritized cycle DataFrame of size n, assert `execution_order` is a permutation of [1..n]

  - [ ]* 3.3 Write property test for APFD range (Property 15)
    - **Property 15: APFD range invariant**
    - **Validates: Requirements 5.5**
    - For random orderings with random fail positions, assert `0.0 <= calculate_apfd(df) <= 1.0`

- [x] 4. APFD-delta reward shaping
  - [x] 4.1 Replace the priority-lookup reward in `run_online_simulation` with APFD-delta reward
    - After sorting by `ai_score`, compute `apfd_agent = calculate_apfd(prioritized_data)`
    - Compute `apfd_random` as APFD of a randomly shuffled copy of the same cycle data
    - For each test: if `test_result == 0` (failing), compute
      `reward = (apfd_agent - apfd_random) * position_weight(k, n)` where
      `position_weight(k, n) = 2 * (n - k + 1) / (n * (n + 1))`; if passing, `reward = 0.0`
    - Clip all rewards to `[-1.0, 1.0]`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [ ]* 4.2 Write property test for reward sign correctness (Property 7)
    - **Property 7: Reward sign correctness**
    - **Validates: Requirements 3.2, 3.3, 3.4**
    - Generate random cycle orderings with known fail positions; assert sign rules hold

  - [ ]* 4.3 Write property test for reward clipping (Property 8)
    - **Property 8: Reward clipping invariant**
    - **Validates: Requirements 3.5**
    - Use `st.floats(min_value=-100, max_value=100)`; assert clipped value is in `[-1.0, 1.0]`

  - [ ]* 4.4 Write property test for reward additivity (Property 9)
    - **Property 9: Reward additivity**
    - **Validates: Requirements 3.6**
    - For random cycle DataFrames, assert `sum(rewards) ≈ apfd_agent - apfd_random` within 1e-4

- [ ] 5. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Baseline prioritization implementations
  - [x] 6.1 Add three baseline functions to `Backend/main.py`
    - `random_baseline(df, seed=42) -> pd.DataFrame`: shuffle with fixed seed
    - `history_baseline(df) -> pd.DataFrame`: sort descending by `fail_count_rolling`
    - `coverage_baseline(df) -> pd.DataFrame`: sort descending by `complexity_score * code_churn`
    - _Requirements: 4.1, 4.2, 4.3_

  - [x] 6.2 Compute and store per-cycle baseline APFD scores during `run_online_simulation`
    - For each cycle in the stream, call all three baselines and `calculate_apfd` on each result
    - Store results in `GlobalState.baseline_apfd_history` as `{"random": [...], "history_based": [...], "coverage_based": [...], "semanti_q": [...]}`
    - Add `baseline_apfd_history: Dict[str, List[float]]` field to `GlobalState`
    - _Requirements: 4.4, 4.5_

  - [x] 6.3 Add `GET /api/baselines/apfd` endpoint to `Backend/main.py`
    - Return `GlobalState.baseline_apfd_history` as JSON
    - Return HTTP 503 if simulation has not completed yet
    - _Requirements: 4.5_

  - [ ]* 6.4 Write property test for random baseline reproducibility (Property 10)
    - **Property 10: Random baseline reproducibility**
    - **Validates: Requirements 4.1**
    - Call `random_baseline(df, seed=42)` twice; assert identical orderings

  - [ ]* 6.5 Write property test for history baseline ordering (Property 11)
    - **Property 11: History baseline ordering invariant**
    - **Validates: Requirements 4.2**
    - Assert output is sorted in non-increasing order of `fail_count_rolling`

  - [ ]* 6.6 Write property test for coverage baseline ordering (Property 12)
    - **Property 12: Coverage baseline ordering invariant**
    - **Validates: Requirements 4.3**
    - Assert output is sorted in non-increasing order of `complexity_score * code_churn`

  - [ ]* 6.7 Write property test for history-beats-random APFD (Property 13)
    - **Property 13: History baseline outperforms random**
    - **Validates: Requirements 4.7**
    - On the full generated dataset, assert `mean(history_apfd) > mean(random_apfd)` over cycles 20–99

- [x] 7. Model persistence via checkpoints
  - [x] 7.1 Add checkpoint save/load logic to `run_simulation_background` in `Backend/main.py`
    - Define `CHECKPOINT_PATH = "Backend/checkpoints/semanti_q_agent.pt"`
    - Call `os.makedirs("Backend/checkpoints", exist_ok=True)` at startup
    - If checkpoint exists: `agent.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))` and skip training
    - If not: run training loop, then `torch.save(agent.state_dict(), CHECKPOINT_PATH)`
    - Catch `RuntimeError` from corrupt checkpoint, log warning, fall back to full retraining
    - _Requirements: 6.1, 6.2, 6.3_

  - [x] 7.2 Add checkpoint metadata fields to `GlobalState` and expose status endpoint
    - Add `checkpoint_saved_at: Optional[datetime]` and `checkpoint_size_bytes: int` to `GlobalState`
    - Set these fields after saving the checkpoint
    - Add `GET /api/model/checkpoint-status` endpoint returning `{exists, size_bytes, saved_at}`
    - _Requirements: 6.5_

  - [ ]* 7.3 Write property test for checkpoint round-trip (Property 16)
    - **Property 16: Checkpoint round-trip**
    - **Validates: Requirements 6.4**
    - For random float tensors of correct shape, assert model output before save equals output after load within 1e-6

- [x] 8. Cross-attention fusion layer in SemantiQ_Agent
  - [x] 8.1 Add `CrossAttentionFusion` module and replace `fusion_layer` in `SemantiQ_Agent`
    - Implement `CrossAttentionFusion(nn.Module)` with `dim=32`:
      - `score = (sem_feat * stat_feat).sum(dim=-1, keepdim=True) / (dim ** 0.5)`
      - `attn = torch.sigmoid(score)` — scalar gate per sample
      - `fused = attn * stat_feat + (1 - attn) * sem_feat`
      - Return `(self.out(fused), attn)` where `self.out = nn.Linear(dim, 1)`
    - Update `SemantiQ_Agent.forward` to use `CrossAttentionFusion` instead of `torch.cat` + linear
    - Store attention weights in `GlobalState.last_attention_weights` after each inference batch
    - Add `last_attention_weights: Optional[np.ndarray]` to `GlobalState`
    - _Requirements: 7.1, 7.2, 7.3, 7.5_

  - [x] 8.2 Add `GET /api/ai/attention-weights` endpoint
    - Return `{"weights": [...], "cycle_id": <int or null>}` from `GlobalState.last_attention_weights`
    - Return empty weights list if no inference has run yet
    - _Requirements: 7.5_

  - [ ]* 8.3 Write property test for attention weight validity (Property 17)
    - **Property 17: Attention weight softmax invariant**
    - **Validates: Requirements 7.2, 7.4**
    - For random batches of size 1–64, assert attention gate values lie in (0, 1) and fused output is a valid convex combination

  - [ ]* 8.4 Write property test for attention output shape (Property 18)
    - **Property 18: Attention output shape invariant**
    - **Validates: Requirements 7.3**
    - Use `st.integers(min_value=1, max_value=64)` for batch size; assert score shape is `(B, 1)` and attn shape is `(B, 1)`

- [ ] 9. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Drift-aware adaptive learning rate and DriftDetector
  - [x] 10.1 Implement `DriftDetector` class in `Backend/main.py`
    - `__init__(self, window=10, threshold=0.05)`: store window, threshold, `deque(maxlen=window)`, `prev_mean=None`
    - `update(self, cycle_pass_rate: float) -> bool`: append to history; return `False` until window is full;
      compute `current_mean`; if `prev_mean` is None set it and return `False`;
      detect drift as `abs(current_mean - prev_mean) > threshold`; update `prev_mean`; return drift flag
    - _Requirements: 8.1_

  - [x] 10.2 Integrate `DriftDetector` and adaptive LR into `run_online_simulation`
    - Instantiate `DriftDetector()` and `cycles_since_drift = 0` before the cycle loop
    - After each cycle: compute `cycle_pass_rate`, call `detector.update(cycle_pass_rate)`
    - On drift: `lr = min(lr * 2.0, 0.01)`, reset `cycles_since_drift = 0`, log event to `GlobalState.drift_history`
    - On no drift and `cycles_since_drift >= 5`: `lr = max(lr * 0.9, 0.001)`
    - Apply updated `lr` to optimizer param groups each cycle
    - Add `drift_history: List[Dict]` to `GlobalState`; each entry: `{cycle_id, old_lr, new_lr, rolling_mean_before, rolling_mean_after}`
    - _Requirements: 8.2, 8.3, 8.4_

  - [x] 10.3 Add `GET /api/model/drift-history` endpoint
    - Return `GlobalState.drift_history` as a JSON list
    - _Requirements: 8.5_

  - [ ]* 10.4 Write property test for drift detection threshold (Property 19)
    - **Property 19: Drift detection threshold property**
    - **Validates: Requirements 8.1**
    - Feed sequences where rolling mean delta exceeds 0.05; assert `update` returns `True`

  - [ ]* 10.5 Write property test for LR bounds after drift (Property 20)
    - **Property 20: Learning rate bounds after drift**
    - **Validates: Requirements 8.2**
    - Use `st.floats(min_value=1e-5, max_value=0.01)`; assert new LR equals `min(lr * 2.0, 0.01)`

  - [ ]* 10.6 Write property test for LR decay after stability (Property 21)
    - **Property 21: Learning rate decay after stability**
    - **Validates: Requirements 8.3**
    - Use `st.floats(min_value=0.001, max_value=0.01)`; assert LR after 5 stable cycles equals `max(lr * 0.9, 0.001)`

- [x] 11. SHAP explainability integration
  - [x] 11.1 Add `ShapWrapper` and SHAP computation to `Backend/main.py`
    - Implement `ShapWrapper(nn.Module)` that accepts only the stats tensor and calls `agent(stats, background_sem)`
      where `background_sem` is a fixed background semantic tensor stored as a class attribute
    - After each online simulation cycle, compute SHAP values using `shap.DeepExplainer` (fallback to `shap.GradientExplainer`)
    - Guard: if `len(batch) < 10`, skip SHAP computation for that cycle (do not raise; the endpoint handles the error)
    - Cache results in `GlobalState.shap_cache: Dict[int, Dict]` keyed by `cycle_id`; only recompute when new data is available
    - Store top-5 features by mean absolute SHAP value per cycle
    - Add `shap_cache: Dict[int, Dict]` to `GlobalState`
    - _Requirements: 9.1, 9.4, 9.5_

  - [x] 11.2 Add `GET /api/ai/shap-summary` endpoint
    - Return `{cycle_id, top_features: [{name, mean_abs_shap}]}` for the most recent cycle
    - Return HTTP 422 if the most recent batch had fewer than 10 samples
    - Return HTTP 503 if agent is not yet trained
    - _Requirements: 9.3, 9.4_

  - [ ]* 11.3 Write property test for SHAP additivity (Property 22)
    - **Property 22: SHAP additivity**
    - **Validates: Requirements 9.2**
    - For random batches of size ≥ 10, assert `sum(shap_values[i]) ≈ model(x_i) - E[model(background)]` within 1e-3

  - [ ]* 11.4 Write property test for SHAP cache idempotence (Property 23)
    - **Property 23: SHAP cache idempotence**
    - **Validates: Requirements 9.5**
    - For a fixed `cycle_id` with cached values, call SHAP computation again without new data; assert result is unchanged

- [x] 12. Frontend enhancements
  - [x] 12.1 Fix `render_sidebar_filters` signature in `Frontend/components.py`
    - Replace `session_state.filters.get(...)` calls with explicit default parameters:
      `def render_sidebar_filters(api_client, default_start_date=None, default_end_date=None, default_drift_phases=None)`
    - Update the five drift phase checkboxes to use `["Backend", "Frontend", "Database", "API", "Mobile"]`
    - _Requirements: 10.4, 10.6_

  - [x] 12.2 Add new `APIClient` methods in `Frontend/api_client.py`
    - `get_baselines_apfd() -> Dict`: GET `/api/baselines/apfd`
    - `get_checkpoint_status() -> Dict`: GET `/api/model/checkpoint-status`
    - `get_drift_history() -> List`: GET `/api/model/drift-history`
    - `get_attention_weights() -> Dict`: GET `/api/ai/attention-weights`
    - `get_shap_summary() -> Dict`: GET `/api/ai/shap-summary`
    - _Requirements: 10.7, 10.8_

  - [x] 12.3 Add "📊 Baselines" tab (tab 6) to `Frontend/app_enhanced.py`
    - Call `api_client.get_baselines_apfd()` and render a Plotly line chart with four series:
      Random, History-Based, Coverage-Based, Semanti-Q over cycles 20–99
    - Handle `None` response with `st.error` and retry button
    - _Requirements: 10.7_

  - [x] 12.4 Add "🔍 Explainability" tab (tab 7) to `Frontend/app_enhanced.py`
    - Sub-section 1: SHAP bar chart — top-5 features by mean |SHAP| from `get_shap_summary()`
    - Sub-section 2: Attention weight heatmap — per-sample gate values from `get_attention_weights()`,
      rendered as a Plotly heatmap
    - Handle `None` responses with `st.error` and retry button
    - _Requirements: 10.8_

  - [ ]* 12.5 Write property test for filter parameter propagation (Property 24)
    - **Property 24: Filter parameter propagation**
    - **Validates: Requirements 10.5**
    - For random subsets of the 5 drift phases, assert the API call's `drift_phase` query param exactly matches the selected phases

- [ ] 13. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- All property tests use `hypothesis` with `@settings(max_examples=100)`
- Each property test must include the comment `# Feature: semanti-q-improvements, Property N: <text>`
- Checkpoints at tasks 5, 9, and 13 ensure incremental validation before moving to the next phase
- The `Backend/checkpoints/` directory is created at runtime; no manual setup needed
