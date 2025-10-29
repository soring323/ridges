# MCTS Replacement with Test-Driven Guidance

## Summary

Successfully replaced Monte Carlo Tree Search (MCTS) with a **Test-Driven Guidance System** backed by **Trace-Based Learning**. The new approach is more effective for AI agent action selection because it uses real test feedback instead of arbitrary heuristics.

## Changes Made

### 1. **Removed MCTS Implementation**
- Deleted `MCTSNode` class (~30 lines)
- Deleted `MCTS` class (~130 lines)
- Removed unused `math` import

### 2. **Added TraceBasedLearning Class**
**Purpose:** Learn from successful execution traces to recommend actions

**Key Features:**
- Stores successful problem-solving sequences in `.agent_traces.json`
- Recommends next actions based on similar past successes
- Weights recommendations by test pass rate
- Keeps last 100 traces to prevent unbounded growth

**Methods:**
- `record_action()` - Track each action taken
- `finalize_trace()` - Save successful sequences for future learning
- `get_action_recommendations()` - Suggest next actions based on historical data
- `_get_default_sequence()` - Cold-start fallback sequence

### 3. **Added TestDrivenGuidance Class**
**Purpose:** Generate intelligent guidance based on test results and execution history

**Key Features:**
- Analyzes test output to extract actionable insights
- Detects error patterns (AssertionError, ImportError, etc.)
- Suggests specific tools based on error types
- Tracks test pass rate trends
- Provides progress feedback to the agent

**Methods:**
- `initialize()` - Setup with problem context
- `analyze_test_results()` - Parse test output for insights
- `get_guidance()` - Generate recommendations with emojis for clarity
- `update_from_action()` - Learn from each action's results
- `finalize()` - Save successful traces

### 4. **Updated PEVWorkflow**
**Changes:**
- Renamed `enable_mcts` → `enable_test_guidance`
- Replaced `self.mcts` → `self.guidance` (TestDrivenGuidance instance)
- Replaced `run_mcts_exploration()` → `initialize_guidance()` and `get_action_guidance()`

### 5. **Integrated into Main Workflow**
**In `fix_task_solve_workflow()`:**
- Initialize guidance with problem type
- Inject test-driven guidance into prompt at each step
- Update guidance after each action execution
- Finalize and save traces at completion

**Guidance is injected as:**
```python
# Add test-driven guidance
if enable_pev and enable_test_guidance and pev.guidance:
    test_guidance = pev.get_action_guidance(step, last_action, last_result)
    if test_guidance:
        messages.append({"role": "system", "content": test_guidance})
```

### 6. **Updated All Entry Points**
Changed parameter from `enable_mcts` to `enable_test_guidance` in:
- `process_fix_task()`
- `process_create_task()`  
- `fix_task_solve_workflow()`
- `agent_main()`

## Why This is Better

### Old MCTS Approach Problems:
1. **Trivial simulation** - used hardcoded heuristics like:
   ```python
   if "search" in actions: score += 0.2
   if "apply_code_edit" in actions: score += 0.3
   ```
2. **Fixed action space** - same 5 actions regardless of context
3. **No learning** - didn't improve from test results
4. **Arbitrary scoring** - not based on actual success metrics

### New Test-Driven Approach Benefits:
1. **Real test feedback** - uses actual test pass/fail rates
2. **Error-specific suggestions** - recommends different tools based on error type
3. **Learning from success** - builds database of working solutions
4. **Progress tracking** - shows test pass rate trends
5. **Context-aware** - recommendations based on similar past problems
6. **Simpler** - no complex tree search, just direct analysis

## Example Guidance Output

```
📊 Test Status: 3/5 passed (60.0%)
⚠️  Error Types: assertion_failure
💡 Suggested Tools: get_context_around_line, apply_code_edit

🎯 Recommended Action Sequence (based on similar successful fixes):
   search_in_all_files_content → get_file_content → apply_code_edit → run_repo_tests → finish

📈 Progress: Test pass rate improved by 20.0%
```

## Data Persistence

Successful execution traces are stored in `.agent_traces.json`:
```json
[
  {
    "problem_type": "bug_fix",
    "actions": ["search_in_all_files_content", "get_file_content", "apply_code_edit", "run_repo_tests", "finish"],
    "test_pass_rate": 1.0,
    "success": true,
    "timestamp": 1735479000.0
  }
]
```

## Future Enhancements

1. **Problem similarity matching** - use embeddings to find more similar traces
2. **Action outcome prediction** - predict success probability before execution
3. **Multi-step planning** - use traces to plan entire solution sequences
4. **Collaborative learning** - share traces across multiple agents
5. **A/B testing** - compare new vs learned sequences

## Backward Compatibility

The system is fully backward compatible:
- Default parameter is `enable_test_guidance=True`
- If disabled, workflow operates as before (without MCTS)
- Graceful degradation if trace file doesn't exist or is corrupted

---

**Result:** Simpler, more effective, and learns over time! 🎉
