# Adaptive Strategy Planner Implementation

## Summary

Enhanced the `StrategicPlanner` with outcome tracking and data-driven strategy selection. The planner now learns which strategies actually work for different problem types, replacing arbitrary confidence scores with real performance data.

## New Components

### 1. **StrategyOutcomeTracker Class**
Tracks and learns from strategy execution outcomes.

**Key Features:**
- Stores outcomes in `.strategy_outcomes.json`
- Classifies strategies into types: `conservative`, `comprehensive`, `incremental`, `refactor`, `general`
- Tracks success rate, test pass rate, efficiency (steps taken)
- Keeps last 200 outcomes

**Metrics Tracked:**
```python
{
    "strategy_name": "Conservative Fix",
    "strategy_type": "conservative",
    "problem_type": "bug_fix",
    "success": true,
    "test_pass_rate": 0.95,
    "steps_taken": 25,
    "execution_time": 120.5,
    "complexity": "low",
    "risk": "low",
    "timestamp": 1735479000.0
}
```

**Methods:**
- `record_outcome()` - Save strategy result
- `get_success_rate()` - Historical success rate for specific strategy
- `get_strategy_type_performance()` - Performance metrics by strategy type
- `get_best_strategy_type()` - Find best performing type for problem

### 2. **Enhanced StrategicPlanner**
Now learns from outcomes instead of using arbitrary scoring.

**Key Changes:**

#### Before (Arbitrary Weights):
```python
def select_best_strategy(strategies):
    def score_strategy(s):
        confidence = s.get("confidence", 0.5)
        risk_score = {"low": 1.0, "medium": 0.7, "high": 0.4}.get(s.get("risk"), 0.7)
        complexity_score = {"low": 1.0, "medium": 0.8, "high": 0.6}.get(s.get("complexity"), 0.8)
        return confidence * 0.5 + risk_score * 0.3 + complexity_score * 0.2  # Arbitrary!
```

#### After (Data-Driven):
```python
def select_best_strategy(strategies, problem_type):
    def score_strategy(s):
        type_performance = self.outcome_tracker.get_strategy_type_performance(strategy_type, problem_type)
        
        if type_performance["sample_count"] >= 5:
            # Use real historical data (70% weight)
            historical_score = (
                type_performance["success_rate"] * 0.6 + 
                type_performance["avg_test_pass"] * 0.3 +
                type_performance["avg_efficiency"] * 0.1
            )
            return historical_score * 0.7 + llm_confidence * 0.3
        else:
            # Fallback to LLM confidence with risk/complexity adjustment
            return llm_confidence * 0.5 + risk_score * 0.3 + complexity_score * 0.2
```

**New Features:**

1. **Context-Aware Generation**
   - Injects historical insights into LLM prompt
   - Suggests strategy types that worked before
   
   ```python
   context_hint = f"""
   Historical insight: For '{problem_type}' problems, '{best_strategy_type}' strategies have 
   {perf['success_rate']:.1%} success rate (based on {perf['sample_count']} past cases).
   Consider including a {best_strategy_type} approach in your strategies.
   """
   ```

2. **Smart Fallback Strategies**
   - Uses historically best strategy type
   - Includes conservative option + best performing type
   - Adaptive based on problem type

3. **Outcome Recording**
   - Stores selected strategy for later recording
   - `record_strategy_outcome()` called at workflow end

### 3. **Updated PEVWorkflow Integration**

**New Methods:**
```python
def run_planning_phase(self, problem_statement: str, problem_type: str = "unknown"):
    """Phase 1: Strategic Planning with outcome tracking"""
    strategies = self.planner.generate_strategies(problem_statement, problem_type)
    selected = self.planner.select_best_strategy(strategies["strategies"], problem_type)
    return selected

def record_strategy_outcome(self, problem_type: str, success: bool, 
                           test_pass_rate: float, steps_taken: int, execution_time: float):
    """Record strategy outcome for learning"""
    self.planner.record_strategy_outcome(problem_type, success, test_pass_rate, steps_taken, execution_time)
```

### 4. **Workflow Integration**

At the **start** of `fix_task_solve_workflow()`:
```python
# Determine problem type
problem_type = "bug_fix"  # Can be enhanced with problem_statement analysis

# Run planning with problem type
strategy = pev.run_planning_phase(problem_statement, problem_type)

# Track workflow start time
workflow_start_time = time.time()
```

At the **end** of workflow:
```python
# Calculate success metrics
overall_success = next_tool_name == "finish" and not cot.thoughts[-1].is_error
test_pass_rate = pev.guidance.test_history[-1]['pass_rate'] if pev.guidance and pev.guidance.test_history else 0.0
execution_time = time.time() - workflow_start_time

# Record strategy outcome for learning
pev.record_strategy_outcome(
    problem_type=problem_type,
    success=overall_success,
    test_pass_rate=test_pass_rate,
    steps_taken=step + 1,
    execution_time=execution_time
)
```

## Example Learning Scenario

### Initial State (No Historical Data)
```
Problem: "Fix authentication bug"
Problem Type: "bug_fix"

Generated Strategies:
1. Conservative Fix (confidence: 0.7) → SELECTED (fallback to LLM confidence)
2. Comprehensive Solution (confidence: 0.6)
```

### After 10 Executions
```json
{
  "conservative": {
    "success_rate": 0.8,
    "avg_test_pass": 0.85,
    "sample_count": 6
  },
  "comprehensive": {
    "success_rate": 0.5,
    "avg_test_pass": 0.6,
    "sample_count": 4
  }
}
```

### Next Execution (With Historical Data)
```
Problem: "Fix authentication bug"
Problem Type: "bug_fix"

LLM Prompt now includes:
"Historical insight: For 'bug_fix' problems, 'conservative' strategies have 
80.0% success rate (based on 6 past cases).
Consider including a conservative approach in your strategies."

Strategy Selection:
- Conservative Fix: score = (0.8*0.6 + 0.85*0.3 + 0.2*0.1) * 0.7 + 0.7*0.3 = 0.725 ✓ SELECTED
- Comprehensive: score = (0.5*0.6 + 0.6*0.3 + 0.15*0.1) * 0.7 + 0.6*0.3 = 0.523
```

## Strategy Type Classification

The system automatically classifies strategies:

| Type | Keywords | Use Case |
|------|----------|----------|
| **conservative** | "conservative", "minimal", "targeted" | Quick fixes, low risk |
| **comprehensive** | "comprehensive", "root cause", "thorough" | Complex issues, deep analysis |
| **incremental** | "incremental", "iterative", "step-by-step" | Gradual improvements |
| **refactor** | "refactor", "redesign", "restructure" | Code quality improvements |
| **general** | (catch-all) | Unclassified strategies |

## Performance Metrics

The system tracks:

1. **Success Rate** - Did the strategy complete successfully?
2. **Test Pass Rate** - What % of tests passed?
3. **Efficiency** - How many steps did it take?
4. **Execution Time** - How long did it run?

**Scoring Formula** (when sample_count >= 5):
```python
score = (
    success_rate * 0.6 +      # Primary metric
    avg_test_pass * 0.3 +      # Quality metric
    avg_efficiency * 0.1        # Speed metric
) * 0.7 + llm_confidence * 0.3  # Blend with LLM
```

## Data Persistence

Outcomes stored in `.strategy_outcomes.json`:
```json
[
  {
    "strategy_name": "Conservative Fix",
    "strategy_type": "conservative",
    "problem_type": "bug_fix",
    "success": true,
    "test_pass_rate": 0.95,
    "steps_taken": 25,
    "execution_time": 120.5,
    "complexity": "low",
    "risk": "low",
    "timestamp": 1735479000.0
  }
]
```

## Key Benefits

### Old System (Arbitrary):
- ❌ Fixed weights (0.5, 0.3, 0.2) with no justification
- ❌ No learning from past successes
- ❌ Same strategy preferences regardless of problem type
- ❌ LLM confidence is only signal

### New System (Adaptive):
- ✅ **Data-driven** - uses real success rates
- ✅ **Learns over time** - improves with more data
- ✅ **Context-aware** - different strategies for different problems
- ✅ **Hybrid approach** - combines LLM confidence with historical data
- ✅ **Cold start handling** - graceful fallback when no data exists
- ✅ **Automatic classification** - categorizes strategies for better learning

## Future Enhancements

1. **Problem Type Detection** - Automatically classify problem from statement
2. **Strategy Combination** - Blend multiple successful strategies
3. **Confidence Calibration** - Learn to adjust LLM confidence scores
4. **Time-based Decay** - Weight recent outcomes more heavily
5. **Cross-problem Learning** - Transfer knowledge between similar problem types

## Integration with TraceBasedLearning

Both systems now work together:

| Component | Learns | Output |
|-----------|--------|--------|
| **TraceBasedLearning** | Action sequences | Next action recommendations |
| **StrategyOutcomeTracker** | Strategy effectiveness | Best strategy type |
| **TestDrivenGuidance** | Test patterns | Error-specific tool suggestions |

**Synergy:** Strategic planning picks the high-level approach, TraceBasedLearning recommends the action sequence, and TestDrivenGuidance provides real-time feedback adjustments.

---

## Example Output Evolution

### Before (Static):
```
Strategic Plan: Conservative Fix - Minimal targeted changes
```

### After (Adaptive, with history):
```
Strategic Plan: Conservative Fix - Minimal targeted changes
(Selected based on 80% success rate for bug_fix problems, 6 historical cases)
```

**Result:** The system gets smarter over time, learning which strategies actually work! 🎯
