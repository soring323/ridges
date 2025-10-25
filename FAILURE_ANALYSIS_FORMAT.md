# Enhanced Failure Analysis Format

## Overview
The `analyze_failure()` method now produces a structured, LLM-friendly report that guides the AI on how to interpret and use local state information to fix failing tests.

## Example Output Format

```
================================================================================
### FAILED TEST: result/benchmark_react/repo/tests.py::ReactTest::test_callback_cells_only_fire_on_change

**Error Type**: IndexError
**Error Message**: list index out of range

💡 **What This Means**: Code tried to access an index that doesn't exist in a list/array.
   ACTION: Check list lengths and index values in the variables below.

### 📊 SOURCE CODE STATE (Your Implementation)
These are the actual runtime values in your code when the error occurred:

**Simple Values:**
  • initial_value = 1
  • val = 111

**Collections (lists, dicts, etc):**
  • cb1_observer = []
  • inputs = [<InputCell(value=1)>]

**Objects (with their internal state):**
  • input = <InputCell(value=1)>
  • dependent = <ComputeCell(value=111, inputs=[<InputCell>], compute_fn=<function>)>
  • callback1 = <partial(callback, args=(output,), kwargs={'cb1_observer': []})>

### 🎯 TEST CONTEXT (What the test was checking)
These variables show what the test set up and what it expected:

  • input = <InputCell(value=1)>
  • output = <ComputeCell(value=111)>
  • cb1_observer = []
  • callback1 = <partial(callback, args=(output,))>

### 🔄 EXECUTION TRACE (51 events)
This shows the sequence of function calls in your code:

**Function Call Sequence:**
  __init__ → add_callback → value → compute → notify_callbacks → __call__

**Key Return Values:**
  • value() returned with: {'value': 1}
  • compute() returned with: {'result': 111}
  • notify_callbacks() returned with: locals hidden

### ✅ HOW TO FIX THIS
1. **Understand the error**: Review the error type and message above
2. **Compare states**: Look at SOURCE CODE STATE vs TEST CONTEXT
3. **Find the mismatch**: Identify which variable has the wrong value
4. **Trace the logic**: Use the execution trace to find where the wrong value came from
5. **Fix the code**: Update your implementation to produce the correct values

================================================================================
```

## Key Improvements

### 1. Error Context Guidance
- Provides human-readable explanation of error type
- Suggests specific actions based on error (AssertionError, IndexError, etc.)
- Helps LLM understand what to look for

### 2. Structured Variable Grouping
- **Simple Values**: Primitives (int, str, bool) for quick scanning
- **Collections**: Lists, dicts, sets with their contents
- **Objects**: Custom objects with internal state revealed

### 3. Source vs Test Comparison
- Clearly separates "what your code did" from "what test expected"
- Makes it easy to spot discrepancies
- Shows actual runtime values, not just types

### 4. Execution Flow Visualization
- Function call sequence shows code path taken
- Return values reveal intermediate computation results
- Helps identify where wrong values originated

### 5. Step-by-Step Fixing Guide
- Clear action items for the LLM
- Guides systematic debugging approach
- Links all sections together for holistic understanding

## Benefits for LLM

1. **Reduced Cognitive Load**: Information is pre-organized and categorized
2. **Clear Action Items**: Explicit instructions on what to do with the data
3. **Contextual Hints**: Error-specific guidance focuses attention on relevant details
4. **Comparative Analysis**: Side-by-side source vs test state reveals mismatches
5. **Execution Insight**: Call sequence helps understand control flow

## Technical Details

### Enhanced Serialization
- Captures actual values: `val = 111` not `<int>`
- Shows object attributes: `<InputCell(value=1)>` not `<InputCell>`
- Handles collections: `[1, 2, 3]` not `<list>`
- Partial functions: `<partial(func, args=(...))>` not `<partial>`

### Thread Safety
- Uses `threading.settrace()` in addition to `sys.settrace()`
- Thread-safe locking on shared state
- Captures execution in all threads, not just main

### Smart Filtering
- Skips private variables (starting with `_` or `__`)
- Truncates long values to prevent context overflow
- Shows first N items of large collections
- Prioritizes most relevant information
