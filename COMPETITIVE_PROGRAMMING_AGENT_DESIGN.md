# Competitive Programming Agent Design

## Overview
A generalized agent architecture for solving algorithm problems using multi-step reasoning, inspired by how competitive programmers approach ACM contests.

## Core Principles

### 1. **Problem Understanding First**
- Don't jump to coding
- Extract constraints, patterns, examples
- Classify problem type

### 2. **Algorithm Design Before Implementation**
- Think through approach logically
- Consider multiple solutions
- Choose optimal based on constraints

### 3. **Implement with Intent**
- Every line has a purpose
- Use appropriate data structures
- Handle edge cases upfront

### 4. **Test-Driven Refinement**
- Run tests early and often
- Debug systematically
- Learn from failures

## Multi-Step Reasoning Framework

### Step 1: Problem Classification
**Prompt Strategy:**
```
Given this problem, identify:
1. Problem Category: [Greedy|DP|Graph|String|Math|Data Structure|Other]
2. Key Constraints: [time limit, space limit, input size]
3. Similar Known Problems: [pattern matching]
4. Difficulty Signals: [what makes this hard?]

Output: Classification + Reasoning
```

### Step 2: Algorithm Design
**Prompt Strategy:**
```
Design an algorithm:
1. Brute Force Approach: [baseline solution]
2. Optimized Approach: [better solution]
3. Complexity Analysis: [time/space]
4. Why This Works: [correctness proof]
5. Edge Cases to Handle: [corner cases]

Output: Pseudo-code + Explanation
```

### Step 3: Implementation
**Prompt Strategy:**
```
Implement the designed algorithm:
- Use the pseudo-code from Step 2
- Add concrete data structures
- Handle all edge cases identified
- Write production-ready code (NOT stubs!)

CRITICAL: Every parameter must be stored and used
CRITICAL: Every method must have real logic

Output: Complete, runnable code
```

### Step 4: Validation
**Prompt Strategy:**
```
Test your implementation:
1. Run provided examples
2. Generate edge cases:
   - Empty input
   - Single element
   - Maximum constraints
   - Special values
3. If tests fail: analyze WHY and fix

Output: Test results + fixes if needed
```

## Implementation in Agent

### Enhanced Prompts

```python
PROBLEM_ANALYSIS_PROMPT = """
You are a competitive programmer analyzing an algorithm problem.

Problem Statement:
{problem_statement}

Analyze step-by-step:
1. **Problem Type**: Is this greedy, DP, graph, string manipulation, math, or other?
2. **Input Constraints**: What are the size limits? (affects algorithm choice)
3. **Output Format**: What exactly needs to be returned?
4. **Key Observations**: What patterns or insights can we exploit?
5. **Similar Problems**: Have you seen similar problems? What patterns apply?

Output your analysis as structured JSON:
{
  "problem_type": "...",
  "constraints": {...},
  "key_insights": [...],
  "similar_patterns": [...]
}
"""

ALGORITHM_DESIGN_PROMPT = """
Based on the analysis:
{analysis}

Design an algorithm:

1. **Approach**: Describe your solution strategy in plain English
2. **Data Structures**: What structures do you need? (list, dict, set, priority queue, etc.)
3. **Algorithm Steps**: Write pseudo-code
4. **Complexity**: Time and space complexity
5. **Edge Cases**: What special cases must be handled?

Think like you're explaining to another competitive programmer.

Output:
- Clear algorithm description
- Pseudo-code
- Complexity analysis
"""

IMPLEMENTATION_PROMPT = """
🚨 IMPLEMENT THE ALGORITHM - NO STUBS! 🚨

Algorithm Design:
{algorithm_design}

Code Skeleton:
{code_skeleton}

Implement the complete, working solution:

REQUIREMENTS:
✅ Store ALL parameters received in __init__ and other methods
✅ Implement ALL methods with real logic (NO `pass` statements)
✅ Use the data structures from the algorithm design
✅ Handle ALL edge cases identified
✅ Return correct values (not None unless specified)

Example of CORRECT implementation:
```python
class Solution:
    def __init__(self, data):
        self.data = data  # ✅ Store it!
        self.cache = {}   # ✅ Initialize structures
    
    def solve(self):
        # ✅ Real implementation
        result = []
        for item in self.data:
            result.append(self.process(item))
        return result
```

Output: Complete Python code in main.py
"""
```

### Workflow

```python
def solve_algorithm_problem(problem_statement, code_skeleton):
    # Step 1: Analyze problem
    analysis = llm_call(PROBLEM_ANALYSIS_PROMPT, problem_statement)
    
    # Step 2: Design algorithm
    algorithm = llm_call(ALGORITHM_DESIGN_PROMPT, analysis)
    
    # Step 3: Implement
    code = llm_call(IMPLEMENTATION_PROMPT, algorithm, code_skeleton)
    
    # Step 4: Test and refine
    while not all_tests_pass(code):
        failures = get_test_failures(code)
        code = llm_call(DEBUG_PROMPT, code, failures)
    
    return code
```

## Key Advantages

1. **Generalized**: Works for any algorithm problem type
2. **Systematic**: Follows competitive programmer thinking
3. **Explicit**: Forces LLM to think before coding
4. **Traceable**: Each step has clear reasoning
5. **Debuggable**: Can identify where reasoning failed

## Example Flow

**Problem**: Implement a reactive spreadsheet with cells that auto-update

**Step 1 Analysis**:
```json
{
  "problem_type": "Observer Pattern + Graph Dependencies",
  "key_insights": [
    "Cells form a dependency graph",
    "Changes must propagate in topological order",
    "Need to track dependents for each cell"
  ]
}
```

**Step 2 Design**:
```
Algorithm:
1. Each cell maintains list of dependents
2. When value changes, notify all dependents
3. Dependents recompute their values
4. Use topological sort to avoid cycles

Data Structures:
- InputCell: stores value + list of dependents
- ComputeCell: stores inputs, function, value, callbacks
```

**Step 3 Implementation**:
```python
class InputCell:
    def __init__(self, initial_value):
        self.value = initial_value  # ✅ Store it
        self.dependents = []         # ✅ Track dependents
    
    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, new_value):
        self._value = new_value
        self._notify_dependents()   # ✅ Propagate changes
    
    def _notify_dependents(self):
        for cell in self.dependents:
            cell.update()          # ✅ Real logic
```

## Next Steps

1. Integrate this multi-step reasoning into agent.py
2. Test on various problem types
3. Measure improvement over single-shot generation
4. Iterate based on failure patterns
