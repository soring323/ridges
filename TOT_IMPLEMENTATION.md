# Tree of Thought (TOT) Implementation

## Overview

Successfully replaced Chain of Thought (COT) with Tree of Thought (TOT) in `tot_agent.py`. TOT explores multiple reasoning branches instead of following a single linear path, enabling the agent to recover from dead ends and find better solutions.

## Key Differences: COT vs TOT

### Chain of Thought (COT) - Previous Implementation
- **Linear search**: Follows a single path of reasoning
- **No backtracking**: Once stuck, cannot explore alternatives
- **Limited exploration**: Commits to first generated action
- **Failure mode**: Dead ends result in incomplete solutions

### Tree of Thought (TOT) - New Implementation
- **Tree-based search**: Maintains multiple reasoning branches
- **Backtracking**: Can return to earlier states and explore alternatives
- **Branch evaluation**: Scores and compares multiple candidates
- **Recovery**: Automatically backtracks when hitting consecutive errors

## Core Components

### 1. Tree Structure
```python
class EnhancedTOT:
    class Node:
        - action: The action taken at this node
        - parent: Reference to parent node
        - children: List of child nodes (branches)
        - depth: Distance from root
        - is_explored: Whether this branch has been fully explored
        - is_terminal: Whether this is a success/failure end state
```

### 2. Branch Exploration
- **`should_explore_branches()`**: Determines when to generate multiple candidates
  - After error recovery
  - At regular intervals (every 5 steps)
  - Limited by depth to prevent excessive branching

- **`max_branches`**: Controls how many alternative candidates to generate (default: 3)

### 3. Action Evaluation
- **`evaluate_action()`**: Scores actions based on:
  - Success/failure status
  - Observation content (positive/negative signals)
  - Tool execution results
  - Returns score 0.0-1.0 (higher is better)

### 4. Backtracking Mechanism
- **`should_backtrack()`**: Triggers after `backtrack_threshold` consecutive errors (default: 2)
- **`backtrack()`**: Navigation strategy:
  1. First try unexplored sibling nodes
  2. Then traverse up to find ancestor with unexplored children
  3. Select highest-scored unexplored branch

### 5. Path Management
- **`get_current_path()`**: Extracts linear path from root to current node
- **`to_str()`**: Converts current path to message format for LLM context
- **`add_action(as_branch=True/False)`**: Adds actions as children or siblings

## Workflow Changes

### Multiple Candidate Generation
When `should_explore_branches()` returns True:
1. Generate `max_branches` candidates with temperature=0.3
2. Execute and evaluate each candidate
3. Add all as sibling branches in the tree
4. Select and navigate to the best-scored branch

### Single Candidate Mode
When not exploring branches:
1. Generate single candidate with temperature=0.0
2. Execute and add to current path
3. Continue linear progression

### Automatic Recovery
```python
if tot.should_backtrack():
    if tot.backtrack():
        # Continue from alternative path
        continue
    else:
        # No alternatives available
        logger.warning("Backtrack failed")
```

## Usage Example

```python
# Initialize with TOT parameters
tot = EnhancedTOT(
    latest_observations_to_keep=5,  # Context window
    max_branches=3,                  # Candidates per exploration
    backtrack_threshold=2            # Errors before backtracking
)

# The workflow automatically:
# 1. Explores multiple branches at decision points
# 2. Scores each candidate action
# 3. Backtracks when stuck
# 4. Selects optimal paths through the search tree
```

## Benefits

1. **Robustness**: Can recover from errors by exploring alternatives
2. **Better Solutions**: Evaluates multiple approaches before committing
3. **Completeness**: Less likely to end with incomplete solutions
4. **Exploration**: Discovers solutions that linear search would miss

## Configuration

Key parameters in `fix_task_solve_workflow()`:
- `max_branches=3`: Number of alternative candidates to explore
- `backtrack_threshold=2`: Consecutive errors before backtracking
- `latest_observations_to_keep=5`: Recent observations in context

## Debugging

The `export_to_csv()` method now exports the entire tree structure including:
- Node depth
- Branch scores
- Exploration history
- Parent-child relationships

## Future Enhancements

Potential improvements:
1. **Pruning**: Remove low-scoring branches to save memory
2. **Beam search**: Keep top-K branches instead of exploring all
3. **Learned evaluation**: Use ML model to score candidates
4. **Dynamic branching**: Adjust `max_branches` based on problem difficulty
