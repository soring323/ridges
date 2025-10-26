# Changes to the Polyglot dataset

The Ridges team has made some changes to the original Polyglot problem set, which was sourced originally from [this repository](https://github.com/Aider-AI/polyglot-benchmark/tree/main/python/exercises/practice). The reason for these changes were because some of the questions are believed (by us) to be practically unsolvable due to errors in the problem statement or tests that do not correlate exactly with the instructions. All changes made to the problem set are documented here for transparency.

<br>
<br>

## affine-cipher

Added typing to `encode()` and `decode()`, as it is unclear what types the parameters should be and what the functions should return. There are some links that are impossible for the agent to follow. This will be resolved in a future version of our sandbox, where we provide restricted Internet access.

## beer-song

Added typing to `recite()`, as it is unclear whether the function should return a list of strings, or a single string, seperated by newlines.

## book-store

Added typing to `total()`, as it is unclear what format of input the function should expect. Changed the return type to `int` and added a comment (`# in cents`) to indicate the format of the output, as the instructions do not specify whether the solution should return an answer as a decimal number of dollars, or an integer number of cents. In fact, the instructions lean towards decimal dollars, whereas the tests use integer cents.

## bottle-song

Added typing to `recite()`, as it is unclear whether the function should return a list of strings, or a single string, seperated by newlines.

## bowling

No changes. The instructions provide enough typing information, as such, it is not necessary to include it in the Python file.

## connect

Added typing to `__init__()` and `get_winner()`, as it is unclear what type the board representation should be and what format the winner should be returned in. There are some links that are impossible for the agent to follow. This will be resolved in a future version of our sandbox, where we provide restricted Internet access.

## dominoes

Added typing to `can_chain()`.

## dot-dsl

Added typing to `Node.__init__()`, `Edge.__init__()`, and `Graph.__init__()`, as it is unclear what types the parameters should be. Fixed and expanded the error message documentation in instructions to include all 6 error messages required by tests: "Graph data malformed", "Graph item incomplete", "Attribute is malformed", "Node is malformed", "Edge is malformed", and "Unknown item". The original instructions only documented 2 of these. There are some links that are impossible for the agent to follow. This will be resolved in a future version of our sandbox, where we provide restricted Internet access.

## food-chain

Added typing to `recite()`, as it is unclear what the parameter types should be and what the function should return. There are some links that are impossible for the agent to follow. This will be resolved in a future version of our sandbox, where we provide restricted Internet access.

## forth

Added typing to `evaluate()`, as it is unclear what the parameter type should be and what the function should return. There are some links that are impossible for the agent to follow. This will be resolved in a future version of our sandbox, where we provide restricted Internet access.

## go-counting

Added typing to `Board.__init__()`, `territory()`, and `territories()`. Moved the `WHITE`, `BLACK`, and `NONE` constants from main.py to tests.py, as the docstrings already specify the exact string values to return and requiring the agent to define these constants is an unnecessary hurdle. There are some links that are impossible for the agent to follow. This will be resolved in a future version of our sandbox, where we provide restricted Internet access.

## grade-school

Added typing to `add_student()`, `roster()`, `grade()`, and `added()`, as it is unclear what the parameter types should be and what the methods should return. The `added()` method is particularly ambiguous as the instructions mention tracking duplicate additions but don't specify how. There are some links that are impossible for the agent to follow. This will be resolved in a future version of our sandbox, where we provide restricted Internet access.

## grep

Added typing to `grep()`, as it is unclear what the parameter types should be and what the function should return. The `flags` parameter format is particularly ambiguous as an agent might assume it's a list rather than a space-separated string. There are some links that are impossible for the agent to follow. This will be resolved in a future version of our sandbox, where we provide restricted Internet access.

## hangman

Added typing to `Hangman.__init__()`, `guess()`, `get_masked_word()`, and `get_status()`, as it is unclear what the parameter types should be and what the methods should return. There are some links that are impossible for the agent to follow. This will be resolved in a future version of our sandbox, where we provide restricted Internet access.

## list-ops

Added typing to all functions (`append()`, `concat()`, `filter()`, `length()`, `map()`, `foldl()`, `foldr()`, `reverse()`), as it is unclear what the parameter types should be and what the functions should return. The fold functions are particularly ambiguous regarding parameter order and types.

## phone-number

Added typing to `PhoneNumber.__init__()` and `pretty()`. Added the missing `pretty()` method stub, as it is required by the tests but was not present in the main.py file.

## pig-latin

Added typing to `translate()`, as it is unclear what the parameter type should be and what the function should return.

## poker

Added typing to `best_hands()` with comments clarifying the hand format and poker hand rankings. The instructions only link to Wikipedia for poker rules, which agents cannot access, making this problem unsolvable without explicitly listing the hand rankings. There are some links that are impossible for the agent to follow. This will be resolved in a future version of our sandbox, where we provide restricted Internet access.

## pov

Added typing to `Tree.__init__()`, `from_pov()`, and `path_to()`, as it is unclear what the parameter types should be and what the methods should return. There are some links that are impossible for the agent to follow. This will be resolved in a future version of our sandbox, where we provide restricted Internet access.

## proverb

Added the complete function signature with typing for `proverb()`, including the `*items` variadic parameter and `qualifier` keyword argument, as the main.py file had no parameters at all. The instructions mention the `qualifier` parameter but don't specify its type or default value.

## react

Added typing to `InputCell.__init__()`, `ComputeCell.__init__()`, `add_callback()`, and `remove_callback()`, as it is unclear what the parameter types should be. The reactive programming paradigm described in the instructions is complex, and type hints help clarify the expected interface.

## rest-api

Added typing to `RestAPI.__init__()`, `get()`, and `post()`, as it is unclear what the parameter types should be and what the methods should return. The payloads are JSON strings, which is not obvious from the instructions alone. There are some links that are impossible for the agent to follow. This will be resolved in a future version of our sandbox, where we provide restricted Internet access.

## robot-name

Added the missing `name` property and `reset()` method with type hints. The instructions mention these behaviors but main.py had no indication that a `name` property or `reset()` method were needed.

## scale-generator

Added typing to `Scale.__init__()`, `chromatic()`, and `interval()`, as it is unclear what the parameter types should be and what the methods should return.

## sgf-parsing

Added typing to `SgfTree.__init__()` and `parse()`, as it is unclear what the parameter types should be and what the function should return. There are some links that are impossible for the agent to follow. This will be resolved in a future version of our sandbox, where we provide restricted Internet access.

## simple-linked-list

Added typing to `Node.__init__()`, `Node.value()`, `Node.next()`, `LinkedList.__init__()`, `LinkedList.__len__()`, `LinkedList.head()`, `LinkedList.push()`, `LinkedList.pop()`, and `LinkedList.reversed()`, as it is unclear what the parameter types should be and what the methods should return. There are some links that are impossible for the agent to follow. This will be resolved in a future version of our sandbox, where we provide restricted Internet access.

## transpose

Added typing to `transpose()`, as it is unclear what the parameter type should be and what the function should return. There are some links that are impossible for the agent to follow. This will be resolved in a future version of our sandbox, where we provide restricted Internet access.

## tree-building

Added typing to `Record.__init__()`, `Node.__init__()`, and `BuildTree()`. Fixed the error messages in the provided implementation to match what the tests expect, as the original code had generic messages like "error!" and "something went wrong!" instead of the specific messages required by the tests. Added comments documenting the required error messages. There are some links that are impossible for the agent to follow. This will be resolved in a future version of our sandbox, where we provide restricted Internet access.

## two-bucket

Added typing to `measure()` with a comment clarifying the return tuple format. The instructions explain what three values should be determined but don't specify they should be returned as a tuple, the order of the values, or that the bucket should be the string "one" or "two". There are some links that are impossible for the agent to follow. This will be resolved in a future version of our sandbox, where we provide restricted Internet access.

## variable-length-quantity

Added typing to `encode()` and `decode()`, as it is unclear what the parameter types should be and what the functions should return. There are some links that are impossible for the agent to follow. This will be resolved in a future version of our sandbox, where we provide restricted Internet access.

## wordy

Added typing to `answer()`, as it is unclear what the parameter type should be and what the function should return. There are some links that are impossible for the agent to follow. This will be resolved in a future version of our sandbox, where we provide restricted Internet access.

## zebra-puzzle

Added typing to `drinks_water()` and `owns_zebra()`, as it is unclear what the functions should return (a nationality string).

## zipper

Added typing to all `Zipper` methods and fixed missing parameters for `set_value()`, `set_left()`, and `set_right()`, as the main.py file had these methods with no parameters but the tests call them with arguments. Added a comment clarifying the tree dict structure (with keys "value", "left", "right"), as the instructions never explicitly state this format. There are some links that are impossible for the agent to follow. This will be resolved in a future version of our sandbox, where we provide restricted Internet access.