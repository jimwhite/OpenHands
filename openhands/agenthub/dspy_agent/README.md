# DSPy Agent Framework

This folder implements the DSPy idea that consolidates LLM agents’ actions into a unified code action space for both simplicity and performance.

## Overview

The DSPy Agent is a coding agent that uses DSPy program optimization for improving prompts. This agent improves itself based on benchmarks, enabling it to adapt optimally when using different LLMs.

## Key Features

1. **Program Optimization**: Utilizes DSPy modules for program optimization to improve prompts.
2. **Adaptability**: Adapts optimally when using different LLMs, as the best prompts for each use case tend to be LLM-dependent.
3. **Self-Improvement**: Improves itself based on benchmarks to achieve better agent performance.

## Implementation

The DSPy Agent is implemented using DSPy modules for program optimization. The key design aspect of the agent is using DSPy Modules.

## Usage

To use the DSPy Agent, follow these steps:

1. **Initialization**: Initialize the DSPy Agent with the required LLM and configuration.
2. **Reset**: Reset the agent to its initial state.
3. **Step**: Perform one step using the DSPy Agent, which includes gathering info on previous steps and prompting the model to make a command to execute.

## Example

Here is an example of how to initialize and use the DSPy Agent:

```python
from openhands.agenthub.dspy_agent import DSPyAgent
from openhands.controller.agent import Agent
from openhands.core.config import AgentConfig
from openhands.llm.llm import LLM

# Initialize the LLM and configuration
llm = LLM(model="your_model_name")
config = AgentConfig()

# Initialize the DSPy Agent
agent = DSPyAgent(llm, config)

# Reset the agent
agent.reset()

# Perform one step
state = ...  # Obtain the current state
action = agent.step(state)
```

## References

- [DSPy Modules](https://dspy.ai/building-blocks/3-modules/)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
- [Example of a Coding Agent using DSPy](https://github.com/stanfordnlp/dspy/tree/main/examples/coding)
