# DSPy Agent Framework

This folder implements the DSPyAgent which uses DSPy program optimization for improving prompts.

## Purpose

The DSPyAgent is designed to improve itself based on benchmarks. This enables it to adapt optimally when using different LLMs, as the best prompts for each use case tend to be LLM-dependent.

## Usage

To use the DSPyAgent, you need to import it and register it with the Agent class. Here is an example:

```python
from openhands.agenthub.dspy_agent.dspy_agent import DSPyAgent
from openhands.controller.agent import Agent

Agent.register('DSPyAgent', DSPyAgent)
```

## Example

Here is an example of how to use the DSPyAgent in the code:

```python
from openhands.agenthub.dspy_agent.dspy_agent import DSPyAgent
from openhands.controller.agent_controller import AgentController
from openhands.core.config import AgentConfig
from openhands.llm.llm import LLM

# Initialize the LLM and AgentConfig
llm = LLM()
config = AgentConfig()

# Create an instance of DSPyAgent
dspy_agent = DSPyAgent(llm, config)

# Create an instance of AgentController and register the agent
controller = AgentController()
controller.register_agent(dspy_agent)

# Use the agent
state = controller.initialize_state()
action = dspy_agent.step(state)
print(action)
```
