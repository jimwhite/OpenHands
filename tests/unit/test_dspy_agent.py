import pytest
from openhands.agenthub.dspy_agent.dspy_agent import DSPyAgent
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.llm.llm import LLM
from openhands.events.action import (
    CmdRunAction,
    IPythonRunCellAction,
    AgentDelegateAction,
    MessageAction,
    AgentFinishAction,
)
from openhands.events.observation import (
    CmdOutputObservation,
    IPythonRunCellObservation,
    FileEditObservation,
    AgentDelegateObservation,
    ErrorObservation,
    UserRejectObservation,
)


@pytest.fixture
def llm():
    return LLM(config={"model": "test-model"})


@pytest.fixture
def config():
    return AgentConfig(function_calling=True)


@pytest.fixture
def agent(llm, config):
    return DSPyAgent(llm, config)


def test_initialization(agent):
    assert isinstance(agent, DSPyAgent)
    assert agent.function_calling_active is True


def test_reset(agent):
    agent.reset()
    assert agent.pending_actions == deque()


def test_step(agent):
    state = State()
    state.history.append(MessageAction(content="Test message"))
    action = agent.step(state)
    assert isinstance(action, Action)


def test_parse_response(agent):
    response = {"choices": [{"message": {"content": "<execute_bash>echo Hello</execute_bash>"}}]}
    action = agent.action_parser.parse(response)
    assert isinstance(action, CmdRunAction)
    assert action.command == "echo Hello"


def test_parse_action(agent):
    action_str = "<execute_bash>echo Hello</execute_bash>"
    action = agent.action_parser.parse_action(action_str)
    assert isinstance(action, CmdRunAction)
    assert action.command == "echo Hello"


def test_action_to_str(agent):
    action = CmdRunAction(command="echo Hello")
    action_str = agent.action_parser.action_to_str(action)
    assert action_str == "<execute_bash>echo Hello</execute_bash>"


def test_get_action_message(agent):
    action = CmdRunAction(command="echo Hello")
    messages = agent.get_action_message(action, {})
    assert len(messages) == 1
    assert messages[0].content[0].text == "<execute_bash>echo Hello</execute_bash>"


def test_get_observation_message(agent):
    obs = CmdOutputObservation(content="Hello", exit_code=0)
    messages = agent.get_observation_message(obs, {})
    assert len(messages) == 1
    assert messages[0].content[0].text == "OBSERVATION:\nHello\n[Command finished with exit code 0]"


def test_get_messages(agent):
    state = State()
    state.history.append(MessageAction(content="Test message"))
    messages = agent._get_messages(state)
    assert len(messages) > 0
    assert messages[0].role == "system"
