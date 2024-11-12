import os
import json
from collections import deque
from itertools import islice

from dspy import DSPyModule
from dspy.optimization import optimize_prompt

from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import ImageContent, Message, TextContent
from openhands.events.action import (
    Action,
    AgentDelegateAction,
    AgentFinishAction,
    CmdRunAction,
    FileEditAction,
    IPythonRunCellAction,
    MessageAction,
)
from openhands.events.observation import (
    AgentDelegateObservation,
    CmdOutputObservation,
    FileEditObservation,
    IPythonRunCellObservation,
    UserRejectObservation,
)
from openhands.events.observation.error import ErrorObservation
from openhands.events.observation.observation import Observation
from openhands.events.serialization.event import truncate_content
from openhands.llm.llm import LLM
from openhands.runtime.plugins import (
    AgentSkillsRequirement,
    JupyterRequirement,
    PluginRequirement,
)
from openhands.utils.prompt import PromptManager


class DSPyAgent(Agent):
    VERSION = '1.0'

    sandbox_plugins: list[PluginRequirement] = [
        AgentSkillsRequirement(),
        JupyterRequirement(),
    ]
    obs_prefix = 'OBSERVATION:\n'

    def __init__(self, llm: LLM, config: AgentConfig) -> None:
        """Initializes a new instance of the DSPyAgent class.

        Parameters:
        - llm (LLM): The llm to be used by this agent
        """
        super().__init__(llm, config)
        self.reset()

        self.prompt_manager = PromptManager(
            microagent_dir=os.path.join(os.path.dirname(__file__), 'micro'),
            prompt_dir=os.path.join(os.path.dirname(__file__), 'prompts', 'default'),
        )

        self.dspy_module = DSPyModule()
        self.pending_actions: deque[Action] = deque()

    def reset(self) -> None:
        """Resets the DSPy Agent."""
        super().reset()

    def step(self, state: State) -> Action:
        """Performs one step using the DSPy Agent.
        This includes gathering info on previous steps and prompting the model to make a command to execute.

        Parameters:
        - state (State): used to get updated info

        Returns:
        - Action: The next action to take based on llm response
        """
        # Continue with pending actions if any
        if self.pending_actions:
            return self.pending_actions.popleft()

        # if we're done, go back
        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == '/exit':
            return AgentFinishAction()

        # prepare what we want to send to the LLM
        messages = self._get_messages(state)
        params: dict = {
            'messages': self.llm.format_messages_for_llm(messages),
        }
        response = self.llm.completion(**params)

        actions = self._response_to_actions(response)
        for action in actions:
            self.pending_actions.append(action)
        return self.pending_actions.popleft()

    def _get_messages(self, state: State) -> list[Message]:
        """Constructs the message history for the LLM conversation.

        This method builds a structured conversation history by processing events from the state
        and formatting them into messages that the LLM can understand.

        Args:
            state (State): The current state object containing conversation history and other metadata

        Returns:
            list[Message]: A list of formatted messages ready for LLM consumption
        """
        messages: list[Message] = [
            Message(
                role='system',
                content=[
                    TextContent(
                        text=self.prompt_manager.get_system_message(),
                        cache_prompt=self.llm.is_caching_prompt_active(),
                    )
                ],
            )
        ]
        example_message = self.prompt_manager.get_example_user_message()
        if example_message:
            messages.append(
                Message(
                    role='user',
                    content=[TextContent(text=example_message)],
                    cache_prompt=self.llm.is_caching_prompt_active(),
                )
            )

        events = list(state.history)
        for event in events:
            if isinstance(event, Action):
                message = self._get_action_message(event)
            else:
                message = self._get_observation_message(event)

            if message:
                if message.role == 'user':
                    self.prompt_manager.enhance_message(message)
                if messages and messages[-1].role == message.role:
                    messages[-1].content.extend(message.content)
                else:
                    messages.append(message)

        if self.llm.is_caching_prompt_active():
            breakpoints_remaining = 3
            for message in reversed(messages):
                if message.role == 'user':
                    if breakpoints_remaining > 0:
                        message.content[-1].cache_prompt = True
                        breakpoints_remaining -= 1
                    else:
                        break

        return messages

    def _get_action_message(
        self,
        action: Action,
        pending_tool_call_action_messages: dict[str, Message],
    ) -> list[Message]:
        """Converts an action into a message format that can be sent to the LLM.

        This method handles different types of actions and formats them appropriately:
        1. For tool-based actions (AgentDelegate, CmdRun, IPythonRunCell, FileEdit) and agent-sourced AgentFinish:
            - In function calling mode: Stores the LLM's response in pending_tool_call_action_messages
            - In non-function calling mode: Creates a message with the action string
        2. For MessageActions: Creates a message with the text content and optional image content

        Args:
            action (Action): The action to convert. Can be one of:
                - CmdRunAction: For executing bash commands
                - IPythonRunCellAction: For running IPython code
                - FileEditAction: For editing files
                - BrowseInteractiveAction: For browsing the web
                - AgentFinishAction: For ending the interaction
                - MessageAction: For sending messages
            pending_tool_call_action_messages (dict[str, Message]): Dictionary mapping response IDs
                to their corresponding messages. Used in function calling mode to track tool calls
                that are waiting for their results.

        Returns:
            list[Message]: A list containing the formatted message(s) for the action.
                May be empty if the action is handled as a tool call in function calling mode.

        Note:
            In function calling mode, tool-based actions are stored in pending_tool_call_action_messages
            rather than being returned immediately. They will be processed later when all corresponding
            tool call results are available.
        """
        # create a regular message from an event
        if isinstance(
            action,
            (
                AgentDelegateAction,
                CmdRunAction,
                IPythonRunCellAction,
                FileEditAction,
            ),
        ) or (isinstance(action, AgentFinishAction) and action.source == 'agent'):
            content = [TextContent(text=self._action_to_str(action))]
            return [
                Message(
                    role='user' if action.source == 'user' else 'assistant',
                    content=content,
                )
            ]
        elif isinstance(action, MessageAction):
            role = 'user' if action.source == 'user' else 'assistant'
            content = [TextContent(text=action.content or '')]
            if self.llm.vision_is_active() and action.images_urls:
                content.append(ImageContent(image_urls=action.images_urls))
            return [
                Message(
                    role=role,
                    content=content,
                )
            ]
        return []

    def _get_observation_message(self, obs: Observation) -> Message:
        """Converts an observation into a message format that can be sent to the LLM.

        Args:
            obs (Observation): The observation to convert

        Returns:
            Message: The formatted message for the observation
        """
        text = 'OBSERVATION:\n' + obs.content
        return Message(role='user', content=[TextContent(text=text)])

    def _action_to_str(self, action: Action) -> str:
        """Converts an action to its string representation.

        Args:
            action (Action): The action to convert

        Returns:
            str: The string representation of the action
        """
        if isinstance(action, MessageAction):
            return action.content
        return ''

    def _response_to_actions(self, response: str) -> list[Action]:
        """Converts the LLM response to a list of actions.

        Args:
            response (str): The response from the LLM

        Returns:
            list[Action]: A list of actions derived from the response
        """
        actions = []
        response_content = response.choices[0].message.content
        optimized_prompt = optimize_prompt(response_content, self.dspy_module)
        actions.append(MessageAction(content=optimized_prompt))
        return actions
