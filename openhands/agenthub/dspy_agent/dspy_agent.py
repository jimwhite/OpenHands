import os
from collections import deque

from dspy import DSPyModule
from dspy.optimization import optimize_prompt

from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message, TextContent
from openhands.events.action import Action, AgentFinishAction, MessageAction
from openhands.llm.llm import LLM
from openhands.utils.prompt import PromptManager


class DSPyAgent(Agent):
    VERSION = '1.0'

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

    def _get_action_message(self, action: Action) -> Message:
        """Converts an action into a message format that can be sent to the LLM.

        Args:
            action (Action): The action to convert

        Returns:
            Message: The formatted message for the action
        """
        content = [TextContent(text=self._action_to_str(action))]
        return Message(
            role='user' if action.source == 'user' else 'assistant',
            content=content,
        )

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
