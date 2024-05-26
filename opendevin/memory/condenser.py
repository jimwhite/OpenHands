from opendevin.core.logger import opendevin_logger as logger
from opendevin.events.action.action import Action
from opendevin.events.action.agent import AgentRecallAction, AgentSummarizeAction
from opendevin.events.action.browse import BrowseInteractiveAction, BrowseURLAction
from opendevin.events.action.commands import (
    CmdKillAction,
    CmdRunAction,
    IPythonRunCellAction,
)
from opendevin.events.action.empty import NullAction
from opendevin.events.action.files import FileReadAction, FileWriteAction
from opendevin.events.action.message import MessageAction
from opendevin.events.event import Event, EventSource
from opendevin.llm.llm import LLM
from opendevin.memory.history import ShortTermHistory
from opendevin.memory.prompts import parse_summary_response

MAX_TOKEN_COUNT_PADDING = (
    512  # estimation of tokens to add to the prompt for the max token count
)


class MemoryCondenser:
    """
    Condenses the prompt with a call to the LLM.
    """

    def __init__(
        self,
        llm: LLM,
    ):
        """
        Initialize the MemoryCondenser.

        llm is the language model to use for summarization.
        config.max_input_tokens is an optional configuration setting specifying the maximum context limit for the LLM.
        If not provided, the condenser will act lazily and only condense when a context window limit error occurs.

        Parameters:
        - llm: The language model to use for summarization.
        """
        self.llm = llm

    def condense(
        self,
        events: ShortTermHistory,
    ) -> AgentSummarizeAction | None:
        """
        Condenses the given list of events using the llm. Returns the condensed list of events.

        Condensation heuristics:
        - Keep initial messages (system, user message setting task)
        - Prioritize more recent history
        - Lazily summarize between initial instruction and most recent, starting with earliest condensable turns
        - Introduce a SummaryObservation event type for textual summaries
        - Split events into chunks delimited by user message actions (messages with EventSource.USER), condense each chunk into a sentence

        Parameters:
        - events: List of events to condense.

        Returns:
        - The condensed list of events.
        """
        # chunk of actions, observations to summarize
        chunk: list[Event] = []
        chunk_start_index = 0

        for i, event in enumerate(events):
            # user messages should be kept if possible
            # FIXME what to do about NullAction?
            if (
                isinstance(event, Action)
                and event.source == EventSource.USER
                or isinstance(event, NullAction)
            ):
                if chunk:
                    summary_action = self._summarize_chunk(chunk)
                    summary_action._chunk_start = chunk_start_index
                    summary_action._chunk_end = i - 1
                    return summary_action
                else:
                    chunk_start_index = i + 1
            elif isinstance(event, self._summarizable_actions()):
                chunk.append(event)
            else:
                chunk_start_index = i + 1

        if chunk:
            summary_action = self._summarize_chunk(chunk)
            summary_action._chunk_start = chunk_start_index
            summary_action._chunk_end = len(events) - 1
            return summary_action
        else:
            return None

    def _summarize_chunk(self, chunk: list[Event]) -> AgentSummarizeAction:
        """
        Summarizes the given chunk of events into a single sentence.

        Parameters:
        - chunk: List of events to summarize.

        Returns:
        - The summary sentence.
        """
        try:
            prompt = f"""
            Given the following actions and observations, create a JSON response with:
                - "action": "summarize"
                - args:
                  - "summarized_actions": A comma-separated list of all the action names from the provided actions
                  - "summary": A single sentence summarizing all the provided observations

                {chunk}
            """
            messages = [{'role': 'user', 'content': prompt}]
            response = self.llm.completion(messages=messages)
            action_response = response['choices'][0]['message']['content']
            action = parse_summary_response(action_response)
            return action
        except Exception as e:
            logger.error(f'Failed to summarize chunk: {e}')
            raise

    def is_over_token_limit(self, messages: list[dict]) -> int:
        """
        Estimates the token count of the given events using litellm tokenizer.

        Parameters:
        - events: List of messages to estimate the token count for.

        Returns:
        - Estimated token count.
        """

        token_count = self.llm.get_token_count(messages) + MAX_TOKEN_COUNT_PADDING
        return token_count >= self.llm.max_input_tokens

    def _summarizable_actions(self):
        """
        Returns the list of actions that can be summarized.
        """
        actions = (
            NullAction,
            CmdKillAction,
            CmdRunAction,
            IPythonRunCellAction,
            BrowseURLAction,
            BrowseInteractiveAction,
            FileReadAction,
            FileWriteAction,
            AgentRecallAction,
            # AgentFinishAction,
            # AgentRejectAction,
            # AgentDelegateAction,
            # AddTaskAction,
            # ModifyTaskAction,
            # ChangeAgentStateAction,
            MessageAction,
            # AgentSummarizeAction, # this is actually fine but separate
        )
        return actions
