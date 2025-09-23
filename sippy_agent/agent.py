"""
Core Re-Act agent implementation for Sippy.
"""

import logging
import re
from typing import List, Optional, Union, Dict, Any, Callable
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import BaseTool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult, AIMessage, HumanMessage

from .config import Config
from .api_models import ChatMessage
from .tools import (
    ExampleTool,
    SippyJobAnalysisTool,
    SippyTestFailureTool,
    SippyProwJobSummaryTool,
    SippyLogAnalyzerTool,
    SippyJiraIncidentTool,
    SippyJiraTicketCreatorTool,
    SippyReleasePayloadTool,
    SippyPayloadDetailsTool,
    SippyReleasesTool,
    JUnitParserTool,
    AggregatedJobAnalyzerTool,
    AggregatedYAMLParserTool
)

logger = logging.getLogger(__name__)


class StreamingThinkingHandler(BaseCallbackHandler):
    """Callback handler to stream thinking process in real-time."""

    def __init__(self, thinking_callback: Optional[Callable[[str, str, str, str], None]] = None):
        """Initialize with optional callback for streaming thoughts."""
        self.thinking_callback = thinking_callback
        self.step_count = 0

    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Called when agent takes an action."""
        if self.thinking_callback:
            self.step_count += 1
            action_name = getattr(action, 'tool', 'Unknown')
            action_input = str(getattr(action, 'tool_input', {}))
            thought = f"Calling tool: `{action_name}` with arguments: `{action_input}`"
            self.thinking_callback(thought, action_name, action_input, "")

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes."""
        if self.thinking_callback:
            # Skip error outputs
            if "Invalid" in output or "Error" in output or "_Exception" in output:
                return
            # Stream the observation
            self.thinking_callback("", "", "", output)

    def _extract_thought_from_log(self, log: str) -> str:
        """Extract the thought portion from the action log."""
        if not log:
            return "Processing..."

        # This method is less relevant for tool-calling agents but kept for compatibility.
        return "Analyzing..."


class TokenCountingHandler(BaseCallbackHandler):
    """Callback handler to count tokens used in LLM calls."""

    def __init__(self):
        """Initialize token counter."""
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.call_count = 0

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM finishes generating."""
        self.call_count += 1

        # Try to extract token usage from response
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            if token_usage:
                self.total_tokens += token_usage.get('total_tokens', 0)
                self.prompt_tokens += token_usage.get('prompt_tokens', 0)
                self.completion_tokens += token_usage.get('completion_tokens', 0)

                logger.info(f"LLM Call {self.call_count}: "
                           f"Prompt: {token_usage.get('prompt_tokens', 0)}, "
                           f"Completion: {token_usage.get('completion_tokens', 0)}, "
                           f"Total: {token_usage.get('total_tokens', 0)}")

        # For Gemini models, try alternative token counting
        elif hasattr(response, 'generations') and response.generations:
            for generation_list in response.generations:
                for generation in generation_list:
                    if hasattr(generation, 'generation_info') and generation.generation_info:
                        usage = generation.generation_info.get('usage_metadata', {})
                        if usage:
                            prompt_tokens = usage.get('prompt_token_count', 0)
                            completion_tokens = usage.get('candidates_token_count', 0)
                            total_tokens = usage.get('total_token_count', prompt_tokens + completion_tokens)

                            self.total_tokens += total_tokens
                            self.prompt_tokens += prompt_tokens
                            self.completion_tokens += completion_tokens

                            logger.info(f"LLM Call {self.call_count} (Gemini): "
                                       f"Prompt: {prompt_tokens}, "
                                       f"Completion: {completion_tokens}, "
                                       f"Total: {total_tokens}")
                            break

    def get_summary(self) -> Dict[str, int]:
        """Get token usage summary."""
        return {
            'total_tokens': self.total_tokens,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'call_count': self.call_count
        }

    def reset(self):
        """Reset token counters."""
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.call_count = 0


class SippyAgent:
    """LangChain Re-Act agent for CI analysis with Sippy."""
    
    def __init__(self, config: Config):
        """Initialize the Sippy agent with configuration."""
        self.config = config
        self.llm = self._create_llm()
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent_executor()
        self.token_counter = TokenCountingHandler()
    
    def _create_llm(self) -> Union[ChatOpenAI, ChatGoogleGenerativeAI]:
        """Create the language model instance."""
        if self.config.verbose:
            logger.info(f"Creating LLM with endpoint: {self.config.llm_endpoint}")
            logger.info(f"Using model: {self.config.model_name}")

        # Use ChatGoogleGenerativeAI for Gemini models
        if self.config.is_gemini_model():
            if not self.config.google_api_key and not self.config.google_credentials_file:
                raise ValueError("Google API key or service account credentials file is required for Gemini models")

            llm_kwargs = {
                "model": self.config.model_name,
                "temperature": self.config.temperature,
            }

            # Use API key if provided, otherwise use service account credentials
            if self.config.google_api_key:
                llm_kwargs["google_api_key"] = self.config.google_api_key
                if self.config.verbose:
                    logger.info(f"Using ChatGoogleGenerativeAI for Gemini model with API key")
            elif self.config.google_credentials_file:
                # Set the environment variable for Google credentials
                import os
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.config.google_credentials_file
                if self.config.verbose:
                    logger.info(f"Using ChatGoogleGenerativeAI for Gemini model with service account: {self.config.google_credentials_file}")

            return ChatGoogleGenerativeAI(**llm_kwargs)

        # Use ChatOpenAI for OpenAI and Ollama endpoints
        else:
            llm_kwargs = {
                "model": self.config.model_name,
                "temperature": self.config.temperature,
                "base_url": self.config.llm_endpoint,
            }

            # Only add API key if it's provided (needed for OpenAI, not for local endpoints)
            if self.config.openai_api_key:
                llm_kwargs["openai_api_key"] = self.config.openai_api_key
            else:
                # For local endpoints like Ollama, use a dummy key
                llm_kwargs["openai_api_key"] = "dummy-key"

            if self.config.verbose:
                logger.info(f"Using ChatOpenAI with base_url: {self.config.llm_endpoint}")

            return ChatOpenAI(**llm_kwargs)
    
    def _create_tools(self) -> List[BaseTool]:
        """Create the list of tools available to the agent."""
        tools = [
            ExampleTool(),
            SippyJobAnalysisTool(),
            SippyTestFailureTool(),
            SippyProwJobSummaryTool(sippy_api_url=self.config.sippy_api_url),
            SippyLogAnalyzerTool(sippy_api_url=self.config.sippy_api_url),
            SippyJiraIncidentTool(
                jira_url=self.config.jira_url,
                jira_username=self.config.jira_username,
                jira_token=self.config.jira_token
            ),
            SippyJiraTicketCreatorTool(),
            SippyReleasePayloadTool(),
            SippyPayloadDetailsTool(),
            SippyReleasesTool(sippy_api_url=self.config.sippy_api_url),
            JUnitParserTool(),
            AggregatedJobAnalyzerTool(sippy_api_url=self.config.sippy_api_url),
            AggregatedYAMLParserTool(),
        ]
        
        if self.config.verbose:
            logger.info(f"Created {len(tools)} tools: {[tool.name for tool in tools]}")
        
        return tools
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the Re-Act agent executor."""
        # Custom prompt template for Sippy CI analysis
        prompt_template = """You are Sippy AI, an expert assistant for analyzing CI job and test failures.

You have access to tools that can help you analyze CI jobs, and test failures.

When users ask about CI issues, use the available tools to gather information and provide detailed analysis. Pay attention to
the user's query and ensure you are answering the direction question they gave you.

When presenting information to users, always use markdown links when URLs are available. NEVER put the
entire markdown link in verbatim ticks -- only put the title in ticks.
- For Prow jobs: Use job names as link text with URLs from tool responses
- For GitHub PRs: Use "PR #123" format with GitHub URLs
- For Jira issues: Use issue keys as link text with Jira URLs
- For repositories: Use repo names as link text with GitHub URLs
- For commits: Use short commit hashes as link text with commit URLs

Example formats:
- Job: [periodic-ci-openshift-release-master-nightly-4.20-e22e-aws-ovn](https://prow.ci.openshift.org/view/...)
- PR: [PR #15155](https://github.com/openshift/console/pull/15155)
- Issue: [CONSOLE-4550](https://issues.redhat.com/browse/CONSOLE-4550)
- Repo: [console](https://github.com/openshift/console)
- Commit: [commit: 89925168](https://github.com/openshift/builder/commit/89925168)

When a tool returns a JSON object or list, do not show the raw JSON to the user. Instead, interpret the data and present a clear, human-readable summary. For example, if the user asks why a payload failed, you should look at the 'results.blockingJobs' in the JSON from the get_payload_details tool, identify the failed jobs, and present them in a summary.

Be proactive. If the user asks 'why' a job or payload failed, you should proactively use other tools to find the root cause. For example, if `get_release_payloads` shows a payload was 'Rejected', and the user asks 'why', you should immediately use `get_payload_details` to investigate further without asking for permission.

CRITICAL RULES:
- You MUST use your tools to answer questions. Do not make up answers.
- ALWAYS use `get_release_payloads` to find the latest payload for a release. Do not invent a payload name.
- ALWAYS use `get_prow_job_summary` to get information about a Prow job.
- If you do not know the answer or the tools do not provide it, simply say that you do not have that information.

PAYLOAD ANALYSIS WORKFLOW:
When a user asks about the 'latest' payload (e.g., "What is the latest 4.20 payload?"), you MUST follow this sequence:
1. Call `get_release_payloads` with the appropriate `release_version`.
2. From the JSON response, identify the most recent payload (usually the first in the list).
3. State the name and phase of that payload to the user.

If the user then asks 'why' it was rejected, you MUST then:
1. Call `get_payload_details` with the payload name you just identified.
2. Summarize the results from the JSON to explain the reason for the rejection (e.g., list the blocking jobs that failed).
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create the tool calling agent
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        # Create the agent executor
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.config.verbose,
            max_iterations=self.config.max_iterations,
            handle_parsing_errors=True,
            max_execution_time=self.config.max_execution_time,
            return_intermediate_steps=True,  # Enable intermediate steps for thinking display
        )
    
    def chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None,
             thinking_callback: Optional[Callable[[str, str, str, str], None]] = None) -> Union[str, Dict[str, Any]]:
        """Process a chat message and return the agent's response.

        Args:
            message: The user's message
            chat_history: Previous conversation context as a list of ChatMessage objects
            thinking_callback: Optional callback for streaming thoughts (thought, action, input, observation)
        """
        try:
            # Reset token counter for this conversation
            self.token_counter.reset()

            # Set up callbacks for streaming thinking and token counting
            callbacks = [self.token_counter]
            if self.config.show_thinking and thinking_callback:
                streaming_handler = StreamingThinkingHandler(thinking_callback)
                callbacks.append(streaming_handler)

            history_messages = []
            if chat_history:
                for msg in chat_history:
                    if msg.role == "user":
                        history_messages.append(HumanMessage(content=msg.content))
                    elif msg.role == "assistant":
                        history_messages.append(AIMessage(content=msg.content))

            result = self.agent_executor.invoke({
                "input": message,
                "chat_history": history_messages
            }, config={"callbacks": callbacks})

            # Get token usage summary
            token_usage = self.token_counter.get_summary()

            # Log token usage
            if token_usage['total_tokens'] > 0:
                logger.info(f"Total token usage for this conversation: {token_usage}")

                # Warn if approaching common limits
                if token_usage['total_tokens'] > 100000:  # 100K tokens
                    logger.warning(f"High token usage detected: {token_usage['total_tokens']} tokens")
                elif token_usage['total_tokens'] > 50000:  # 50K tokens
                    logger.info(f"Moderate token usage: {token_usage['total_tokens']} tokens")

            if self.config.show_thinking:
                # Parse the intermediate steps to extract thinking process
                thinking_steps = self._parse_thinking_steps(result)

                # Debug: Always log when thinking is enabled
                logger.info(f"Thinking enabled - found {len(thinking_steps)} steps")
                logger.info(f"Result keys: {list(result.keys())}")

                response_dict = {
                    "output": result["output"],
                    "thinking_steps": thinking_steps
                }

                # Add token usage if available
                if token_usage['total_tokens'] > 0:
                    response_dict["token_usage"] = token_usage

                return response_dict
            else:
                # Return simple response, but include token usage if verbose
                if self.config.verbose and token_usage['total_tokens'] > 0:
                    return {
                        "output": result["output"],
                        "token_usage": token_usage
                    }
                else:
                    return result["output"]
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_msg = f"I encountered an error while processing your request: {str(e)}"
            if self.config.show_thinking:
                return {
                    "output": error_msg,
                    "thinking_steps": []
                }
            else:
                return error_msg

    def _parse_thinking_steps(self, result: Dict[str, Any]) -> List[Dict[str, str]]:
        """Parse the agent's intermediate steps to extract thinking process."""
        thinking_steps = []

        # Get intermediate steps from the result
        intermediate_steps = result.get("intermediate_steps", [])

        # Always log when thinking is enabled (not just verbose)
        if self.config.show_thinking:
            logger.info(f"Parsing thinking: Found {len(intermediate_steps)} intermediate steps")
            logger.info(f"Available result keys: {list(result.keys())}")

        for i, step in enumerate(intermediate_steps):
            if self.config.verbose:
                logger.info(f"Step {i}: {type(step)} with length {len(step) if hasattr(step, '__len__') else 'N/A'}")

            # Step is a tuple of (AgentAction, observation)
            if len(step) >= 2:
                action, observation = step

                # Handle single or multiple tool calls
                actions = [action] if not isinstance(action, list) else action
                observations = [observation] if not isinstance(observation, list) else observation

                for agent_action, obs in zip(actions, observations):
                    action_name = getattr(agent_action, 'tool', 'Unknown')
                    action_input = getattr(agent_action, 'tool_input', {})
                    thought = f"Called tool: `{action_name}` with arguments: `{action_input}`"

                    # Skip error/exception actions in the final display
                    if action_name in ['_Exception', 'Invalid', 'Error'] or 'Invalid' in str(obs):
                        continue

                    thinking_steps.append({
                        "thought": thought,
                        "action": action_name,
                        "action_input": str(action_input),
                        "observation": str(obs)
                    })
        return thinking_steps

    def _extract_thought_from_log(self, log: str) -> str:
        """Extract the thought portion from the action log."""
        if not log:
            return "Processing..."

        # This method is less relevant for tool-calling agents but kept for compatibility.
        return "Analyzing..."

    def add_tool(self, tool: BaseTool) -> None:
        """Add a new tool to the agent."""
        self.tools.append(tool)
        # Recreate the agent executor with the new tool
        self.agent_executor = self._create_agent_executor()
        
        if self.config.verbose:
            logger.info(f"Added tool: {tool.name}")
    
    def list_tools(self) -> List[str]:
        """Get a list of available tool names."""
        return [tool.name for tool in self.tools]
