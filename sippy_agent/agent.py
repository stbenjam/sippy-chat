"""
Core Re-Act agent implementation for Sippy.
"""

import logging
from typing import List, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool

from .config import Config
from .tools.base import ExampleTool, SippyJobAnalysisTool, SippyTestFailureTool, SippyProwJobSummaryTool, SippyLogAnalyzerTool, SippyJiraIncidentTool

logger = logging.getLogger(__name__)


class SippyAgent:
    """LangChain Re-Act agent for CI analysis with Sippy."""
    
    def __init__(self, config: Config):
        """Initialize the Sippy agent with configuration."""
        self.config = config
        self.llm = self._create_llm()
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent_executor()
    
    def _create_llm(self) -> ChatOpenAI:
        """Create the language model instance."""
        # Prepare kwargs for ChatOpenAI
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
            logger.info(f"Creating LLM with endpoint: {self.config.llm_endpoint}")
            logger.info(f"Using model: {self.config.model_name}")

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
        ]
        
        if self.config.verbose:
            logger.info(f"Created {len(tools)} tools: {[tool.name for tool in tools]}")
        
        return tools
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the Re-Act agent executor."""
        # Custom prompt template for Sippy CI analysis
        prompt_template = """You are Sippy AI, an expert assistant for analyzing CI/CD pipelines, test failures, and build issues.

You have access to tools that can help you analyze CI jobs, test failures, and provide insights about build problems.

When users ask about CI issues, use the available tools to gather information and provide detailed analysis.

ANALYSIS WORKFLOW:
-----------------
When analyzing a job failure, follow this recommended workflow:
1. First, use get_prow_job_summary with just the numeric job ID (e.g., 1934795512955801600)
2. Then, use analyze_job_logs with the same numeric job ID to get detailed error context
3. Use check_known_incidents to see if the failure matches any known open TRT incidents
4. If needed, use analyze_job_logs with different path_glob patterns for additional insights

CORRELATING WITH KNOWN ISSUES:
-----------------------------
After identifying error patterns (like registry issues, timeouts, etc.), use check_known_incidents with relevant search terms to see if this is a known problem. For example:
- Registry errors: search for "registry", "build11", "503"
- Timeout issues: search for "timeout", "hang"
- Infrastructure: search for "infrastructure", "node"

IMPORTANT: Always pass ONLY the numeric job ID to tools, never include extra text or descriptions.

TOOLS:
------
You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""

        prompt = PromptTemplate.from_template(prompt_template)
        
        # Create the Re-Act agent
        agent = create_react_agent(
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
        )
    
    def chat(self, message: str, chat_history: Optional[str] = None) -> str:
        """Process a chat message and return the agent's response."""
        try:
            result = self.agent_executor.invoke({
                "input": message,
                "chat_history": chat_history or ""
            })
            return result["output"]
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"I encountered an error while processing your request: {str(e)}"
    
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
