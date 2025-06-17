"""
Core Re-Act agent implementation for Sippy.
"""

import logging
from typing import List, Optional, Union
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import BaseTool

from .config import Config
from .tools import (
    ExampleTool,
    SippyJobAnalysisTool,
    SippyTestFailureTool,
    SippyProwJobSummaryTool,
    SippyLogAnalyzerTool,
    SippyJiraIncidentTool,
    SippyReleasePayloadTool,
    SippyPayloadDetailsTool
)

logger = logging.getLogger(__name__)


class SippyAgent:
    """LangChain Re-Act agent for CI analysis with Sippy."""
    
    def __init__(self, config: Config):
        """Initialize the Sippy agent with configuration."""
        self.config = config
        self.llm = self._create_llm()
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent_executor()
    
    def _create_llm(self) -> Union[ChatOpenAI, ChatGoogleGenerativeAI]:
        """Create the language model instance."""
        if self.config.verbose:
            logger.info(f"Creating LLM with endpoint: {self.config.llm_endpoint}")
            logger.info(f"Using model: {self.config.model_name}")

        # Use ChatGoogleGenerativeAI for Gemini models
        if self.config.is_gemini_model():
            if not self.config.google_api_key:
                raise ValueError("Google API key is required for Gemini models")

            llm_kwargs = {
                "model": self.config.model_name,
                "temperature": self.config.temperature,
                "google_api_key": self.config.google_api_key,
            }

            if self.config.verbose:
                logger.info(f"Using ChatGoogleGenerativeAI for Gemini model")

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
            SippyReleasePayloadTool(),
            SippyPayloadDetailsTool(),
        ]
        
        if self.config.verbose:
            logger.info(f"Created {len(tools)} tools: {[tool.name for tool in tools]}")
        
        return tools
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the Re-Act agent executor."""
        # Custom prompt template for Sippy CI analysis
        prompt_template = """You are Sippy AI, an expert assistant for analyzing CI job and test failures.

🚨 CRITICAL EFFICIENCY RULES - READ FIRST:
==========================================
1. If user asks for information available in the job summary, DO NOT search logs! However, if you need additional information consider searching the build logs for errors.
2. READ tool responses carefully - extract information directly before calling more tools
3. Call check_known_incidents only ONCE per analysis, not per job
4. Use information you already have instead of making redundant tool calls
5. 🚨 NEVER call the same tool with the same parameters twice! If you already called analyze_job_logs with job ID X and pathGlob Y, use those results!
6. If a tool call didn't give you what you need, try DIFFERENT parameters, don't repeat the same call

You have access to tools that can help you analyze CI jobs, and test failures.

When users ask about CI issues, use the available tools to gather information and provide detailed analysis. Pay attention to
the user's query and ensure you are answering the direction question they gave you.

Example: If the question is answerable by the first tool call, you don't need to continue on.

CI JOB ANALYSIS WORKFLOW:
-------------------------
When analyzing a job failure, follow this recommended workflow:
1. First, use get_prow_job_summary with just the numeric job ID (e.g., 1934795512955801600)
2. Then, use analyze_job_logs with the same numeric job ID to get detailed error context from build logs
3. ANALYZE THE ACTUAL ERRORS: Look at what specifically failed in the job
4. Only then use check_known_incidents with search terms that match the ACTUAL errors found
5. If needed, use analyze_job_logs with different path_glob patterns for additional insights

PAYLOAD ANALYSIS WORKFLOW:
-------------------------
When users ask about release payloads, follow this two-stage approach:

STAGE 1 - Basic Status (for questions like "tell me about payload X"):
1. Use get_release_payloads with the payload_name parameter to get basic status
2. Report whether the payload was accepted/rejected/ready
3. If rejected, offer to investigate WHY: "This payload was rejected! Would you like details about why?"
4. STOP HERE unless user asks for details

STAGE 2 - Detailed Analysis (only when user asks for details about WHY a payload failed):
1. Use get_payload_details to get failed job IDs
2. Analyze 2-3 key failed jobs:
   - get_prow_job_summary for each job ID
   - analyze_job_logs for each job ID
   - Collect error patterns from ALL jobs first
3. ONLY AFTER analyzing all jobs, check incidents ONCE:
   - Use check_known_incidents with the most common error patterns found
   - Do NOT call check_known_incidents for each individual job
4. Summarize findings with correlation to incidents

IMPORTANT: Be decisive and efficient. Don't over-analyze or repeat incident checks.

ANALYZING TEST FAILURES:
------------------------
When the job summary shows test failures, provide detailed analysis:
1. Examine the specific test names - they indicate the failure area (e.g., [sig-network], [sig-storage], [sig-auth])
2. Look at the test failure messages for specific error details and root causes
3. Identify patterns in test names (e.g., multiple networking tests suggest networking issues)
4. Explain what the failing tests are trying to validate and why they might have failed
5. Provide actionable insights based on the actual test failure content
6. Do NOT just say "test failures occurred" - analyze the specific failures and their implications

EVIDENCE-BASED ANALYSIS:
-----------------------
Always base your conclusions on the actual evidence from the job:
- If logs show "failed to install" → investigate installation issues
- If logs show "test failed" → focus on test failures
- If logs show "timeout" → then check for timeout-related incidents
- When considering an incident, the job's start time should be no earlier than 12 hours before the incident was created
- Prioritize direct evidence (log entries, etc) when correlating with incidents

Do not assume correlation without evidence. Many job failures are unrelated to infrastructure incidents.

CORRELATING WITH KNOWN ISSUES:
-----------------------------
After identifying error patterns use check_known_incidents with relevant search terms to see if this is a known problem. For example:
- Test failure: search for key words in the test name
- Registry errors: search for "registry"
- Timeout issues: search for "timeout"
- Infrastructure: search for "infrastructure", "node"

IMPORTANT: Only correlate job failures with known incidents when there is CLEAR EVIDENCE of a connection.

IMPORTANT: Always pass ONLY the numeric job ID to tools, never include extra text or descriptions.

IMPORTANT: Only correlate with a known issue when you're sure it's related, make sure the failure symptoms and incident description match.

IMPORTANT: Don't call the same tool with the same arguments multiple times.

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

EXAMPLE - User asks "What's the URL for job 1234567890":
CORRECT:
```
Thought: I need to get the job summary which contains the URL
Action: get_prow_job_summary
Action Input: 1234567890
Observation: {{"url": "https://prow.ci.openshift.org/view/...", ...}}
Thought: I have the URL from the summary
Final Answer: The URL is https://prow.ci.openshift.org/view/...
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
            max_execution_time=1800,  # 30 minute timeout
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
