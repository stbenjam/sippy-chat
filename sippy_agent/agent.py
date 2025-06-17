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
from .tools import (
    ExampleTool,
    SippyJobAnalysisTool,
    SippyTestFailureTool,
    SippyProwJobSummaryTool,
    SippyLogAnalyzerTool,
    SippyJiraIncidentTool,
    SippyReleasePayloadTool
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
            SippyReleasePayloadTool(),
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
3. ANALYZE THE ACTUAL ERRORS: Look at what specifically failed in the job
4. Only then use check_known_incidents with search terms that match the ACTUAL errors found
5. If needed, use analyze_job_logs with different path_glob patterns for additional insights

ANALYZING TEST FAILURES:
------------------------
When the job summary shows test failures, provide detailed analysis:
1. Examine the specific test names - they indicate the failure area (e.g., [sig-network], [sig-storage], [sig-auth])
2. Look at the test failure messages for specific error details and root causes
3. Identify patterns in test names (e.g., multiple networking tests suggest networking issues)
4. Explain what the failing tests are trying to validate and why they might have failed
5. Provide actionable insights based on the actual test failure content
6. Do NOT just say "test failures occurred" - analyze the specific failures and their implications

ANALYZING INSTALLATION FAILURES:
-------------------------------
When a job fails during installation or shows "failed to install" errors:
1. Look for cluster operator status information in test failures or logs
2. Check if any operators are in Degraded, Progressing, or Available=False states
3. Identify which specific operators are failing (e.g., network, storage, authentication)
4. Look for operator-specific error messages that explain why they failed
5. Focus on the root cause of operator failures rather than assuming infrastructure issues
6. Common operator failure patterns:
   - Network operator: DNS, CNI, or network configuration issues
   - Storage operator: PV provisioning or storage class problems
   - Authentication operator: Certificate or RBAC issues
   - Image registry operator: Internal registry configuration (different from external registry issues)

Example good analysis: "Installation failed due to the network operator being in a Degraded state with error 'failed to configure CNI', indicating a cluster networking configuration issue."

EVIDENCE-BASED ANALYSIS:
-----------------------
Always base your conclusions on the actual evidence from the job:
- If logs show "failed to install" → investigate installation issues, not registry problems
- If logs show "test failed" → focus on test failures, not infrastructure
- If logs show "registry 503 errors" → then check for registry incidents
- If logs show "timeout" → then check for timeout-related incidents

Do not assume correlation without evidence. Many job failures are unrelated to infrastructure incidents.

CORRELATING WITH KNOWN ISSUES:
-----------------------------
IMPORTANT: Only correlate job failures with known incidents when there is CLEAR EVIDENCE of a connection.

After identifying specific error patterns, use check_known_incidents to see if there are related known issues:
- Infrastructure issues: search for "infrastructure", "node", "capacity" - BUT only if logs show infrastructure problems
- Network problems: search for "network", "dns", "connectivity" - BUT only if network-related failures are evident
- Storage issues: search for "storage", "pv", "volume" - BUT only if storage operations are failing
- Authentication problems: search for "auth", "rbac", "certificate" - BUT only if auth-related errors appear
- Registry errors: search for "registry" - BUT only if the job logs show external registry connectivity issues

CRITICAL: Do NOT assume a job failure is related to an open incident unless:
1. The job's actual error messages match the incident description
2. The failure symptoms are specifically mentioned in the incident
3. The timing/cluster/infrastructure details align

If a job failed for reasons unrelated to open incidents (e.g., test failures, installation issues, code problems), clearly state that the failure appears unrelated to known incidents.

EXAMPLES OF WHEN NOT TO CORRELATE:
- Job failed due to "failed to install" with operator errors, but open incidents are about external registry issues
- Test failures in application code when incidents are about infrastructure capacity
- Installation failures due to configuration issues when incidents are about network outages
- Operator degradation due to misconfiguration when incidents are about hardware problems

EXAMPLES OF WHEN TO CORRELATE:
- Job shows "network operator degraded: DNS resolution failed" and there's a DNS infrastructure incident
- Job times out during cluster provisioning and there's an infrastructure capacity incident
- Installation fails with "storage operator unavailable" and there's a storage backend incident
- Job fails with specific error messages that directly match incident descriptions

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
