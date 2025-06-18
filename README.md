# Sippy AI Agent

A LangChain Re-Act AI Agent for analyzing CI/CD pipelines, test failures, and build issues using the Sippy platform.

## Features

- 🤖 **LangChain Re-Act Agent**: Intelligent reasoning and action-taking capabilities
- 🧠 **Thinking Display**: Optional visualization of the agent's thought process
- 🔧 **CI/CD Analysis**: Tools for analyzing jobs, test failures, and build patterns
- 💬 **Interactive CLI**: Rich command-line interface with chat functionality
- 🛠️ **Extensible Tools**: Modular tool system ready for Sippy API integration
- ⚙️ **Configurable**: Environment-based configuration management

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd sippy-chat

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` for your LLM setup:

**For local Ollama (default):**
```env
LLM_ENDPOINT=http://localhost:11434/v1
MODEL_NAME=llama3.1:8b
```

**For OpenAI:**
```env
LLM_ENDPOINT=https://api.openai.com/v1
MODEL_NAME=gpt-3.5-turbo
OPENAI_API_KEY=your_openai_api_key_here
```

**For Google Gemini:**
```env
MODEL_NAME=gemini-1.5-pro
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Run the Agent

```bash
python main.py
```

Or with options:

```bash
# Using Ollama with a different model and thinking display
python main.py --verbose --thinking --model llama3.1:70b --temperature 0.2

# Using OpenAI with thinking process visible
python main.py --thinking --model gpt-4 --endpoint https://api.openai.com/v1

# Using Google Gemini
python main.py --model gemini-1.5-pro
```

## Thinking Display

The agent supports a "thinking display" mode that shows the LLM's reasoning process:

```bash
# Enable thinking display from command line
python main.py --thinking

# Or toggle it during runtime
> thinking
```

When enabled, you'll see:
- 💭 **Thoughts**: The agent's reasoning about what to do next
- 🔧 **Actions**: Which tools the agent decides to use
- 📝 **Inputs**: The parameters passed to each tool
- 👁️ **Observations**: The results returned from each tool

This is helpful for understanding how the agent approaches complex analysis tasks and debugging when things don't work as expected.

## Usage

Once started, you can interact with the Sippy AI Agent through the CLI:

```
🔧 Sippy AI Agent - Your CI/CD Analysis Assistant

Available tools: example_tool, analyze_job, analyze_test_failures

Type 'help' for commands, 'quit' or 'exit' to leave

You: help
```

### Available Commands

- `help` - Show help message
- `tools` - List available tools
- `history` - Show chat history
- `clear` - Clear chat history
- `thinking` - Toggle showing the agent's thinking process
- `quit` / `exit` - Exit the application

### Example Queries

- "Analyze job 12345 for failures"
- "What are the common test failures for test_login?"
- "Show me patterns in recent CI failures"

## Architecture

### Project Structure

```
sippy-chat/
├── sippy_agent/
│   ├── __init__.py              # Package initialization
│   ├── agent.py                 # Core Re-Act agent
│   ├── cli.py                   # Command-line interface
│   ├── config.py                # Configuration management
│   └── tools/
│       ├── __init__.py          # Tools package exports
│       ├── README.md            # Tools documentation
│       ├── base_tool.py         # Base tool classes
│       ├── sippy_job_summary.py # Job summary tool
│       ├── sippy_log_analyzer.py# Log analysis tool
│       ├── jira_incidents.py    # Jira incident tool
│       ├── placeholder_tools.py # Future tools
│       ├── test_analysis_helpers.py # Test analysis utilities
│       └── log_analysis_helpers.py  # Log analysis utilities
├── main.py                      # Entry point
├── requirements.txt             # Dependencies
├── .env.example                # Environment template
└── README.md                   # This file
```

### Components

1. **SippyAgent**: Core LangChain Re-Act agent with custom prompt for CI analysis
2. **Tools**: Extensible tool system with base classes for Sippy API integration
3. **CLI**: Rich interactive command-line interface with chat functionality
4. **Config**: Environment-based configuration with validation

## Development

### Adding New Tools

To add a new tool for Sippy API integration:

1. Create a new tool class inheriting from `SippyBaseTool`
2. Define the input schema using Pydantic
3. Implement the `_run` method
4. Add the tool to the agent in `agent.py`

Example:

```python
class MyNewTool(SippyBaseTool):
    name: str = "my_new_tool"
    description: str = "Description of what this tool does"

    class MyInput(SippyToolInput):
        param: str = Field(description="Parameter description")

    args_schema: Type[BaseModel] = MyInput

    def _run(self, param: str) -> str:
        # Implement your tool logic here
        return f"Result for {param}"
```

### Configuration Options

The agent supports various configuration options through environment variables:

- `LLM_ENDPOINT`: LLM API endpoint (default: http://localhost:11434/v1 for Ollama)
- `MODEL_NAME`: Model name to use (default: llama3.1:8b)
- `OPENAI_API_KEY`: OpenAI API key (only required when using OpenAI endpoint)
- `SIPPY_API_URL`: Sippy API base URL (for future use)
