"""
Command-line interface for Sippy Agent.
"""

import logging
import sys
from typing import Optional
import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.logging import RichHandler

from .agent import SippyAgent
from .config import Config

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Setup logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


class SippyCLI:
    """Command-line interface for the Sippy Agent."""
    
    def __init__(self, config: Config):
        """Initialize the CLI with configuration."""
        self.config = config
        self.agent = SippyAgent(config)
        self.chat_history = []
    
    def display_welcome(self) -> None:
        """Display welcome message."""
        welcome_text = Text()
        welcome_text.append("🔧 ", style="bold blue")
        welcome_text.append("Sippy AI Agent", style="bold cyan")
        welcome_text.append(" - Your CI/CD Analysis Assistant", style="bold white")
        
        welcome_panel = Panel(
            welcome_text,
            title="Welcome",
            border_style="blue",
            padding=(1, 2)
        )
        console.print(welcome_panel)
        console.print()
        
        # Display available tools
        tools = self.agent.list_tools()
        tools_text = "Available tools: " + ", ".join(f"[bold green]{tool}[/bold green]" for tool in tools)
        console.print(tools_text)
        console.print()
        console.print("[dim]Type 'help' for commands, 'quit' or 'exit' to leave[/dim]")
        console.print()
    
    def display_help(self) -> None:
        """Display help information."""
        help_text = """
[bold cyan]Sippy AI Agent Commands:[/bold cyan]

[bold green]help[/bold green]     - Show this help message
[bold green]tools[/bold green]    - List available tools
[bold green]history[/bold green]  - Show chat history
[bold green]clear[/bold green]    - Clear chat history
[bold green]quit[/bold green]     - Exit the application
[bold green]exit[/bold green]     - Exit the application

[bold cyan]Example queries:[/bold cyan]
• "Analyze job 12345 for failures"
• "What are the common test failures for test_login?"
• "Show me patterns in recent CI failures"
"""
        console.print(Panel(help_text, title="Help", border_style="green"))
    
    def display_tools(self) -> None:
        """Display available tools."""
        tools = self.agent.list_tools()
        tools_text = "\n".join(f"• [bold green]{tool}[/bold green]" for tool in tools)
        console.print(Panel(tools_text, title="Available Tools", border_style="blue"))
    
    def display_history(self) -> None:
        """Display chat history."""
        if not self.chat_history:
            console.print("[dim]No chat history yet.[/dim]")
            return
        
        history_text = ""
        for i, (user_msg, agent_msg) in enumerate(self.chat_history, 1):
            history_text += f"[bold blue]{i}. User:[/bold blue] {user_msg}\n"
            history_text += f"[bold green]   Agent:[/bold green] {agent_msg}\n\n"
        
        console.print(Panel(history_text.strip(), title="Chat History", border_style="yellow"))
    
    def clear_history(self) -> None:
        """Clear chat history."""
        self.chat_history.clear()
        console.print("[green]Chat history cleared.[/green]")
    
    def process_user_input(self, user_input: str) -> bool:
        """Process user input and return False if should exit."""
        user_input = user_input.strip()
        
        # Handle special commands
        if user_input.lower() in ['quit', 'exit']:
            return False
        elif user_input.lower() == 'help':
            self.display_help()
            return True
        elif user_input.lower() == 'tools':
            self.display_tools()
            return True
        elif user_input.lower() == 'history':
            self.display_history()
            return True
        elif user_input.lower() == 'clear':
            self.clear_history()
            return True
        elif not user_input:
            return True
        
        # Process with agent
        try:
            with console.status("[bold green]Thinking...", spinner="dots"):
                # Prepare chat history for context
                history_context = "\n".join([
                    f"User: {user_msg}\nAssistant: {agent_msg}"
                    for user_msg, agent_msg in self.chat_history[-3:]  # Last 3 exchanges
                ])
                
                response = self.agent.chat(user_input, history_context)
            
            # Display response
            console.print()
            console.print(Panel(response, title="Sippy AI", border_style="green"))
            console.print()
            
            # Add to history
            self.chat_history.append((user_input, response))
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user.[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {str(e)}[/red]")
        
        return True
    
    def run(self) -> None:
        """Run the interactive CLI."""
        self.display_welcome()
        
        try:
            while True:
                user_input = Prompt.ask("[bold blue]You")
                
                if not self.process_user_input(user_input):
                    break
                    
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
        except EOFError:
            console.print("\n[yellow]Goodbye![/yellow]")


@click.command()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--model', default='llama3.1:8b', help='Model name to use (e.g., llama3.1:8b, gpt-4)')
@click.option('--endpoint', default='http://localhost:11434/v1', help='LLM API endpoint')
@click.option('--temperature', default=0.1, type=float, help='Temperature for the model')
def main(verbose: bool, model: str, endpoint: str, temperature: float) -> None:
    """Sippy AI Agent - Your CI/CD Analysis Assistant."""
    setup_logging(verbose)
    
    try:
        # Create configuration
        config = Config.from_env()
        config.verbose = verbose
        config.model_name = model
        config.llm_endpoint = endpoint
        config.temperature = temperature
        
        # Create and run CLI
        cli = SippyCLI(config)
        cli.run()
        
    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == '__main__':
    main()
