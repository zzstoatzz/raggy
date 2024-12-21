import argparse
import os
import sys
import asyncio
from pathlib import Path
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from pydantic_ai.models import ModelMessage
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markdown import CodeBlock, Markdown
from rich.syntax import Syntax
from rich.text import Text

from pydantic_ai import Agent


# Prettify code fences with Rich
class SimpleCodeBlock(CodeBlock):
    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        code = str(self.text).rstrip()
        yield Text(self.lexer_name, style="dim")
        yield Syntax(
            code,
            self.lexer_name,
            theme=self.theme,
            background_color="default",
            word_wrap=True,
        )
        yield Text(f"/{self.lexer_name}", style="dim")


Markdown.elements["fence"] = SimpleCodeBlock


def app() -> int:
    parser = argparse.ArgumentParser(
        prog="aicli",
        description="""\
Pydantic AI powered CLI

Special prompts:
* `show-markdown` - show the markdown output from the previous response
* `multiline` - toggle multiline mode
""",
    )
    parser.add_argument("prompt", nargs="?", help="AI Prompt, else interactive mode")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    parser.add_argument("--version", action="store_true", help="Show version and exit")

    args = parser.parse_args()

    console = Console()
    console.print("Pydantic AI CLI", style="green bold", highlight=False)
    if args.version:
        return 0

    # Check for an API key (e.g. OPENAI_API_KEY)
    if "OPENAI_API_KEY" not in os.environ:
        console.print(
            "You must set the OPENAI_API_KEY environment variable", style="red"
        )
        return 1

    # Create your agent; we set a global system prompt
    agent: Agent[None, str] = Agent(
        "openai:gpt-4o",
        system_prompt="Be a helpful assistant and respond in concise markdown.",
    )

    # We'll accumulate the conversation in here (both user and assistant messages)
    conversation: list[ModelMessage] = []
    stream = not args.no_stream

    # If the user supplied a single prompt, just run once
    if args.prompt:
        try:
            conversation = asyncio.run(
                run_and_display(agent, args.prompt, conversation, stream, console)
            )
        except KeyboardInterrupt:
            pass
        return 0

    # Otherwise, interactive mode with prompt_toolkit
    history = Path.home() / ".openai-prompt-history.txt"
    session = PromptSession[str](history=FileHistory(str(history)))
    multiline = False

    while True:
        try:
            text = session.prompt(
                "aicli ➤ ", auto_suggest=AutoSuggestFromHistory(), multiline=multiline
            )
        except (KeyboardInterrupt, EOFError):
            return 0

        cmd = text.lower().strip()
        if not cmd:
            continue

        if cmd == "show-markdown":
            # Show last assistant message
            if not conversation:
                console.print("No messages yet.", style="dim")
                continue
            # The last run result's assistant message is the last item
            # (the user might have broken the loop, so we search from end)
            assistant_msg = None
            for m in reversed(conversation):
                if m.kind == "response":
                    # Collect text parts from the response
                    text_part = "".join(
                        p.content for p in m.parts if p.part_kind == "text"
                    )
                    assistant_msg = text_part
                    break
            if assistant_msg:
                console.print("[dim]Last assistant markdown output:[/dim]\n")
                console.print(
                    Syntax(assistant_msg, lexer="markdown", background_color="default")
                )
            else:
                console.print("No assistant response found.", style="dim")
            continue

        elif cmd == "multiline":
            multiline = not multiline
            if multiline:
                console.print(
                    "Enabling multiline mode. "
                    "[dim]Press [Meta+Enter] or [Esc] then [Enter] to submit.[/dim]"
                )
            else:
                console.print("Disabling multiline mode.")
            continue

        # Normal user prompt
        try:
            conversation = asyncio.run(
                run_and_display(agent, text, conversation, stream, console)
            )
        except KeyboardInterrupt:
            return 0

    return 0


async def run_and_display(
    agent: Agent[None, str],
    user_text: str,
    conversation: list[ModelMessage],
    stream: bool,
    console: Console,
):
    """
    Runs the agent (stream or not) with user_text, returning the updated conversation.
    If conversation is None, run from scratch (includes system prompt).
    Otherwise pass conversation as message_history to continue it.
    """
    console.print("\nResponse:", style="green")

    with Live(
        "[dim]Working on it…[/dim]",
        console=console,
        refresh_per_second=15,
        vertical_overflow="visible",
    ) as live:
        if stream:
            async with agent.run_stream(user_text, message_history=conversation) as run:
                try:
                    async for chunk in run.stream_text():
                        live.update(Markdown(chunk))
                except Exception as e:
                    console.print(f"Error: {e}", style="red")
            new_conversation = run.all_messages()
        else:
            run_result = await agent.run(user_text, message_history=conversation)
            live.update(Markdown(run_result.data))
            new_conversation = run_result.all_messages()

    return new_conversation


if __name__ == "__main__":
    sys.exit(app())
