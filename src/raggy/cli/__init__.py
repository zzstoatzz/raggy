import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import openai
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markdown import CodeBlock, Markdown
from rich.status import Status
from rich.syntax import Syntax
from rich.text import Text


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
OpenAI powered AI CLI (thank you samuelcolvin)

Special prompts:
* `show-markdown` - show the markdown output from the previous response
* `multiline` - toggle multiline mode
""",
    )
    parser.add_argument(
        "prompt", nargs="?", help="AI Prompt, if omitted fall into interactive mode"
    )

    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Whether to stream responses from OpenAI",
    )

    parser.add_argument("--version", action="store_true", help="Show version and exit")

    args = parser.parse_args()

    console = Console()
    console.print("OpenAI powered AI CLI", style="green bold", highlight=False)
    if args.version:
        return 0

    try:
        openai_api_key = os.environ["OPENAI_API_KEY"]
    except KeyError:
        console.print(
            "You must set the OPENAI_API_KEY environment variable", style="red"
        )
        return 1

    client = openai.OpenAI(api_key=openai_api_key)

    now_utc = datetime.now(timezone.utc)
    t = now_utc.astimezone().tzinfo.tzname(now_utc)  # type: ignore
    setup = f"""\
Help the user by responding to their request, the output should 
be concise and always written in markdown. The current date and time
is {datetime.now()} {t}. The user is running {sys.platform}."""

    stream = not args.no_stream
    messages = [{"role": "system", "content": setup}]

    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})
        try:
            ask_openai(client, messages, stream, console)
        except KeyboardInterrupt:
            pass
        return 0

    history = Path().home() / ".openai-prompt-history.txt"
    session = PromptSession(history=FileHistory(str(history)))
    multiline = False

    while True:
        try:
            text = session.prompt(
                "aicli ➤ ", auto_suggest=AutoSuggestFromHistory(), multiline=multiline
            )
        except (KeyboardInterrupt, EOFError):
            return 0

        if not text.strip():
            continue

        ident_prompt = text.lower().strip(" ").replace(" ", "-")
        if ident_prompt == "show-markdown":
            last_content = messages[-1]["content"]
            console.print("[dim]Last markdown output of last question:[/dim]\n")
            console.print(
                Syntax(last_content, lexer="markdown", background_color="default")
            )
            continue
        elif ident_prompt == "multiline":
            multiline = not multiline
            if multiline:
                console.print(
                    "Enabling multiline mode. "
                    "[dim]Press [Meta+Enter] or [Esc] followed by [Enter] to accept input.[/dim]"
                )
            else:
                console.print("Disabling multiline mode.")
            continue

        messages.append({"role": "user", "content": text})

        try:
            content = ask_openai(client, messages, stream, console)
        except KeyboardInterrupt:
            return 0
        messages.append({"role": "assistant", "content": content})


def ask_openai(
    client: openai.OpenAI,
    messages: list[dict[str, str]],
    stream: bool,
    console: Console,
) -> str:
    with Status("[dim]Working on it…[/dim]", console=console):
        response = client.chat.completions.create(
            model="gpt-4", messages=messages, stream=stream
        )

    console.print("\nResponse:", style="green")
    if stream:
        content = ""
        interrupted = False
        with Live("", refresh_per_second=15, console=console) as live:
            try:
                for chunk in response:
                    if chunk.choices[0].finish_reason is not None:
                        break
                    chunk_text = chunk.choices[0].delta.content
                    content += chunk_text
                    live.update(Markdown(content))
            except KeyboardInterrupt:
                interrupted = True

        if interrupted:
            console.print("[dim]Interrupted[/dim]")
    else:
        content = response.choices[0].message.content
        console.print(Markdown(content))

    return content


if __name__ == "__main__":
    sys.exit(app())
