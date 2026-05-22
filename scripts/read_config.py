from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import ArgumentParser
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


def make_table(title, data):
    table = Table(
        title=f"[bold]{title}[/bold]",
        title_style="bright_white",
        width=80,
        box=None,
        header_style="bold bright_cyan",
        padding=(0, 2),
    )
    table.add_column("Key", style="bright_yellow", no_wrap=True)
    table.add_column("Value", overflow="fold")
    for k, v in sorted(data.items()):
        val = str(v)
        style = "dim" if val in ("None", "False", "[]", "{}") else "bright_white"
        table.add_row(str(k), Text(val, style=style))
    return table


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_src")

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_src)
    model = AutoModelForCausalLM.from_pretrained(args.model_src, device_map="auto")

    console = Console()
    console.print()
    console.print(Panel(f"{args.model_src}", border_style="bright_cyan", expand=False))
    console.print(make_table("Model Config", model.config.to_dict()))

    if qc := getattr(model.config, "quantization_config", None):
        qc_dict = dict(qc) if isinstance(qc, dict) else qc.to_dict()

        console.print()
        console.print(make_table("Quantization Config", qc_dict))
