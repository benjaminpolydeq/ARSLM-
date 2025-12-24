"""
Command-line interface for ARSLM.
"""

import click
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

console = Console()


@click.group()
@click.version_option()
def main():
    """
    ARSLM - Adaptive Reasoning Semantic Language Model
    
    Lightweight AI engine for intelligent response generation.
    """
    pass


@main.command()
@click.option(
    '--prompt',
    '-p',
    required=True,
    help='Text prompt for generation'
)
@click.option(
    '--max-length',
    '-l',
    default=100,
    help='Maximum generation length'
)
@click.option(
    '--temperature',
    '-t',
    default=1.0,
    help='Sampling temperature'
)
@click.option(
    '--model-path',
    '-m',
    help='Path to model checkpoint'
)
def generate(prompt, max_length, temperature, model_path):
    """Generate text from a prompt."""
    try:
        from arslm import ARSLM
        
        console.print(f"\n[bold cyan]Loading model...[/bold cyan]")
        
        if model_path:
            model = ARSLM.from_pretrained(model_path)
        else:
            model = ARSLM()
        
        console.print(f"[bold green]✓ Model loaded[/bold green]\n")
        
        console.print(Panel(f"[yellow]{prompt}[/yellow]", title="Prompt"))
        
        console.print("\n[bold cyan]Generating...[/bold cyan]")
        
        response = model.generate(
            prompt,
            max_length=max_length,
            temperature=temperature
        )
        
        console.print(Panel(response, title="Response", border_style="green"))
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1)


@main.command()
@click.option(
    '--host',
    '-h',
    default='0.0.0.0',
    help='Server host'
)
@click.option(
    '--port',
    '-p',
    default=8000,
    help='Server port'
)
@click.option(
    '--model-path',
    '-m',
    help='Path to model checkpoint'
)
@click.option(
    '--reload',
    is_flag=True,
    help='Enable auto-reload'
)
def serve(host, port, model_path, reload):
    """Start ARSLM API server."""
    try:
        import uvicorn
        
        console.print(f"\n[bold cyan]Starting ARSLM server...[/bold cyan]")
        console.print(f"Host: {host}")
        console.print(f"Port: {port}")
        if model_path:
            console.print(f"Model: {model_path}")
        console.print()
        
        uvicorn.run(
            "arslm.api.server:app",
            host=host,
            port=port,
            reload=reload
        )
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1)


@main.command()
@click.option(
    '--model-path',
    '-m',
    help='Path to model checkpoint'
)
def info(model_path):
    """Display model information."""
    try:
        from arslm import ARSLM
        
        if model_path:
            model = ARSLM.from_pretrained(model_path)
        else:
            model = ARSLM()
        
        # Create info table
        table = Table(title="ARSLM Model Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        config = model.config
        table.add_row("Vocabulary Size", str(config.vocab_size))
        table.add_row("Hidden Size", str(config.hidden_size))
        table.add_row("Number of Layers", str(config.num_layers))
        table.add_row("Number of Heads", str(config.num_heads))
        table.add_row("Max Length", str(config.max_length))
        table.add_row("Total Parameters", f"{model.num_parameters():,}")
        table.add_row("Trainable Parameters", f"{model.num_parameters(only_trainable=True):,}")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1)


@main.command()
@click.option(
    '--texts',
    '-t',
    multiple=True,
    required=True,
    help='Training texts'
)
@click.option(
    '--output',
    '-o',
    required=True,
    help='Output tokenizer path'
)
@click.option(
    '--vocab-size',
    '-v',
    default=10000,
    help='Vocabulary size'
)
def train_tokenizer(texts, output, vocab_size):
    """Train a tokenizer on text data."""
    try:
        from arslm.utils.tokenizer import ARSLMTokenizer
        
        console.print(f"\n[bold cyan]Training tokenizer...[/bold cyan]")
        
        tokenizer = ARSLMTokenizer()
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Building vocabulary...", total=1)
            tokenizer.build_vocab(
                list(texts),
                vocab_size=vocab_size
            )
            progress.update(task, advance=1)
        
        tokenizer.save(output)
        
        console.print(f"[bold green]✓ Tokenizer saved to {output}[/bold green]")
        console.print(f"Vocabulary size: {tokenizer.vocab_size}")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1)


@main.command()
def chat():
    """Interactive chat with ARSLM."""
    try:
        from arslm import ARSLM
        
        console.print(Panel(
            "[bold cyan]ARSLM Interactive Chat[/bold cyan]\n"
            "Type 'exit' or 'quit' to end the conversation",
            border_style="cyan"
        ))
        
        model = ARSLM()
        session_id = "cli_session"
        
        while True:
            try:
                user_input = console.input("\n[bold green]You:[/bold green] ")
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    console.print("\n[bold cyan]Goodbye![/bold cyan]")
                    break
                
                if not user_input.strip():
                    continue
                
                response = model.generate(user_input)
                
                console.print(f"[bold blue]ARSLM:[/bold blue] {response}")
                
            except KeyboardInterrupt:
                console.print("\n\n[bold cyan]Goodbye![/bold cyan]")
                break
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1)


@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option(
    '--output',
    '-o',
    help='Output file'
)
def process_file(input_file, output):
    """Process text file with ARSLM."""
    try:
        from arslm import ARSLM
        
        console.print(f"\n[bold cyan]Processing file: {input_file}[/bold cyan]")
        
        model = ARSLM()
        
        # Read input
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Process
        result = model.generate(text)
        
        # Output
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(result)
            console.print(f"[bold green]✓ Result saved to {output}[/bold green]")
        else:
            console.print(Panel(result, title="Result", border_style="green"))
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1)


if __name__ == '__main__':
    main()