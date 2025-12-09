import click
import argparse
import sys
from argparse import Namespace


@click.command()
@click.option("--infer_round", default=1, type=int, help="iteration round")
@click.option("--top_rank", default=1, type=int, help="top rank")
@click.option("--min_mer", default=8, type=int, help="prepare k-mer files")
@click.option("--max_mer", default=13, type=int, help="prepare k-mer files")
@click.option(
    "--input_sequences_file",
    default="data/humab25_sequences.csv",
    type=str,
    help="Input CSV file for sequences",
)
@click.option(
    "--fixed_mutations_file",
    default=None,
    type=str,
    help="Optional: Path to file defining fixed CDR regions",
)
def get_conf(**kwargs):
    return argparse.Namespace(**kwargs)


class ConfNamespace:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if value == "None":
                value = None
            setattr(self, key, value)


# This is a bit of a hack to make click arguments available globally
# without refactoring all the scripts that import 'conf'.


# Create a dummy command function that will be decorated
@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option(
    "--infer_round", default=1, type=int, help="Iteration round for inference."
)
@click.option(
    "--top_rank", default=1, type=int, help="Top rank to consider in inference."
)
@click.option("--min_mer", default=8, type=int, help="Minimum k-mer length.")
@click.option("--max_mer", default=13, type=int, help="Maximum k-mer length.")
@click.option(
    "--input_sequences_file",
    default="data/humab25_sequences.csv",
    help="Path to the input sequences CSV file.",
)
@click.option(
    "--fixed_mutations_file",
    default=None,
    type=str,
    help="Path to the fixed mutations file.",
)
def cli(**kwargs):
    """This function is a dummy to hold the click decorators."""
    pass


# Parse the arguments and expose them in a 'conf' object
# The 'standalone_mode=False' prevents click from taking over the script execution.
try:
    # Use a context to parse args without executing the command function
    with cli.make_context(sys.argv[0], sys.argv[1:]) as ctx:
        conf = Namespace(**ctx.params)
except Exception:
    # Fallback to defaults if parsing fails (e.g., when running in a notebook)
    conf = Namespace(
        infer_round=1,
        top_rank=1,
        min_mer=8,
        max_mer=13,
        input_sequences_file="data/humab25_sequences.csv",
        fixed_mutations_file=None,
    )
