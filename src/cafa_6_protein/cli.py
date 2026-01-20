"""CAFA-6 Protein Function Prediction CLI.

Minimal CLI for local evaluation and pipeline execution.
"""

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

app = typer.Typer(
    name="cafa6",
    help="CAFA-6 Protein Function Prediction toolkit",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


@app.command()
def evaluate(
    predictions: Annotated[
        Path,
        typer.Argument(
            help="Directory containing prediction files",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    ground_truth: Annotated[
        Path,
        typer.Argument(
            help="Ground truth file (TSV: protein_id, GO_term)",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ],
    ontology: Annotated[
        Path,
        typer.Option(
            "--ontology",
            "-o",
            help="GO ontology file (OBO format)",
        ),
    ] = Path("data/Train/go-basic.obo"),
    ia_file: Annotated[
        Path | None,
        typer.Option(
            "--ia",
            "-i",
            help="Information Accretion weights file",
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output",
            "-O",
            help="Output directory for results",
        ),
    ] = Path("results"),
    threads: Annotated[
        int,
        typer.Option(
            "--threads",
            "-t",
            help="Number of threads (0 = all available)",
        ),
    ] = 0,
    propagate: Annotated[
        str,
        typer.Option(
            "--propagate",
            "-p",
            help="Ancestor propagation strategy: max or fill",
        ),
    ] = "max",
) -> None:
    """Evaluate predictions using CAFA-evaluator.

    Runs the official CAFA-evaluator to compute F-max and S-min metrics.

    [bold]Example:[/bold]
        cafa6 evaluate ./predictions ./ground_truth.tsv -i data/IA.tsv
    """
    # Validate ontology exists
    if not ontology.exists():
        console.print(f"[red]Error:[/red] Ontology file not found: {ontology}")
        raise typer.Exit(1)

    console.print(
        Panel.fit("[bold blue]CAFA-6 Local Evaluation[/bold blue]", subtitle="Using CAFA-evaluator")
    )

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "cafaeval",
        str(ontology),
        str(predictions),
        str(ground_truth),
        "-out_dir",
        str(output_dir),
        "-threads",
        str(threads),
        "-prop",
        propagate,
    ]

    if ia_file and ia_file.exists():
        cmd.extend(["-ia", str(ia_file)])
        console.print(f"[dim]Using IA weights:[/dim] {ia_file}")

    # Display config
    table = Table(title="Evaluation Configuration", show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Predictions", str(predictions))
    table.add_row("Ground Truth", str(ground_truth))
    table.add_row("Ontology", str(ontology))
    table.add_row("Output", str(output_dir))
    table.add_row("Threads", str(threads) if threads > 0 else "all")
    table.add_row("Propagation", propagate)
    console.print(table)

    console.print("\n[bold]Running CAFA-evaluator...[/bold]\n")

    # Run evaluation
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        console.print(f"\n[green]✓ Evaluation complete![/green] Results saved to: {output_dir}")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Evaluation failed with exit code {e.returncode}[/red]")
        raise typer.Exit(1) from None
    except FileNotFoundError:
        console.print("[red]✗ cafaeval not found. Install with:[/red]")
        console.print("  uv pip install git+https://github.com/BioComputingUP/CAFA-evaluator.git")
        raise typer.Exit(1) from None


@app.command()
def validate(
    submission: Annotated[
        Path,
        typer.Argument(
            help="Submission file to validate",
            exists=True,
        ),
    ],
    sample: Annotated[
        Path,
        typer.Option(
            "--sample",
            "-s",
            help="Sample submission file for format reference",
        ),
    ] = Path("data/sample_submission.tsv"),
) -> None:
    """Validate submission file format.

    Checks that the submission file has the correct format for Kaggle.
    """
    import pandas as pd

    console.print(Panel.fit("[bold blue]Submission Validation[/bold blue]"))

    errors = []
    warnings = []

    try:
        df = pd.read_csv(submission, sep="\t" if submission.suffix == ".tsv" else ",")
    except Exception as e:
        console.print(f"[red]✗ Failed to read file:[/red] {e}")
        raise typer.Exit(1) from None

    # Check columns
    expected_cols = {"Protein Id", "GO Term Id", "Prediction"}
    if set(df.columns) != expected_cols:
        errors.append(f"Expected columns {expected_cols}, got {set(df.columns)}")

    # Check for missing values
    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                errors.append(f"Column '{col}' has {count} missing values")

    # Check prediction range
    if "Prediction" in df.columns:
        pred_min, pred_max = df["Prediction"].min(), df["Prediction"].max()
        if pred_min < 0 or pred_max > 1:
            errors.append(f"Predictions must be in [0, 1], got [{pred_min:.4f}, {pred_max:.4f}]")

    # Check GO term format
    if "GO Term Id" in df.columns:
        invalid_go = df[~df["GO Term Id"].str.match(r"^GO:\d{7}$", na=False)]
        if len(invalid_go) > 0:
            errors.append(f"{len(invalid_go)} rows have invalid GO term format")

    # Load sample for comparison if available
    if sample.exists():
        try:
            sample_df = pd.read_csv(sample, sep="\t")
            sample_proteins = set(sample_df["Protein Id"].unique())
            sub_proteins = set(df["Protein Id"].unique())

            missing = sample_proteins - sub_proteins
            extra = sub_proteins - sample_proteins

            if missing:
                warnings.append(f"{len(missing)} proteins from sample are missing")
            if extra:
                warnings.append(f"{len(extra)} extra proteins not in sample")
        except Exception:
            pass

    # Display results
    table = Table(title="Submission Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total rows", f"{len(df):,}")
    table.add_row(
        "Unique proteins",
        f"{df['Protein Id'].nunique():,}" if "Protein Id" in df.columns else "N/A",
    )
    table.add_row(
        "Unique GO terms",
        f"{df['GO Term Id'].nunique():,}" if "GO Term Id" in df.columns else "N/A",
    )
    if "Prediction" in df.columns:
        table.add_row("Prediction range", f"[{pred_min:.4f}, {pred_max:.4f}]")
        table.add_row("Mean prediction", f"{df['Prediction'].mean():.4f}")
    console.print(table)

    if warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for w in warnings:
            console.print(f"  [yellow]⚠[/yellow] {w}")

    if errors:
        console.print("\n[red]Errors:[/red]")
        for err in errors:
            console.print(f"  [red]✗[/red] {err}")
        raise typer.Exit(1)
    else:
        console.print("\n[green]✓ Submission format is valid![/green]")


@app.command()
def baseline(
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output submission file path",
        ),
    ] = Path("submissions/frequency_baseline.tsv"),
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k",
            "-k",
            help="Number of GO terms per protein",
        ),
    ] = 500,
    use_ia: Annotated[
        bool,
        typer.Option(
            "--use-ia/--no-ia",
            help="Weight terms by Information Accretion",
        ),
    ] = True,
    propagate: Annotated[
        bool,
        typer.Option(
            "--propagate/--no-propagate",
            help="Propagate predictions to parent GO terms",
        ),
    ] = True,
) -> None:
    """Generate frequency baseline submission.

    Creates predictions based on GO term frequency in training data.
    This is a naive baseline where all test proteins get the same predictions.

    [bold]Example:[/bold]
        cafa6 baseline -o submissions/freq_baseline.tsv -k 500 --use-ia
    """
    import pandas as pd
    from Bio import SeqIO

    from cafa_6_protein.models.frequency import FrequencyBaseline

    console.print(Panel.fit("[bold blue]Frequency Baseline Generator[/bold blue]"))

    # Paths
    train_terms = Path("data/Train/train_terms.tsv")
    ia_file = Path("data/IA.tsv")
    test_fasta = Path("data/Test/testsuperset.fasta")
    ontology_file = Path("data/Train/go-basic.obo")

    # Validate files exist
    for path in [train_terms, test_fasta]:
        if not path.exists():
            console.print(f"[red]Error:[/red] Required file not found: {path}")
            raise typer.Exit(1)

    # Load training annotations
    console.print("[dim]Loading training annotations...[/dim]")
    train_df = pd.read_csv(train_terms, sep="\t")
    train_df = train_df.rename(columns={"EntryID": "protein_id", "term": "go_term"})
    console.print(
        f"  Loaded {len(train_df):,} annotations for {train_df['protein_id'].nunique():,} proteins"
    )

    # Load IA weights if requested
    ia_weights = None
    if use_ia and ia_file.exists():
        console.print("[dim]Loading IA weights...[/dim]")
        ia_df = pd.read_csv(ia_file, sep="\t", header=None, names=["go_term", "ia"])
        ia_weights = ia_df
        console.print(f"  Loaded weights for {len(ia_df):,} terms")

    # Get test protein IDs
    console.print("[dim]Loading test protein IDs...[/dim]")
    test_proteins = [record.id for record in SeqIO.parse(test_fasta, "fasta")]
    console.print(f"  Found {len(test_proteins):,} test proteins")

    # Fit baseline
    console.print(f"[dim]Fitting frequency baseline (top_k={top_k})...[/dim]")
    model = FrequencyBaseline(top_k=top_k)
    model.fit(train_df, ia_weights=ia_weights)
    console.print(f"  Computed scores for {len(model.term_scores):,} terms")

    # Load ontology for propagation if needed
    graph = None
    if propagate and ontology_file.exists():
        console.print("[dim]Loading GO ontology for propagation...[/dim]")
        from cafa_6_protein.data.ontology import load_go_ontology, propagate_term_scores

        graph = load_go_ontology(ontology_file)
        console.print(f"  Loaded {graph.number_of_nodes():,} GO terms")

    # Get the terms we'll predict (same for all proteins in frequency baseline)
    sorted_terms = model._get_sorted_terms()
    base_term_scores = dict(sorted_terms)

    # If propagating, compute propagated terms ONCE (since all proteins get same terms)
    if graph is not None:
        console.print("[dim]Pre-computing propagated terms...[/dim]")
        propagated_scores = propagate_term_scores(base_term_scores, graph)
        console.print(
            f"  {len(base_term_scores)} base terms → {len(propagated_scores)} after propagation"
        )
    else:
        propagated_scores = base_term_scores

    # Prepare final term list
    final_terms = list(propagated_scores.keys())
    final_scores = list(propagated_scores.values())

    # Stream write to file with progress
    console.print("[dim]Writing predictions to file...[/dim]")
    output.parent.mkdir(parents=True, exist_ok=True)

    batch_size = 10_000
    n_proteins = len(test_proteins)
    n_terms = len(final_terms)
    total_predictions = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed:,}/{task.total:,} proteins)"),
        console=console,
    ) as progress:
        task = progress.add_task("Writing predictions", total=n_proteins)

        with output.open("w") as f:
            # Write header (Kaggle format)
            f.write("Protein Id\tGO Term Id\tPrediction\n")

            # Process in batches
            for i in range(0, n_proteins, batch_size):
                batch_proteins = test_proteins[i : i + batch_size]

                # Write each protein's predictions
                for protein_id in batch_proteins:
                    for term, score in zip(final_terms, final_scores, strict=False):
                        f.write(f"{protein_id}\t{term}\t{score:.6f}\n")
                        total_predictions += 1

                progress.update(task, completed=min(i + batch_size, n_proteins))

    # Summary stats
    file_size_mb = output.stat().st_size / 1_000_000

    table = Table(title="Submission Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total predictions", f"{total_predictions:,}")
    table.add_row("Unique proteins", f"{n_proteins:,}")
    table.add_row("GO terms per protein", f"{n_terms:,}")
    table.add_row("Prediction range", f"[{min(final_scores):.4f}, {max(final_scores):.4f}]")
    table.add_row("Output file", str(output))
    table.add_row("File size", f"{file_size_mb:.1f} MB")
    console.print(table)

    console.print(f"\n[green]✓ Baseline submission saved to:[/green] {output}")


@app.command()
def cv(
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k",
            "-k",
            help="Number of GO terms per protein",
        ),
    ] = 100,
    use_ia: Annotated[
        bool,
        typer.Option(
            "--use-ia/--no-ia",
            help="Weight terms by Information Accretion",
        ),
    ] = True,
    propagate: Annotated[
        bool,
        typer.Option(
            "--propagate/--no-propagate",
            help="Propagate predictions to parent GO terms",
        ),
    ] = False,
    val_fraction: Annotated[
        float,
        typer.Option(
            "--val-fraction",
            "-v",
            help="Fraction of proteins to hold out for validation",
        ),
    ] = 0.1,
    max_val_proteins: Annotated[
        int,
        typer.Option(
            "--max-val",
            "-m",
            help="Maximum validation proteins (for faster testing)",
        ),
    ] = 5000,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            "-s",
            help="Random seed for reproducibility",
        ),
    ] = 42,
) -> None:
    """Run cross-validation to estimate local score.

    Splits training data, trains on subset, evaluates on held-out proteins.
    Reports F-max and S-min scores using CAFA-evaluator.

    [bold]Example:[/bold]
        cafa6 cv -k 100 --val-fraction 0.1 --max-val 5000
    """
    import tempfile

    import numpy as np
    import pandas as pd

    from cafa_6_protein.data.ontology import load_go_ontology, propagate_term_scores
    from cafa_6_protein.models.frequency import FrequencyBaseline

    console.print(Panel.fit("[bold blue]Cross-Validation[/bold blue]"))

    # Paths
    train_terms = Path("data/Train/train_terms.tsv")
    ia_file = Path("data/IA.tsv")
    ontology_file = Path("data/Train/go-basic.obo")

    # Validate files exist
    for path in [train_terms, ontology_file]:
        if not path.exists():
            console.print(f"[red]Error:[/red] Required file not found: {path}")
            raise typer.Exit(1)

    # Load training annotations
    console.print("[dim]Loading training annotations...[/dim]")
    train_df = pd.read_csv(train_terms, sep="\t")
    train_df = train_df.rename(columns={"EntryID": "protein_id", "term": "go_term"})
    all_proteins = train_df["protein_id"].unique()
    console.print(f"  Loaded {len(train_df):,} annotations for {len(all_proteins):,} proteins")

    # Split proteins
    console.print(
        f"[dim]Splitting proteins ({1-val_fraction:.0%} train / {val_fraction:.0%} val)...[/dim]"
    )
    np.random.seed(seed)
    np.random.shuffle(all_proteins)
    n_val = int(len(all_proteins) * val_fraction)

    # Limit validation size for faster testing
    if n_val > max_val_proteins:
        console.print(
            f"  [yellow]Limiting validation to {max_val_proteins:,} proteins (use --max-val to change)[/yellow]"
        )
        n_val = max_val_proteins

    val_proteins = set(all_proteins[:n_val])
    train_proteins = set(all_proteins[n_val:])
    console.print(f"  Train: {len(train_proteins):,} proteins, Val: {len(val_proteins):,} proteins")

    # Split annotations
    train_annot = train_df[train_df["protein_id"].isin(train_proteins)]
    val_annot = train_df[train_df["protein_id"].isin(val_proteins)]
    console.print(f"  Train annotations: {len(train_annot):,}, Val annotations: {len(val_annot):,}")

    # Load IA weights if requested
    ia_weights = None
    if use_ia and ia_file.exists():
        console.print("[dim]Loading IA weights...[/dim]")
        ia_df = pd.read_csv(ia_file, sep="\t", header=None, names=["go_term", "ia"])
        ia_weights = ia_df

    # Fit baseline on training split
    console.print(f"[dim]Fitting frequency baseline (top_k={top_k})...[/dim]")
    model = FrequencyBaseline(top_k=top_k)
    model.fit(train_annot, ia_weights=ia_weights)
    console.print(f"  Computed scores for {len(model.term_scores):,} terms")

    # Load ontology for propagation
    graph = None
    if propagate and ontology_file.exists():
        console.print("[dim]Loading GO ontology for propagation...[/dim]")
        graph = load_go_ontology(ontology_file)

    # Get prediction terms
    sorted_terms = model._get_sorted_terms()
    base_term_scores = dict(sorted_terms)

    if graph is not None:
        propagated_scores = propagate_term_scores(base_term_scores, graph)
        console.print(
            f"  {len(base_term_scores)} base terms → {len(propagated_scores)} after propagation"
        )
    else:
        propagated_scores = base_term_scores

    final_terms = list(propagated_scores.keys())
    final_scores = list(propagated_scores.values())

    # Create temp directory for evaluation
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        pred_dir = tmpdir / "predictions"
        pred_dir.mkdir()
        pred_file = pred_dir / "predictions.tsv"
        gt_file = tmpdir / "ground_truth.tsv"
        out_dir = tmpdir / "results"

        # Write predictions in batches
        console.print("[dim]Writing predictions to temp file...[/dim]")
        val_protein_list = list(val_proteins)
        n_val = len(val_protein_list)
        n_terms = len(final_terms)
        batch_size = 1000
        total_preds = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed:,}/{task.total:,})"),
            console=console,
        ) as progress:
            task = progress.add_task("Writing predictions", total=n_val)

            with pred_file.open("w") as f:
                for batch_start in range(0, n_val, batch_size):
                    batch_end = min(batch_start + batch_size, n_val)
                    batch_proteins = val_protein_list[batch_start:batch_end]

                    # Write batch
                    for protein_id in batch_proteins:
                        for term, score in zip(final_terms, final_scores, strict=False):
                            f.write(f"{protein_id}\t{term}\t{score:.6f}\n")
                            total_preds += 1

                    progress.update(task, completed=batch_end)

        console.print(
            f"  Wrote {total_preds:,} predictions ({n_val:,} proteins x {n_terms:,} terms)"
        )

        # Write ground truth (CAFA format: protein_id \t term)
        console.print("[dim]Writing ground truth...[/dim]")
        gt_count = 0
        with gt_file.open("w") as f:
            for _, row in val_annot.iterrows():
                f.write(f"{row['protein_id']}\t{row['go_term']}\n")
                gt_count += 1
        console.print(f"  Wrote {gt_count:,} ground truth annotations")

        # Run CAFA-evaluator with streaming output
        console.print("[dim]Running CAFA-evaluator...[/dim]")
        console.print(f"  Ontology: {ontology_file.resolve()}")
        console.print(f"  Predictions: {pred_file}")
        console.print(f"  Ground truth: {gt_file}")
        console.print(f"  Val proteins: {n_val:,}, Terms per protein: {n_terms:,}")
        console.print(f"  Total predictions: {total_preds:,}")

        cmd = [
            sys.executable,
            "-m",
            "cafaeval",
            str(ontology_file.resolve()),
            str(pred_dir.resolve()),
            str(gt_file.resolve()),
            "-out_dir",
            str(out_dir.resolve()),
            "-ia",
            str(ia_file.resolve()),
            "-prop",
            "max",
            "-norm",
            "cafa",
            "-threads",
            "0",
            "-log_level",
            "info",
        ]

        console.print(f"\n[dim]$ {' '.join(cmd[:6])} ...[/dim]")
        console.print("[dim]─" * 60 + "[/dim]")

        # Stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Read and display output line by line
        output_lines = []
        if process.stdout is not None:
            for line in process.stdout:
                line = line.rstrip()
                output_lines.append(line)
                # Show INFO lines, skip DEBUG spam
                if "[INFO]" in line or "[WARNING]" in line or "[ERROR]" in line:
                    console.print(f"  {line}")

        process.wait()
        console.print("[dim]─" * 60 + "[/dim]")

        if process.returncode != 0:
            console.print(f"[red]CAFA-evaluator failed (exit code {process.returncode})[/red]")
            console.print("[red]Last 20 lines of output:[/red]")
            for line in output_lines[-20:]:
                console.print(f"  {line}")
            raise typer.Exit(1)

        console.print("[green]✓ CAFA-evaluator completed[/green]")

        # Parse and display results
        console.print("\n[bold]Evaluation Results:[/bold]")

        # Look for best F-score results (the key metrics)
        best_f_file = out_dir / "evaluation_best_f.tsv"
        best_s_file = out_dir / "evaluation_best_s.tsv"

        if best_f_file.exists():
            console.print("\n[cyan]Best F-max scores (per ontology):[/cyan]")
            try:
                best_f_df = pd.read_csv(best_f_file, sep="\t")

                table = Table()
                table.add_column("Ontology", style="cyan")
                table.add_column("F-max", style="green")
                table.add_column("Threshold", style="dim")
                table.add_column("Precision", style="dim")
                table.add_column("Recall", style="dim")

                for _, row in best_f_df.iterrows():
                    table.add_row(
                        row.get("ns", "?"),
                        f"{row.get('f', 0):.4f}",
                        f"{row.get('tau', 0):.2f}",
                        f"{row.get('pr', 0):.4f}",
                        f"{row.get('rc', 0):.4f}",
                    )
                console.print(table)
            except Exception as e:
                console.print(f"  [red]Could not parse: {e}[/red]")

        if best_s_file.exists():
            console.print("\n[cyan]Best S-min scores (per ontology):[/cyan]")
            try:
                best_s_df = pd.read_csv(best_s_file, sep="\t")

                table = Table()
                table.add_column("Ontology", style="cyan")
                table.add_column("S-min", style="green")
                table.add_column("Threshold", style="dim")
                table.add_column("MI", style="dim")
                table.add_column("RU", style="dim")

                for _, row in best_s_df.iterrows():
                    table.add_row(
                        row.get("ns", "?"),
                        f"{row.get('s', 0):.4f}",
                        f"{row.get('tau', 0):.2f}",
                        f"{row.get('mi', 0):.4f}",
                        f"{row.get('ru', 0):.4f}",
                    )
                console.print(table)
            except Exception as e:
                console.print(f"  [red]Could not parse: {e}[/red]")

        if not best_f_file.exists() and not best_s_file.exists():
            console.print("[yellow]No result files found[/yellow]")
            console.print("[dim]Output directory contents:[/dim]")
            for path in out_dir.rglob("*"):
                if path.is_file():
                    console.print(f"  {path.relative_to(out_dir)} ({path.stat().st_size} bytes)")


@app.command(name="cv-retrieval")
def cv_retrieval(
    k: Annotated[
        int,
        typer.Option(
            "--k",
            "-k",
            help="Number of nearest neighbors",
        ),
    ] = 50,
    alpha: Annotated[
        float,
        typer.Option(
            "--alpha",
            help="Weight for annotation scores (vs literature)",
        ),
    ] = 1.0,
    use_literature: Annotated[
        bool,
        typer.Option(
            "--literature/--no-literature",
            help="Use literature enrichment",
        ),
    ] = False,
    val_fraction: Annotated[
        float,
        typer.Option(
            "--val-fraction",
            "-v",
            help="Fraction of proteins to hold out for validation",
        ),
    ] = 0.1,
    max_val_proteins: Annotated[
        int,
        typer.Option(
            "--max-val",
            "-m",
            help="Maximum validation proteins (for faster testing)",
        ),
    ] = 1000,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            "-s",
            help="Random seed for reproducibility",
        ),
    ] = 42,
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data",
            "-d",
            help="Data directory",
        ),
    ] = Path("data"),
) -> None:
    """Run cross-validation with retrieval-augmented predictor.

    Splits training data, trains on subset, evaluates on held-out proteins
    using k-NN retrieval with optional literature enrichment.

    [bold]Example:[/bold]
        cafa6 cv-retrieval -k 50 --alpha 0.7 --literature --max-val 1000
    """
    import tempfile

    import numpy as np
    import pandas as pd

    from cafa_6_protein.models import AggregationConfig, RetrievalAugmentedPredictor

    console.print(Panel.fit("[bold blue]Cross-Validation (Retrieval)[/bold blue]"))

    # Paths
    train_terms = data_dir / "Train" / "train_terms.tsv"
    ia_file = data_dir / "IA.tsv"
    ontology_file = data_dir / "Train" / "go-basic.obo"
    embeddings_file = data_dir / "embeddings.npy"
    protein_ids_file = data_dir / "embedding_protein_ids.txt"

    # Validate files exist
    for path in [train_terms, ontology_file, embeddings_file, protein_ids_file]:
        if not path.exists():
            console.print(f"[red]Error:[/red] Required file not found: {path}")
            raise typer.Exit(1)

    # Load training annotations
    console.print("[dim]Loading training annotations...[/dim]")
    train_df = pd.read_csv(train_terms, sep="\t")
    train_df = train_df.rename(columns={"EntryID": "protein_id", "term": "go_term"})
    all_proteins = train_df["protein_id"].unique()
    console.print(f"  Loaded {len(train_df):,} annotations for {len(all_proteins):,} proteins")

    # Load embeddings
    console.print("[dim]Loading embeddings...[/dim]")
    embeddings_array = np.load(embeddings_file)
    with protein_ids_file.open() as f:
        embedding_protein_ids = [line.strip() for line in f if line.strip()]
    pid_to_idx = {pid: i for i, pid in enumerate(embedding_protein_ids)}
    console.print(
        f"  Loaded {len(embedding_protein_ids):,} embeddings ({embeddings_array.shape[1]}-dim)"
    )

    # Filter to proteins with embeddings
    proteins_with_embeddings = set(all_proteins) & set(embedding_protein_ids)
    console.print(f"  Proteins with embeddings: {len(proteins_with_embeddings):,}")

    # Split proteins
    console.print(
        f"[dim]Splitting proteins ({1-val_fraction:.0%} train / {val_fraction:.0%} val)...[/dim]"
    )
    protein_list = list(proteins_with_embeddings)
    np.random.seed(seed)
    np.random.shuffle(protein_list)
    n_val = int(len(protein_list) * val_fraction)

    if n_val > max_val_proteins:
        console.print(f"  [yellow]Limiting validation to {max_val_proteins:,} proteins[/yellow]")
        n_val = max_val_proteins

    val_proteins = set(protein_list[:n_val])
    train_proteins = set(protein_list[n_val:])
    console.print(f"  Train: {len(train_proteins):,} proteins, Val: {len(val_proteins):,} proteins")

    # Split annotations
    train_annot = train_df[train_df["protein_id"].isin(train_proteins)]
    val_annot = train_df[train_df["protein_id"].isin(val_proteins)]
    console.print(f"  Train annotations: {len(train_annot):,}, Val annotations: {len(val_annot):,}")

    # Build training embeddings dict
    train_embeddings = {pid: embeddings_array[pid_to_idx[pid]] for pid in train_proteins}

    # Build validation embeddings dict
    val_embeddings = {pid: embeddings_array[pid_to_idx[pid]] for pid in val_proteins}

    # Load ontology
    console.print("[dim]Loading GO ontology...[/dim]")
    import obonet

    ontology = obonet.read_obo(str(ontology_file))
    console.print(f"  Loaded {len(ontology):,} GO terms")

    # Create predictor
    console.print(f"[dim]Creating retrieval predictor (k={k}, alpha={alpha})...[/dim]")
    config = AggregationConfig(alpha=alpha, propagate_ancestors=True)
    predictor = RetrievalAugmentedPredictor(k=k, config=config)
    predictor.set_ontology(ontology)

    # Set up literature enrichment if requested
    if use_literature:
        from cafa_6_protein.pubmed import AbstractCache, GOExtractor, PublicationCache

        publications_file = data_dir / "publications.parquet"
        abstracts_file = data_dir / "abstracts.db"

        if publications_file.exists() and abstracts_file.exists():
            pub_cache = PublicationCache(data_dir)
            abs_cache = AbstractCache(data_dir)
            go_extractor = GOExtractor.from_obo(ontology_file)
            predictor.set_literature_enrichment(pub_cache, abs_cache, go_extractor)
            console.print("  [green]Literature enrichment enabled[/green]")
        else:
            console.print("  [yellow]Literature caches not found, disabled[/yellow]")

    # Fit predictor on training data
    console.print("[dim]Fitting predictor on training data...[/dim]")
    predictor.fit(train_embeddings, train_annot)
    console.print(f"  Fitted with {len(train_embeddings):,} training proteins")

    # Generate predictions for validation set
    console.print("[dim]Generating predictions for validation set...[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console,
    ) as progress:
        task = progress.add_task("Predicting", total=len(val_proteins))

        predictions_list = []
        for pid, embedding in val_embeddings.items():
            preds = predictor.predict_one(pid, embedding)
            predictions_list.append(preds)
            progress.advance(task)

    predictions = pd.concat(predictions_list, ignore_index=True)
    console.print(
        f"  Generated {len(predictions):,} predictions for {len(val_proteins):,} proteins"
    )
    console.print(f"  Unique GO terms: {predictions['go_term'].nunique():,}")

    # Create temp directory for evaluation
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        pred_dir = tmpdir / "predictions"
        pred_dir.mkdir()
        pred_file = pred_dir / "predictions.tsv"
        gt_file = tmpdir / "ground_truth.tsv"
        out_dir = tmpdir / "results"

        # Write predictions
        console.print("[dim]Writing predictions to temp file...[/dim]")
        predictions.to_csv(
            pred_file,
            sep="\t",
            index=False,
            header=False,
            columns=["protein_id", "go_term", "score"],
        )
        console.print(f"  Wrote {len(predictions):,} predictions")

        # Write ground truth
        console.print("[dim]Writing ground truth...[/dim]")
        with gt_file.open("w") as f:
            for _, row in val_annot.iterrows():
                f.write(f"{row['protein_id']}\t{row['go_term']}\n")
        console.print(f"  Wrote {len(val_annot):,} ground truth annotations")

        # Run CAFA-evaluator
        console.print("[dim]Running CAFA-evaluator...[/dim]")

        cmd = [
            sys.executable,
            "-m",
            "cafaeval",
            str(ontology_file.resolve()),
            str(pred_dir.resolve()),
            str(gt_file.resolve()),
            "-out_dir",
            str(out_dir.resolve()),
            "-ia",
            str(ia_file.resolve()),
            "-prop",
            "max",
            "-norm",
            "cafa",
            "-threads",
            "0",
            "-log_level",
            "info",
        ]

        console.print("\n[dim]$ cafaeval ...[/dim]")
        console.print("[dim]─" * 60 + "[/dim]")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines = []
        if process.stdout is not None:
            for line in process.stdout:
                line = line.rstrip()
                output_lines.append(line)
                if "[INFO]" in line or "[WARNING]" in line or "[ERROR]" in line:
                    console.print(f"  {line}")

        process.wait()
        console.print("[dim]─" * 60 + "[/dim]")

        if process.returncode != 0:
            console.print(f"[red]CAFA-evaluator failed (exit code {process.returncode})[/red]")
            for line in output_lines[-20:]:
                console.print(f"  {line}")
            raise typer.Exit(1)

        console.print("[green]✓ CAFA-evaluator completed[/green]")

        # Parse and display results
        console.print("\n[bold]Evaluation Results:[/bold]")
        console.print(f"[dim]Config: k={k}, alpha={alpha}, literature={use_literature}[/dim]")

        best_f_file = out_dir / "evaluation_best_f.tsv"
        best_s_file = out_dir / "evaluation_best_s.tsv"

        if best_f_file.exists():
            console.print("\n[cyan]Best F-max scores (per ontology):[/cyan]")
            try:
                best_f_df = pd.read_csv(best_f_file, sep="\t")

                table = Table()
                table.add_column("Ontology", style="cyan")
                table.add_column("F-max", style="green")
                table.add_column("Threshold", style="dim")
                table.add_column("Precision", style="dim")
                table.add_column("Recall", style="dim")

                for _, row in best_f_df.iterrows():
                    table.add_row(
                        row.get("ns", "?"),
                        f"{row.get('f', 0):.4f}",
                        f"{row.get('tau', 0):.2f}",
                        f"{row.get('pr', 0):.4f}",
                        f"{row.get('rc', 0):.4f}",
                    )
                console.print(table)
            except Exception as e:
                console.print(f"  [red]Could not parse: {e}[/red]")

        if best_s_file.exists():
            console.print("\n[cyan]Best S-min scores (per ontology):[/cyan]")
            try:
                best_s_df = pd.read_csv(best_s_file, sep="\t")

                table = Table()
                table.add_column("Ontology", style="cyan")
                table.add_column("S-min", style="green")
                table.add_column("Threshold", style="dim")
                table.add_column("MI", style="dim")
                table.add_column("RU", style="dim")

                for _, row in best_s_df.iterrows():
                    table.add_row(
                        row.get("ns", "?"),
                        f"{row.get('s', 0):.4f}",
                        f"{row.get('tau', 0):.2f}",
                        f"{row.get('mi', 0):.4f}",
                        f"{row.get('ru', 0):.4f}",
                    )
                console.print(table)
            except Exception as e:
                console.print(f"  [red]Could not parse: {e}[/red]")


@app.command()
def pubmed(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data",
            "-d",
            help="Data directory containing training files",
        ),
    ] = Path("data"),
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="Proteins per API request batch (max 100)",
        ),
    ] = 100,
    max_proteins: Annotated[
        int | None,
        typer.Option(
            "--max-proteins",
            "-m",
            help="Maximum proteins to process (for testing)",
        ),
    ] = None,
    fetch_abstracts: Annotated[
        bool,
        typer.Option(
            "--fetch-abstracts/--no-abstracts",
            help="Also fetch abstracts from NCBI",
        ),
    ] = False,
    max_abstracts: Annotated[
        int,
        typer.Option(
            "--max-abstracts",
            help="Maximum abstracts to fetch per run",
        ),
    ] = 10000,
) -> None:
    """Fetch publication references for training proteins from UniProt/NCBI.

    Builds a local cache of protein -> PMID mappings and optionally fetches
    abstracts for GO term extraction.
    """
    import http.client
    import logging

    # Suppress noisy urllib3/requests debug logging that prints URLs
    http.client.HTTPConnection.debuglevel = 0
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("http.client").setLevel(logging.WARNING)

    from cafa_6_protein.pubmed import NCBIClient, UniProtClient

    console.print(Panel.fit("[bold blue]PubMed Mining Pipeline[/bold blue]"))

    # Load training protein IDs
    train_terms_path = data_dir / "Train" / "train_terms.tsv"
    if not train_terms_path.exists():
        console.print(f"[red]Error: {train_terms_path} not found[/red]")
        raise typer.Exit(1)

    import pandas as pd

    terms_df = pd.read_csv(train_terms_path, sep="\t")
    all_proteins = terms_df["EntryID"].unique().tolist()

    if max_proteins:
        all_proteins = all_proteins[:max_proteins]

    console.print(f"[dim]Training proteins:[/dim] {len(all_proteins):,}")

    # Initialize UniProt client
    uniprot = UniProtClient(data_dir, batch_size=batch_size)

    # Check what's already cached
    cached_count = sum(1 for p in all_proteins if uniprot.cache.has_protein(p))
    missing_count = len(all_proteins) - cached_count

    console.print(f"[dim]Already cached:[/dim] {cached_count:,}")
    console.print(f"[dim]To fetch:[/dim] {missing_count:,}")

    if missing_count > 0:
        console.print("\n[bold]Fetching publications from UniProt...[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching", total=missing_count)

            def update_progress(fetched: int, _total: int) -> None:
                progress.update(task, completed=fetched)

            uniprot.fetch_publications(all_proteins, progress_callback=update_progress)

        stats = uniprot.stats()
        console.print(
            f"\n[green]✓[/green] Cached {stats.total_proteins:,} proteins "
            f"with {stats.total_pmids:,} total PMIDs"
        )
    else:
        console.print("[green]✓[/green] All proteins already cached")
        stats = uniprot.stats()

    # Show cache stats
    table = Table(title="Publication Cache")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total proteins", f"{stats.total_proteins:,}")
    table.add_row("Total PMIDs", f"{stats.total_pmids:,}")
    table.add_row("Proteins with pubs", f"{stats.proteins_with_pubs:,}")
    console.print(table)

    # Optionally fetch abstracts
    if fetch_abstracts:
        console.print("\n[bold]Fetching abstracts from NCBI...[/bold]")

        ncbi = NCBIClient(data_dir)
        all_pmids = list(uniprot.cache.get_all_pmids())

        # Check what's already cached
        abstract_stats = ncbi.stats()
        cached_abstracts = abstract_stats.total_abstracts
        missing_pmids = ncbi.cache.get_missing_pmids(all_pmids)

        console.print(f"[dim]Total unique PMIDs:[/dim] {len(all_pmids):,}")
        console.print(f"[dim]Already cached:[/dim] {cached_abstracts:,}")
        console.print(f"[dim]To fetch:[/dim] {len(missing_pmids):,}")

        # Limit fetching
        pmids_to_fetch = missing_pmids[:max_abstracts]

        if pmids_to_fetch:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                console=console,
            ) as progress:
                task = progress.add_task("Fetching abstracts", total=len(pmids_to_fetch))

                def update_abstract_progress(fetched: int, _total: int) -> None:
                    progress.update(task, completed=fetched)

                ncbi.fetch_abstracts(pmids_to_fetch, progress_callback=update_abstract_progress)

            abstract_stats = ncbi.stats()
            console.print(f"\n[green]✓[/green] Cached {abstract_stats.total_abstracts:,} abstracts")
        else:
            console.print("[green]✓[/green] All abstracts already cached")

        # Show abstract cache stats
        table = Table(title="Abstract Cache")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total abstracts", f"{abstract_stats.total_abstracts:,}")
        table.add_row("Total processed", f"{abstract_stats.total_processed:,}")
        table.add_row("Not found", f"{abstract_stats.not_found:,}")
        console.print(table)


@app.command()
def embeddings(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data",
            "-d",
            help="Data directory containing training files",
        ),
    ] = Path("data"),
    max_proteins: Annotated[
        int | None,
        typer.Option(
            "--max-proteins",
            "-m",
            help="Maximum proteins to fetch embeddings for (for testing)",
        ),
    ] = None,
    build_index: Annotated[
        bool,
        typer.Option(
            "--build-index/--no-index",
            help="Build FAISS index after fetching",
        ),
    ] = True,
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset",
            help="Which dataset to fetch embeddings for: train, test, or both",
        ),
    ] = "train",
) -> None:
    """Fetch protein embeddings from UniProt and build FAISS index.

    Downloads pre-computed T5 embeddings for proteins and optionally
    builds a similarity search index.
    """
    import http.client
    import logging

    # Suppress noisy HTTP debug logging
    http.client.HTTPConnection.debuglevel = 0
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    from cafa_6_protein.embeddings import EmbeddingDownloader, EmbeddingIndex

    console.print(Panel.fit("[bold blue]Protein Embeddings Pipeline[/bold blue]"))

    # Load protein IDs
    import pandas as pd

    protein_ids = []

    if dataset in ("train", "both"):
        train_terms_path = data_dir / "Train" / "train_terms.tsv"
        if not train_terms_path.exists():
            console.print(f"[red]Error: {train_terms_path} not found[/red]")
            raise typer.Exit(1)
        terms_df = pd.read_csv(train_terms_path, sep="\t")
        train_proteins = terms_df["EntryID"].unique().tolist()
        protein_ids.extend(train_proteins)
        console.print(f"[dim]Training proteins:[/dim] {len(train_proteins):,}")

    if dataset in ("test", "both"):
        from cafa_6_protein.data import load_fasta_ids

        test_fasta = data_dir / "Test" / "testsuperset.fasta"
        if not test_fasta.exists():
            console.print(f"[red]Error: {test_fasta} not found[/red]")
            raise typer.Exit(1)
        test_proteins = load_fasta_ids(test_fasta)
        protein_ids.extend(test_proteins)
        console.print(f"[dim]Test proteins:[/dim] {len(test_proteins):,}")

    # Remove duplicates
    protein_ids = list(dict.fromkeys(protein_ids))

    if max_proteins:
        protein_ids = protein_ids[:max_proteins]

    console.print(f"[dim]Total proteins to process:[/dim] {len(protein_ids):,}")

    # Initialize downloader
    downloader = EmbeddingDownloader(data_dir)

    # Check what's already cached
    cached_count = sum(1 for p in protein_ids if downloader.has_embedding(p))
    missing_count = len(protein_ids) - cached_count

    console.print(f"[dim]Already cached:[/dim] {cached_count:,}")
    console.print(f"[dim]To extract:[/dim] {missing_count:,}")

    # Check if bulk file needs to be downloaded
    stats = downloader.stats()
    if not stats["bulk_file_exists"]:
        console.print("\n[bold]Downloading SwissProt embeddings (1.3 GB)...[/bold]")
        console.print("[dim]This is a one-time download[/dim]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total} MB)"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading", total=1300)

            def update_download_progress(downloaded_mb: int, _total_mb: int) -> None:
                progress.update(task, completed=downloaded_mb)

            downloader.download_bulk_embeddings(progress_callback=update_download_progress)

        console.print("[green]✓[/green] Download complete")

    if missing_count > 0:
        console.print("\n[bold]Extracting embeddings for target proteins...[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting", total=missing_count)

            def update_progress(fetched: int, _total: int) -> None:
                progress.update(task, completed=fetched)

            downloader.extract_embeddings(protein_ids, progress_callback=update_progress)

        stats = downloader.stats()
        console.print(
            f"\n[green]✓[/green] Cached {stats['total_proteins']:,} embeddings "
            f"({stats['cache_size_mb']:.1f} MB)"
        )
    else:
        console.print("[green]✓[/green] All embeddings already cached")
        stats = downloader.stats()

    # Show stats
    table = Table(title="Embedding Cache")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total proteins", f"{stats['total_proteins']:,}")
    table.add_row("Embedding dimension", f"{stats['embedding_dim']}")
    table.add_row("Cache size", f"{stats['cache_size_mb']:.1f} MB")
    console.print(table)

    # Build FAISS index
    if build_index and stats["total_proteins"] > 0:
        console.print("\n[bold]Building FAISS index...[/bold]")

        embeddings_array, embed_protein_ids = downloader.get_all_embeddings()

        index = EmbeddingIndex(embedding_dim=int(stats["embedding_dim"]))
        index.build(embeddings_array, embed_protein_ids, normalize=True)
        index.save(data_dir / "embedding_index")

        console.print(f"[green]✓[/green] Built index with {len(index):,} proteins")
        console.print(f"[dim]Index saved to:[/dim] {data_dir / 'embedding_index'}")


@app.command(name="predict")
def predict_cmd(
    output: Annotated[
        Path,
        typer.Argument(
            help="Output file for predictions (TSV format)",
        ),
    ],
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data",
            "-d",
            help="Data directory containing embeddings and training data",
        ),
    ] = Path("data"),
    k: Annotated[
        int,
        typer.Option(
            "--k",
            "-k",
            help="Number of nearest neighbors",
        ),
    ] = 50,
    alpha: Annotated[
        float,
        typer.Option(
            "--alpha",
            help="Weight for annotation scores (vs literature)",
        ),
    ] = 1.0,
    use_literature: Annotated[
        bool,
        typer.Option(
            "--literature/--no-literature",
            help="Use literature enrichment",
        ),
    ] = False,
    max_proteins: Annotated[
        int | None,
        typer.Option(
            "--max-proteins",
            "-m",
            help="Maximum proteins to predict (for testing)",
        ),
    ] = None,
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset",
            help="Dataset to predict: 'train' for CV or 'test' for submission",
        ),
    ] = "test",
) -> None:
    """Generate predictions using retrieval-augmented method.

    Uses k-NN over protein embeddings to predict GO terms.
    """
    import numpy as np

    from cafa_6_protein.models.retrieval import load_retrieval_predictor_from_cache

    console.print(Panel.fit("[bold blue]Retrieval-Augmented Prediction[/bold blue]"))

    # Load predictor
    console.print("[dim]Loading predictor...[/dim]")
    try:
        predictor = load_retrieval_predictor_from_cache(
            data_dir=data_dir,
            k=k,
            alpha=alpha,
            use_literature=use_literature,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    params = predictor.get_params()
    console.print(f"[dim]Predictor config:[/dim] k={params.k}, alpha={params.alpha}")

    # Load test embeddings
    console.print("[dim]Loading test embeddings...[/dim]")

    if dataset == "test":
        # Load test protein IDs
        from cafa_6_protein.data import load_fasta_ids

        test_fasta = data_dir / "Test" / "testsuperset.fasta"
        if not test_fasta.exists():
            console.print(f"[red]Error: {test_fasta} not found[/red]")
            raise typer.Exit(1)
        protein_ids = load_fasta_ids(test_fasta)
    else:
        # Use training proteins (for CV)
        import pandas as pd

        train_terms = data_dir / "Train" / "train_terms.tsv"
        terms_df = pd.read_csv(train_terms, sep="\t")
        protein_ids = terms_df["EntryID"].unique().tolist()

    if max_proteins:
        protein_ids = protein_ids[:max_proteins]

    console.print(f"[dim]Proteins to predict:[/dim] {len(protein_ids):,}")

    # Load embeddings for test proteins
    embeddings_file = data_dir / "embeddings.npy"
    protein_ids_file = data_dir / "embedding_protein_ids.txt"

    embeddings_array = np.load(embeddings_file)
    with protein_ids_file.open() as f:
        cached_protein_ids = [line.strip() for line in f if line.strip()]

    pid_to_idx = {pid: i for i, pid in enumerate(cached_protein_ids)}

    # Filter to proteins with embeddings
    test_embeddings = {}
    missing = 0
    for pid in protein_ids:
        if pid in pid_to_idx:
            test_embeddings[pid] = embeddings_array[pid_to_idx[pid]]
        else:
            missing += 1

    if missing > 0:
        console.print(f"[yellow]Warning: {missing:,} proteins missing embeddings[/yellow]")

    console.print(f"[dim]Proteins with embeddings:[/dim] {len(test_embeddings):,}")

    # Generate predictions
    console.print("\n[bold]Generating predictions...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Predicting", total=len(test_embeddings))

        predictions_list = []
        for pid, embedding in test_embeddings.items():
            preds = predictor.predict_one(pid, embedding)
            predictions_list.append(preds)
            progress.advance(task)

    import pandas as pd

    predictions = pd.concat(predictions_list, ignore_index=True)

    console.print(f"\n[green]✓[/green] Generated {len(predictions):,} predictions")
    console.print(f"[dim]Unique proteins:[/dim] {predictions['protein_id'].nunique():,}")
    console.print(f"[dim]Unique GO terms:[/dim] {predictions['go_term'].nunique():,}")

    # Save predictions
    predictions.to_csv(output, sep="\t", index=False, header=False)
    console.print(f"[green]✓[/green] Saved to {output}")


@app.command()
def info() -> None:
    """Display project and data information."""
    console.print(Panel.fit("[bold blue]CAFA-6 Project Info[/bold blue]"))

    # Check data files
    data_dir = Path("data")
    train_dir = data_dir / "Train"
    test_dir = data_dir / "Test"

    table = Table(title="Data Files")
    table.add_column("File", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Size", style="dim")

    files_to_check = [
        ("IA weights", data_dir / "IA.tsv"),
        ("Sample submission", data_dir / "sample_submission.tsv"),
        ("GO ontology", train_dir / "go-basic.obo"),
        ("Train sequences", train_dir / "train_sequences.fasta"),
        ("Train terms", train_dir / "train_terms.tsv"),
        ("Train taxonomy", train_dir / "train_taxonomy.tsv"),
        ("Test sequences", test_dir / "testsuperset.fasta"),
        ("Test taxon list", test_dir / "testsuperset-taxon-list.tsv"),
    ]

    for name, path in files_to_check:
        if path.exists():
            size = path.stat().st_size
            if size > 1_000_000:
                size_str = f"{size / 1_000_000:.1f} MB"
            elif size > 1_000:
                size_str = f"{size / 1_000:.1f} KB"
            else:
                size_str = f"{size} B"
            table.add_row(name, "[green]✓[/green]", size_str)
        else:
            table.add_row(name, "[red]✗ missing[/red]", "-")

    console.print(table)

    # Show quick stats if files exist
    if (train_dir / "train_terms.tsv").exists():
        import pandas as pd

        terms_df = pd.read_csv(train_dir / "train_terms.tsv", sep="\t")
        console.print(
            f"\n[dim]Training:[/dim] {terms_df['EntryID'].nunique():,} proteins, "
            f"{terms_df['term'].nunique():,} GO terms, "
            f"{len(terms_df):,} annotations"
        )

    if (data_dir / "IA.tsv").exists():
        import pandas as pd

        ia_df = pd.read_csv(data_dir / "IA.tsv", sep="\t", header=None)
        console.print(f"[dim]IA weights:[/dim] {len(ia_df):,} terms")


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
