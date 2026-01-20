"""CAFA-6 Protein Function Prediction CLI.

Minimal CLI for local evaluation and pipeline execution.
"""

import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
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
            "--ontology", "-o",
            help="GO ontology file (OBO format)",
        ),
    ] = Path("data/Train/go-basic.obo"),
    ia_file: Annotated[
        Optional[Path],
        typer.Option(
            "--ia", "-i",
            help="Information Accretion weights file",
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output", "-O",
            help="Output directory for results",
        ),
    ] = Path("results"),
    threads: Annotated[
        int,
        typer.Option(
            "--threads", "-t",
            help="Number of threads (0 = all available)",
        ),
    ] = 0,
    propagate: Annotated[
        str,
        typer.Option(
            "--propagate", "-p",
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
    
    console.print(Panel.fit(
        "[bold blue]CAFA-6 Local Evaluation[/bold blue]",
        subtitle="Using CAFA-evaluator"
    ))
    
    # Build command
    cmd = [
        sys.executable, "-m", "cafaeval",
        str(ontology),
        str(predictions),
        str(ground_truth),
        "-out_dir", str(output_dir),
        "-threads", str(threads),
        "-prop", propagate,
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
        result = subprocess.run(cmd, check=True, capture_output=False)
        console.print(f"\n[green]✓ Evaluation complete![/green] Results saved to: {output_dir}")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Evaluation failed with exit code {e.returncode}[/red]")
        raise typer.Exit(1)
    except FileNotFoundError:
        console.print("[red]✗ cafaeval not found. Install with:[/red]")
        console.print("  uv pip install git+https://github.com/BioComputingUP/CAFA-evaluator.git")
        raise typer.Exit(1)


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
            "--sample", "-s",
            help="Sample submission file for format reference",
        ),
    ] = Path("data/sample_submission.tsv"),
) -> None:
    """Validate submission file format.
    
    Checks that the submission file has the correct format for Kaggle.
    """
    import pandas as pd
    
    console.print(Panel.fit(
        "[bold blue]Submission Validation[/bold blue]"
    ))
    
    errors = []
    warnings = []
    
    try:
        df = pd.read_csv(submission, sep="\t" if submission.suffix == ".tsv" else ",")
    except Exception as e:
        console.print(f"[red]✗ Failed to read file:[/red] {e}")
        raise typer.Exit(1)
    
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
    table.add_row("Unique proteins", f"{df['Protein Id'].nunique():,}" if "Protein Id" in df.columns else "N/A")
    table.add_row("Unique GO terms", f"{df['GO Term Id'].nunique():,}" if "GO Term Id" in df.columns else "N/A")
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
        for e in errors:
            console.print(f"  [red]✗[/red] {e}")
        raise typer.Exit(1)
    else:
        console.print("\n[green]✓ Submission format is valid![/green]")


@app.command()
def baseline(
    output: Annotated[
        Path,
        typer.Option(
            "--output", "-o",
            help="Output submission file path",
        ),
    ] = Path("submissions/frequency_baseline.tsv"),
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k", "-k",
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
    
    console.print(Panel.fit(
        "[bold blue]Frequency Baseline Generator[/bold blue]"
    ))
    
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
    console.print(f"  Loaded {len(train_df):,} annotations for {train_df['protein_id'].nunique():,} proteins")
    
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
        console.print(f"  {len(base_term_scores)} base terms → {len(propagated_scores)} after propagation")
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
        
        with open(output, "w") as f:
            # Write header (Kaggle format)
            f.write("Protein Id\tGO Term Id\tPrediction\n")
            
            # Process in batches
            for i in range(0, n_proteins, batch_size):
                batch_proteins = test_proteins[i:i + batch_size]
                
                # Write each protein's predictions
                for protein_id in batch_proteins:
                    for term, score in zip(final_terms, final_scores):
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
            "--top-k", "-k",
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
            "--val-fraction", "-v",
            help="Fraction of proteins to hold out for validation",
        ),
    ] = 0.1,
    max_val_proteins: Annotated[
        int,
        typer.Option(
            "--max-val", "-m",
            help="Maximum validation proteins (for faster testing)",
        ),
    ] = 5000,
    seed: Annotated[
        int,
        typer.Option(
            "--seed", "-s",
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
    import shutil
    import tempfile
    
    import numpy as np
    import pandas as pd
    
    from cafa_6_protein.models.frequency import FrequencyBaseline
    from cafa_6_protein.data.ontology import load_go_ontology, propagate_term_scores
    
    console.print(Panel.fit(
        "[bold blue]Cross-Validation[/bold blue]"
    ))
    
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
    console.print(f"[dim]Splitting proteins ({1-val_fraction:.0%} train / {val_fraction:.0%} val)...[/dim]")
    np.random.seed(seed)
    np.random.shuffle(all_proteins)
    n_val = int(len(all_proteins) * val_fraction)
    
    # Limit validation size for faster testing
    if n_val > max_val_proteins:
        console.print(f"  [yellow]Limiting validation to {max_val_proteins:,} proteins (use --max-val to change)[/yellow]")
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
        console.print(f"  {len(base_term_scores)} base terms → {len(propagated_scores)} after propagation")
    else:
        propagated_scores = base_term_scores
    
    final_terms = list(propagated_scores.keys())
    final_scores = list(propagated_scores.values())
    
    # Create temp directory for evaluation
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
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
            
            with open(pred_file, "w") as f:
                for batch_start in range(0, n_val, batch_size):
                    batch_end = min(batch_start + batch_size, n_val)
                    batch_proteins = val_protein_list[batch_start:batch_end]
                    
                    # Write batch
                    for protein_id in batch_proteins:
                        for term, score in zip(final_terms, final_scores):
                            f.write(f"{protein_id}\t{term}\t{score:.6f}\n")
                            total_preds += 1
                    
                    progress.update(task, completed=batch_end)
        
        console.print(f"  Wrote {total_preds:,} predictions ({n_val:,} proteins × {n_terms:,} terms)")
        
        # Write ground truth (CAFA format: protein_id \t term)
        console.print("[dim]Writing ground truth...[/dim]")
        gt_count = 0
        with open(gt_file, "w") as f:
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
            sys.executable, "-m", "cafaeval",
            str(ontology_file.resolve()),
            str(pred_dir.resolve()),
            str(gt_file.resolve()),
            "-out_dir", str(out_dir.resolve()),
            "-ia", str(ia_file.resolve()),
            "-prop", "max",
            "-norm", "cafa",
            "-threads", "0",
            "-log_level", "info",
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
            console.print(f"[dim]Output directory contents:[/dim]")
            for f in out_dir.rglob("*"):
                if f.is_file():
                    console.print(f"  {f.relative_to(out_dir)} ({f.stat().st_size} bytes)")


@app.command()
def info() -> None:
    """Display project and data information."""
    console.print(Panel.fit(
        "[bold blue]CAFA-6 Project Info[/bold blue]"
    ))
    
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
        console.print(f"\n[dim]Training:[/dim] {terms_df['EntryID'].nunique():,} proteins, "
                      f"{terms_df['term'].nunique():,} GO terms, "
                      f"{len(terms_df):,} annotations")
    
    if (data_dir / "IA.tsv").exists():
        import pandas as pd
        ia_df = pd.read_csv(data_dir / "IA.tsv", sep="\t", header=None)
        console.print(f"[dim]IA weights:[/dim] {len(ia_df):,} terms")


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
