"""Typer + rich CLI for the Austin Restaurant Sales & Guest Count Forecasting Engine."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from src import config, exporter, feedback_loop, forecaster

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = typer.Typer(
    name="restaurant-forecaster",
    help="Austin restaurant sales & guest count forecasting CLI.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def train(
    data_path: Path = typer.Option(..., "--data", exists=True, help="CSV of historical sales/guest_count data."),
    no_weather: bool = typer.Option(False, "--no-weather", help="Skip fetching live weather data."),
):
    """Train (or retrain from scratch) the sales and guest_count ensemble models."""
    with console.status("Training ensemble models..."):
        models = forecaster.train_all_targets(data_path, fetch_weather=not no_weather)

    table = Table(title="Training complete")
    table.add_column("Target")
    table.add_column("Model version")
    table.add_column("Training rows")
    for target, model in models.items():
        table.add_row(target, model.metadata.model_version, str(model.metadata.training_rows))
    console.print(table)


@app.command()
def forecast(
    start: str = typer.Option(..., "--start", help="Start date YYYY-MM-DD."),
    end: str = typer.Option(..., "--end", help="End date YYYY-MM-DD."),
    export: Optional[Path] = typer.Option(None, "--export", help="Export path (.csv or .json)."),
    no_weather: bool = typer.Option(False, "--no-weather", help="Skip fetching live weather data."),
):
    """Generate daily sales & guest count projections for a date range."""
    start_date = _parse_date(start)
    end_date = _parse_date(end)
    if end_date < start_date:
        raise typer.BadParameter("--end must not be before --start")

    try:
        with console.status("Generating forecast..."):
            results = forecaster.generate_forecast(start_date, end_date, fetch_weather=not no_weather)
    except RuntimeError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    table = Table(title=f"Forecast {start_date} -> {end_date}")
    table.add_column("Date")
    table.add_column("Sales ($)", justify="right")
    table.add_column("Sales range", justify="right")
    table.add_column("Guests", justify="right")
    table.add_column("Guests range", justify="right")
    table.add_column("Key drivers")

    for f in results:
        driver_str = ", ".join(d["description"] for d in f.drivers) if f.drivers else "-"
        table.add_row(
            f.date,
            f"${f.projected_sales:,.0f}",
            f"${f.sales_low:,.0f} - ${f.sales_high:,.0f}",
            str(f.projected_guests),
            f"{f.guests_low} - {f.guests_high}",
            driver_str,
        )
    console.print(table)

    if export:
        _export_forecast(results, export)
        console.print(f"[green]Exported forecast to {export}[/green]")


@app.command("log-actuals")
def log_actuals(
    date_str: str = typer.Option(..., "--date", help="Date YYYY-MM-DD."),
    sales: float = typer.Option(..., "--sales", help="Actual sales for the day."),
    guests: int = typer.Option(..., "--guests", help="Actual guest count for the day."),
    data_path: Optional[Path] = typer.Option(
        None, "--data", help="Historical CSV, used only if a drift-triggered retrain fires."
    ),
    no_weather: bool = typer.Option(False, "--no-weather", help="Skip fetching live weather data during any retrain."),
):
    """Log actual sales/guests for a date and update the adaptive feedback loop."""
    _parse_date(date_str)
    results = feedback_loop.log_actuals(date_str, sales, guests)

    table = Table(title=f"Actuals logged for {date_str}")
    table.add_column("Target")
    table.add_column("Actual", justify="right")
    table.add_column("Predicted", justify="right")
    table.add_column("Abs error", justify="right")
    table.add_column("% error", justify="right")
    table.add_column("Bias")

    for r in results:
        if r.had_forecast:
            bias_label = "over" if r.bias > 0 else "under" if r.bias < 0 else "even"
            table.add_row(
                r.target, f"{r.actual:,.2f}", f"{r.predicted:,.2f}",
                f"{r.abs_error:,.2f}", f"{r.pct_error * 100:,.1f}%", bias_label,
            )
        else:
            table.add_row(r.target, f"{r.actual:,.2f}", "-", "-", "-", "no forecast on file")
    console.print(table)

    for target in feedback_loop.TARGETS:
        report = feedback_loop.detect_drift(target)
        if report.drift_detected:
            console.print(
                f"[yellow]Drift detected on {target}[/yellow]: mean bias "
                f"{report.mean_bias_pct * 100:.1f}% over trailing {report.lookback_days} days."
            )
            if data_path is None:
                console.print(
                    "[yellow]  Skipping auto-retrain: pass --data <historical.csv> to enable it.[/yellow]"
                )
                continue
            with console.status(f"Retraining {target} model due to drift..."):
                feedback_loop.retrain_model(
                    data_path, target, fetch_weather=not no_weather,
                    trigger_reason=f"drift(bias={report.mean_bias_pct:.3f})",
                )
            console.print(f"[green]  Retrained {target} model.[/green]")


@app.command("push-projections")
def push_projections(
    endpoint: str = typer.Option(..., "--endpoint", help="Downstream webhook/REST URL."),
    api_key: str = typer.Option(
        ..., "--api-key", envvar=config.API_KEY_ENV_VAR, help="API key (or set RESTAURANT_FORECASTER_API_KEY)."
    ),
    start: str = typer.Option(..., "--start", help="Start date YYYY-MM-DD."),
    end: str = typer.Option(..., "--end", help="End date YYYY-MM-DD."),
    no_weather: bool = typer.Option(False, "--no-weather", help="Skip fetching live weather data."),
):
    """Generate a forecast and push it to a downstream endpoint via webhook/REST."""
    start_date = _parse_date(start)
    end_date = _parse_date(end)

    try:
        with console.status("Generating forecast..."):
            results = forecaster.generate_forecast(start_date, end_date, fetch_weather=not no_weather)
    except RuntimeError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    daily_projections = [f.as_dict() for f in results]
    accuracy_metrics = {
        "sales": feedback_loop.trailing_accuracy_summary("sales"),
        "guest_count": feedback_loop.trailing_accuracy_summary("guest_count"),
    }

    with console.status(f"Pushing to {endpoint}..."):
        result = exporter.push_projections(endpoint, api_key, daily_projections, accuracy_metrics)

    if result.success:
        console.print(f"[green]Pushed {len(daily_projections)} day(s) to {endpoint} (HTTP {result.status_code}).[/green]")
    else:
        console.print(f"[bold red]Push failed:[/bold red] {result.error}")
        raise typer.Exit(code=1)


@app.command()
def accuracy(
    target: str = typer.Option("sales", "--target", help="'sales' or 'guest_count'."),
):
    """Show trailing 7/30/90-day accuracy metrics for a target."""
    summary = feedback_loop.trailing_accuracy_summary(target)
    table = Table(title=f"Accuracy summary: {target}")
    table.add_column("Window")
    table.add_column("WAPE", justify="right")
    table.add_column("MAPE", justify="right")
    table.add_column("Bias %", justify="right")
    table.add_column("Accuracy %", justify="right")
    table.add_column("N", justify="right")
    for window, metrics in summary.items():
        table.add_row(
            window.replace("trailing_", "").replace("d", "d"),
            f"{metrics['wape'] * 100:.1f}%",
            f"{metrics['mape'] * 100:.1f}%",
            f"{metrics['bias_pct'] * 100:.1f}%",
            f"{metrics['accuracy_pct']:.1f}%",
            str(metrics["n_observations"]),
        )
    console.print(table)


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise typer.BadParameter(f"'{value}' is not a valid YYYY-MM-DD date") from exc


def _export_forecast(results, export_path: Path) -> None:
    export_path = Path(export_path)
    if export_path.suffix.lower() == ".json":
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump([r.as_dict() for r in results], f, indent=2)
    elif export_path.suffix.lower() == ".csv":
        forecaster.forecasts_to_dataframe(results).to_csv(export_path, index=False)
    else:
        raise typer.BadParameter("--export must end in .csv or .json")


if __name__ == "__main__":
    app()
