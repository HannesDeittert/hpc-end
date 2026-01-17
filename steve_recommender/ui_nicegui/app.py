"""NiceGUI UI for browsing devices, launching training, and running evaluation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List

from nicegui import ui

from steve_recommender.domain import TrainingConfig
from steve_recommender.services import evaluation_service, library_service, run_service, training_service


def _apply_theme() -> None:
    ui.add_head_html(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
        :root {
            --bg: #f3efe6;
            --surface: #fffaf2;
            --ink: #1d1d1d;
            --muted: #6b6b6b;
            --accent: #0f766e;
            --accent-2: #d97706;
            --border: #e3dccf;
        }
        html, body, .q-layout, .q-page {
            background: radial-gradient(1200px 600px at 10% 0%, #f9f2dd, var(--bg));
            font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
            color: var(--ink);
        }
        .app-shell {
            max-width: 1200px;
            margin: 0 auto;
        }
        .app-title {
            font-size: 30px;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        .muted {
            color: var(--muted);
        }
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
        }
        .pill {
            background: #e8f4f2;
            border-radius: 999px;
            padding: 2px 10px;
            font-size: 12px;
        }
        </style>
        """
    )


@ui.refreshable
def _library_panel() -> None:
    models = library_service.list_models()
    if not models:
        ui.label("No models found under data/.").classes("muted")
        return
    for model in models:
        with ui.card().classes("card w-full"):
            with ui.row().classes("items-center justify-between"):
                ui.label(model.name).classes("text-lg font-semibold")
                ui.label(model.description or "No description").classes("muted")
            wires = library_service.list_wires(model.name)
            if not wires:
                ui.label("No wires for this model.").classes("muted")
                continue
            for wire in wires:
                agents = library_service.list_agents(model.name, wire.name)
                with ui.row().classes("items-center gap-4"):
                    ui.label(f"{wire.name}").classes("font-medium")
                    ui.label(f"{len(agents)} agents").classes("pill")


@ui.refreshable
def _runs_panel() -> None:
    runs = run_service.list_training_runs()
    if not runs:
        ui.label("No training runs found.").classes("muted")
        return
    for run in runs:
        with ui.card().classes("card w-full"):
            ui.label(run.name).classes("font-semibold")
            if run.log_path:
                ui.label(f"log: {run.log_path}").classes("muted text-xs")
            if run.results_csv:
                ui.label(f"results: {run.results_csv}").classes("muted text-xs")
            if run.log_path:
                with ui.expansion("Log tail").classes("mt-2"):
                    ui.code(run_service.tail_text(run.log_path, max_lines=60)).classes(
                        "w-full"
                    )


def _training_panel() -> None:
    models = [m.name for m in library_service.list_models()]
    model_select = ui.select(models, label="Model").classes("w-64")
    wire_select = ui.select([], label="Wire").classes("w-64")
    run_name = ui.input("Run name", value="paper_run").classes("w-64")
    device = ui.select(["cpu", "cuda", "cuda:0", "cuda:1"], value="cpu", label="Trainer device").classes("w-64")
    n_workers = ui.number("Workers", value=2, min=1, step=1).classes("w-32")

    def _refresh_wires() -> None:
        if not model_select.value:
            wire_select.options = []
            return
        wire_select.options = [w.name for w in library_service.list_wires(model_select.value)]

    model_select.on_value_change(lambda _: _refresh_wires())

    if models:
        model_select.value = models[0]
        _refresh_wires()
        if wire_select.options:
            wire_select.value = wire_select.options[0]

    async def _start_training() -> None:
        if not model_select.value or not wire_select.value:
            ui.notify("Select model and wire first.", type="warning")
            return
        cfg = TrainingConfig(
            tool=wire_select.value,
            model=model_select.value,
            name=run_name.value.strip() or "paper_run",
            trainer_device=device.value,
            n_worker=int(n_workers.value or 1),
        )
        ui.notify("Training started. This may take a while...", type="info")
        try:
            run_dir = await asyncio.to_thread(training_service.run_training, cfg)
            ui.notify(f"Training done: {run_dir}", type="positive")
            _runs_panel.refresh()
            _library_panel.refresh()
        except Exception as exc:  # noqa: BLE001
            ui.notify(f"Training failed: {exc}", type="negative")

    ui.button("Start training", on_click=_start_training).props("unelevated").classes("bg-[var(--accent)] text-white")


def _evaluation_panel() -> None:
    config_path = ui.input("Evaluation config (YAML)", value="docs/eval_example.yml").classes("w-96")

    async def _run_eval() -> None:
        path = Path(config_path.value).expanduser()
        if not path.exists():
            ui.notify(f"Config not found: {path}", type="warning")
            return
        cfg = evaluation_service.load_evaluation_config(path)
        ui.notify("Evaluation started...", type="info")
        try:
            run_dir = await asyncio.to_thread(evaluation_service.run_evaluation, cfg)
            ui.notify(f"Evaluation done: {run_dir}", type="positive")
        except Exception as exc:  # noqa: BLE001
            ui.notify(f"Evaluation failed: {exc}", type="negative")

    ui.button("Run evaluation", on_click=_run_eval).props("unelevated").classes("bg-[var(--accent-2)] text-white")


def main() -> None:
    _apply_theme()

    with ui.column().classes("app-shell w-full gap-6"):
        with ui.row().classes("items-baseline justify-between w-full pt-6"):
            ui.label("stEVE Recommender Studio").classes("app-title")
            ui.label("Library, training, and evaluation").classes("muted")

        tabs = ui.tabs().classes("w-full")
        with tabs:
            ui.tab("library", label="Library")
            ui.tab("training", label="Training")
            ui.tab("evaluation", label="Evaluation")
            ui.tab("runs", label="Runs")

        with ui.tab_panels(tabs, value="library").classes("w-full"):
            with ui.tab_panel("library"):
                ui.button("Refresh", on_click=_library_panel.refresh).props("flat").classes("mb-2")
                _library_panel()
            with ui.tab_panel("training"):
                _training_panel()
            with ui.tab_panel("evaluation"):
                _evaluation_panel()
            with ui.tab_panel("runs"):
                ui.button("Refresh", on_click=_runs_panel.refresh).props("flat").classes("mb-2")
                _runs_panel()

    ui.run(title="stEVE Recommender Studio", reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
