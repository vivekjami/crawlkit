# python/crawlkit/cli.py
"""crawlkit CLI -- powered by Typer."""

from __future__ import annotations

import asyncio
from typing import cast

import typer

from .client import CrawlKit
from .events import (
    BudgetWarningEvent,
    CompletedEvent,
    FailedEvent,
    PageEvent,
)

app = typer.Typer(
    name="crawlkit",
    help="Production reliability layer for async web crawling pipelines.",
    no_args_is_help=True,
)


@app.command()  # type: ignore[untyped-decorator]
def crawl(
    url: str = typer.Argument(..., help="Root URL to crawl"),
    limit: int = typer.Option(100, "--limit", "-l", help="Max pages"),
    max_depth: int = typer.Option(3, "--max-depth", "-d", help="Max link depth"),
    max_credits: int | None = typer.Option(None, "--max-credits"),
    warn_at: int | None = typer.Option(None, "--warn-at"),
    checkpoint_path: str | None = typer.Option(None, "--checkpoint"),
    api_key: str | None = typer.Option(None, "--api-key", envvar="FIRECRAWL_API_KEY"),
) -> None:
    """Crawl a URL and stream events to stdout."""
    from .budget import BudgetConfig
    from .checkpoint import CheckpointConfig

    kit = CrawlKit(
        api_key=api_key,
        budget=BudgetConfig(max_credits=max_credits, warn_at=warn_at) if max_credits else None,
        checkpoint=CheckpointConfig(path=checkpoint_path) if checkpoint_path else None,
    )

    async def _run() -> None:
        typer.echo(f"Starting crawl: {url}")
        async for event in kit.crawl(url, limit=limit, max_depth=max_depth):
            if event.type == "page":
                page = cast(PageEvent, event)
                src = page.page.get("metadata", {}).get("sourceURL", "?")
                md_len = len(page.page.get("markdown", ""))
                fp8 = page.fingerprint[:8] if page.fingerprint else "--------"
                typer.echo(f"  + {src} ({md_len} chars, fp:{fp8}...)")
            elif event.type == "budget_warning":
                warning = cast(BudgetWarningEvent, event)
                typer.secho(
                    f"  ! Budget: {warning.credits_used}/{warning.budget} credits",
                    fg=typer.colors.YELLOW,
                )
            elif event.type == "completed":
                completed = cast(CompletedEvent, event)
                typer.secho(
                    f"  Done: {completed.total_pages} pages, {completed.credits_used} credits",
                    fg=typer.colors.GREEN,
                )
            elif event.type == "failed":
                failed = cast(FailedEvent, event)
                typer.secho(f"  Failed: {failed.error}", fg=typer.colors.RED)
                raise typer.Exit(code=1)

    asyncio.run(_run())


@app.command()  # type: ignore[untyped-decorator]
def version() -> None:
    """Print version."""
    from crawlkit import __version__

    typer.echo(f"crawlkit {__version__}")


if __name__ == "__main__":
    app()
