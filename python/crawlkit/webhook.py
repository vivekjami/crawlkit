# python/crawlkit/webhook.py
"""
Production-ready Firecrawl webhook receiver.

Security features
-----------------
- HMAC-SHA256 signature verification (X-Firecrawl-Signature header)
- `hmac.compare_digest` prevents timing attacks (never use plain ==)
- Idempotency via in-memory `_processed` set (dedup by event `id`)
- BackgroundTasks -- return 200 immediately, process async after ACK
- Mark processed AFTER handler completes (re-delivery on crash is fine)

Usage
-----
    from crawlkit.webhook import WebhookServer, WebhookConfig

    server = WebhookServer(WebhookConfig(secret="your-webhook-secret"))

    @server.on("crawl.page")
    async def handle_page(event: dict) -> None:
        for page in event.get("data", []):
            print(page.get("metadata", {}).get("sourceURL"))

    @server.on("crawl.completed")
    def handle_done(event: dict) -> None:
        print(f"Job {event['id']} done")

    import asyncio
    asyncio.run(server.serve())
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any  # ← NEW

logger = logging.getLogger(__name__)

try:
    import uvicorn
    from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
    from fastapi import FastAPI as _FastAPI

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


@dataclass
class WebhookConfig:
    """
    Configuration for the webhook server.

    Args:
        secret:         Webhook signing secret from Firecrawl dashboard.
                        Set to empty string to DISABLE verification (never in prod).
        host:           Bind address. Default: 0.0.0.0
        port:           Bind port.    Default: 8000
        path:           URL path for the webhook endpoint.
        max_processed:  Max event IDs to keep before pruning.
    """

    secret: str
    host: str = "0.0.0.0"
    port: int = 8000
    path: str = "/webhook"
    max_processed: int = 10_000


class WebhookServer:
    """
    FastAPI-based Firecrawl webhook server.

    Register handlers with @server.on("event.type").
    Handlers may be sync or async -- both are supported.
    Multiple handlers per event type are allowed.
    """

    def __init__(self, config: WebhookConfig) -> None:
        if not _FASTAPI_AVAILABLE:
            raise RuntimeError("fastapi and uvicorn are required: pip install fastapi uvicorn")
        self._config = config
        self._handlers: dict[str, list[Callable[[dict[str, Any]], Any]]] = {}  # ← fixed
        self._processed: set[str] = set()
        self.router = APIRouter()
        self.router.add_api_route(
            config.path,
            self._handle_webhook,
            methods=["POST"],
            status_code=200,
        )

    def on(
        self, event_type: str
    ) -> Callable[[Callable[[dict[str, Any]], Any]], Callable[[dict[str, Any]], Any]]:
        """
        Decorator. Register a handler for a Firecrawl event type.

        Known event types: crawl.page, crawl.completed, crawl.failed, crawl.started
        """

        def decorator(
            fn: Callable[[dict[str, Any]], Any],
        ) -> Callable[[dict[str, Any]], Any]:  # ← fixed
            self._handlers.setdefault(event_type, []).append(fn)
            logger.debug("Registered %r for event %r", fn.__name__, event_type)
            return fn

        return decorator

    async def _handle_webhook(
        self, request: Request, background_tasks: BackgroundTasks
    ) -> dict[str, Any]:
        raw_body = await request.body()

        if self._config.secret:
            self._verify_signature(raw_body, request)

        try:
            payload: dict[str, Any] = await request.json()  # ← fixed
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid JSON: {exc}"
            ) from None  # ← fixed (B904)

        event_id: str = payload.get("id", "")
        event_type: str = payload.get("type", "")

        if event_id and event_id in self._processed:
            logger.debug("Duplicate event %s -- ignoring", event_id)
            return {"status": "duplicate"}

        background_tasks.add_task(self._dispatch, event_id, event_type, payload)
        return {"status": "accepted"}

    async def _dispatch(
        self, event_id: str, event_type: str, payload: dict[str, Any]
    ) -> None:  # ← fixed
        handlers = self._handlers.get(event_type, [])
        if not handlers:
            logger.debug("No handler for %r", event_type)

        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    await handler(payload)
                else:
                    handler(payload)
            except Exception as exc:
                logger.exception(
                    "Handler %r raised for event %s/%s: %s",
                    handler.__name__,
                    event_type,
                    event_id,
                    exc,
                )

        if event_id:
            self._processed.add(event_id)
            if len(self._processed) > self._config.max_processed:
                overage = len(self._processed) - self._config.max_processed // 2
                for old in list(self._processed)[:overage]:
                    self._processed.discard(old)

    def _verify_signature(self, raw_body: bytes, request: Request) -> None:
        """
        Verify Firecrawl's HMAC-SHA256 webhook signature.

        Uses hmac.compare_digest -- NEVER use plain == for secret comparison.
        Plain == is vulnerable to timing attacks.
        compare_digest always takes the same time regardless of content.
        """
        sig_header = request.headers.get("X-Firecrawl-Signature", "")
        if not sig_header:
            logger.warning("Webhook received without signature header")
            raise HTTPException(status_code=401, detail="Missing signature")

        expected = hmac.new(
            self._config.secret.encode(),
            raw_body,
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(expected, sig_header):
            logger.warning("Webhook signature mismatch")
            raise HTTPException(status_code=401, detail="Invalid signature")

    async def serve(
        self,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        """
        Spin up a uvicorn server serving only this webhook.

        For production: mount server.router on your existing FastAPI app instead.
        """
        app = _FastAPI(title="crawlkit webhook receiver")
        app.include_router(self.router)
        config = uvicorn.Config(
            app,
            host=host or self._config.host,
            port=port or self._config.port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()
