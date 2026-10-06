from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from PIL import Image, UnidentifiedImageError
from starlette.formparsers import MultiPartParser

from .config import Settings

LOGGER = logging.getLogger("immich_ml_adapter")
EXPECTED_DIMENSION = 768
MultiPartParser.max_file_size = 64 * 1024 * 1024


class AdapterError(Exception):
    def __init__(
        self, status_code: int, code: str, message: str, *, upstream_status: int | None = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.upstream_status = upstream_status


class InvalidEntries(AdapterError):
    def __init__(self, message: str):
        super().__init__(400, "invalid_entries", message)


@dataclass(frozen=True)
class RequestEntries:
    embedding: dict[str, Any] | None
    stock: dict[str, Any]


@dataclass(frozen=True)
class ImageInfo:
    content_type: str
    width: int
    height: int


@dataclass
class Runtime:
    settings: Settings
    client: httpx.AsyncClient
    owns_client: bool


def _log(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    LOGGER.info(json.dumps(payload, separators=(",", ":"), sort_keys=True))


def _request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or str(uuid4())


def _parse_entries(raw: str, alias: str) -> RequestEntries:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise InvalidEntries("entries must be a JSON object") from error

    if not isinstance(payload, dict) or not payload:
        raise InvalidEntries("entries must be a non-empty JSON object")

    embedding_entry: dict[str, Any] | None = None
    stock: dict[str, Any] = {}
    for task, types in payload.items():
        if not isinstance(task, str) or not isinstance(types, dict):
            raise InvalidEntries("entries must map task names to model type objects")
        stock_types: dict[str, Any] = {}
        for model_type, entry in types.items():
            if not isinstance(model_type, str) or not isinstance(entry, dict):
                raise InvalidEntries("each model entry must be an object")
            model_name = entry.get("modelName")
            if not isinstance(model_name, str) or not model_name:
                raise InvalidEntries("each model entry requires modelName")
            selected = (
                task == "clip" and model_type in {"textual", "visual"} and model_name == alias
            )
            if selected:
                if embedding_entry is not None:
                    raise InvalidEntries(
                        "one /predict request cannot contain multiple EmbeddingGemma entries"
                    )
                embedding_entry = {
                    "task": task,
                    "type": model_type,
                    "entry": entry,
                }
            else:
                stock_types[model_type] = entry
        if stock_types:
            stock[task] = stock_types

    if embedding_entry is not None and "clip" in stock:
        raise InvalidEntries("a request cannot mix two clip models with one response key")
    if embedding_entry is None and not stock:
        raise InvalidEntries("entries did not contain a routable model")
    return RequestEntries(embedding=embedding_entry, stock=stock)


def _image_info(image_bytes: bytes, supplied_content_type: str | None) -> ImageInfo:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            image_format = (image.format or "").upper()
            content_type = Image.MIME.get(image_format) or supplied_content_type or "image/jpeg"
            return ImageInfo(content_type=content_type, width=image.width, height=image.height)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise AdapterError(400, "invalid_image", "image is not a readable image") from error


def _data_url(image_bytes: bytes, image: ImageInfo) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{image.content_type};base64,{encoded}"


def _error_response(error: AdapterError, request_id: str) -> JSONResponse:
    body: dict[str, Any] = {
        "error": {"code": error.code, "message": error.message},
        "request_id": request_id,
    }
    if error.upstream_status is not None:
        body["error"]["upstream_status"] = error.upstream_status
    return JSONResponse(body, status_code=error.status_code)


def _embedding_body(
    settings: Settings,
    entry: dict[str, Any],
    text: str | None,
    image_bytes: bytes | None,
    image: ImageInfo | None,
) -> dict[str, Any]:
    model_type = entry["type"]
    if model_type == "textual":
        if text is None:
            raise InvalidEntries("clip.textual requires the text multipart field")
        return {
            "model": settings.embedding_model,
            "input": text,
            "input_type": "query",
        }
    if model_type == "visual":
        if image_bytes is None or image is None:
            raise InvalidEntries("clip.visual requires the image multipart field")
        return {
            "model": settings.embedding_model,
            "input": [
                {
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(image_bytes, image)},
                        }
                    ]
                }
            ],
            "input_type": "document",
        }
    raise InvalidEntries(f"unsupported EmbeddingGemma model type: {model_type}")


def _validate_embedding(payload: Any) -> list[float]:
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("data"), list)
        or len(payload["data"]) != 1
    ):
        raise AdapterError(
            502, "invalid_embedding_response", "embedding upstream returned an invalid response"
        )
    item = payload["data"][0]
    vector = item.get("embedding") if isinstance(item, dict) else None
    if not isinstance(vector, list) or len(vector) != EXPECTED_DIMENSION:
        raise AdapterError(
            502, "invalid_embedding_dimension", "embedding upstream returned the wrong dimension"
        )
    try:
        values = [float(value) for value in vector]
    except (TypeError, ValueError) as error:
        raise AdapterError(
            502, "invalid_embedding_values", "embedding upstream returned non-numeric values"
        ) from error
    if not all(math.isfinite(value) for value in values):
        raise AdapterError(
            502, "invalid_embedding_values", "embedding upstream returned non-finite values"
        )
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or abs(norm - 1.0) > 0.01:
        raise AdapterError(
            502, "unnormalized_embedding", "embedding upstream returned an unnormalized vector"
        )
    return values


async def _get_runtime(request: Request) -> Runtime:
    return request.app.state.runtime


async def _post_stock(
    runtime: Runtime,
    entries: dict[str, Any],
    image_bytes: bytes | None,
    image_filename: str | None,
    image_content_type: str | None,
    text: str | None,
    request_id: str,
) -> httpx.Response:
    data = {"entries": json.dumps(entries, separators=(",", ":"))}
    files: dict[str, tuple[str, bytes, str]] = {}
    if image_bytes is not None:
        files["image"] = (
            image_filename or "image",
            image_bytes,
            image_content_type or "application/octet-stream",
        )
    if text is not None:
        data["text"] = text
    try:
        return await runtime.client.post(
            f"{runtime.settings.stock_ml_base_url}/predict",
            data=data,
            files=files or None,
            headers={"x-request-id": request_id},
            timeout=runtime.settings.upstream_timeout_seconds,
        )
    except httpx.TimeoutException as error:
        raise AdapterError(504, "stock_ml_timeout", "stock Immich ML timed out") from error
    except httpx.RequestError as error:
        raise AdapterError(502, "stock_ml_unavailable", "stock Immich ML is unavailable") from error


async def _post_embedding(
    runtime: Runtime,
    body: dict[str, Any],
    request_id: str,
) -> httpx.Response:
    if not runtime.settings.litellm_api_key:
        raise AdapterError(503, "missing_litellm_api_key", "LiteLLM API key is not configured")
    try:
        return await runtime.client.post(
            runtime.settings.litellm_embedding_url,
            json=body,
            headers={
                "authorization": f"Bearer {runtime.settings.litellm_api_key}",
                "x-request-id": request_id,
            },
            timeout=runtime.settings.upstream_timeout_seconds,
        )
    except httpx.TimeoutException as error:
        raise AdapterError(
            504, "embedding_timeout", "LiteLLM embedding request timed out"
        ) from error
    except httpx.RequestError as error:
        raise AdapterError(
            502, "litellm_unavailable", "LiteLLM embedding route is unavailable"
        ) from error


async def _embedding_result(
    runtime: Runtime,
    entry: dict[str, Any],
    text: str | None,
    image_bytes: bytes | None,
    image: ImageInfo | None,
    request_id: str,
) -> dict[str, Any]:
    body = _embedding_body(runtime.settings, entry, text, image_bytes, image)
    started = time.monotonic()
    response = await _post_embedding(runtime, body, request_id)
    upstream_latency_ms = round((time.monotonic() - started) * 1000, 2)
    if not response.is_success:
        status = response.status_code
        code = "embedding_upstream_error"
        if status in {401, 403}:
            code = "litellm_authentication_failed"
        elif status == 429:
            code = "litellm_rate_limited"
        elif status >= 500:
            code = "embedding_backend_error"
        _log(
            "upstream_error",
            request_id=request_id,
            selected_upstream="litellm",
            upstream_status=status,
            upstream_latency_ms=upstream_latency_ms,
        )
        raise AdapterError(
            502, code, "LiteLLM rejected the embedding request", upstream_status=status
        )
    try:
        values = _validate_embedding(response.json())
    except ValueError as error:
        raise AdapterError(
            502, "invalid_embedding_response", "embedding upstream returned invalid JSON"
        ) from error
    result: dict[str, Any] = {"clip": json.dumps(values, separators=(",", ":"), allow_nan=False)}
    if entry["type"] == "visual" and image is not None:
        result.update(imageHeight=image.height, imageWidth=image.width)
    _log(
        "upstream_complete",
        request_id=request_id,
        route="embeddinggemma",
        task="clip",
        model_name=entry["entry"]["modelName"],
        selected_upstream="litellm",
        modality=entry["type"],
        image_bytes=len(image_bytes) if image_bytes is not None else 0,
        upstream_latency_ms=upstream_latency_ms,
        upstream_status=response.status_code,
        http_status=200,
    )
    return result


@asynccontextmanager
async def _lifespan(app: FastAPI):
    settings = Settings.from_env()
    client = httpx.AsyncClient()
    app.state.runtime = Runtime(settings=settings, client=client, owns_client=True)
    _log("startup", model_alias=settings.model_alias, embedding_model=settings.embedding_model)
    try:
        yield
    finally:
        await client.aclose()


def create_app() -> FastAPI:
    application = FastAPI(title="Immich ML Adapter", lifespan=_lifespan)

    @application.get("/")
    async def root() -> dict[str, str]:
        return {"message": "Immich ML Adapter"}

    @application.get("/ping", response_class=PlainTextResponse)
    async def ping() -> str:
        return "pong"

    @application.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/health/ready")
    async def ready() -> dict[str, str]:
        return {"status": "ready"}

    @application.get("/health/upstreams")
    async def upstreams(request: Request) -> Response:
        runtime = await _get_runtime(request)

        async def check(url: str, path: str) -> dict[str, Any]:
            started = time.monotonic()
            try:
                response = await runtime.client.get(
                    f"{url}{path}", timeout=min(runtime.settings.upstream_timeout_seconds, 10.0)
                )
                return {
                    "status": "ok" if response.is_success else "error",
                    "http_status": response.status_code,
                    "latency_ms": round((time.monotonic() - started) * 1000, 2),
                }
            except httpx.RequestError:
                return {"status": "unavailable"}

        stock, litellm = await asyncio.gather(
            check(runtime.settings.stock_ml_base_url, "/ping"),
            check(runtime.settings.litellm_base_url, "/health/readiness"),
        )
        result = {"stock_ml": stock, "litellm": litellm}
        status_code = 200 if all(item["status"] == "ok" for item in result.values()) else 503
        return JSONResponse(result, status_code=status_code)

    @application.post("/predict")
    async def predict(
        request: Request,
        entries: str = Form(...),
        image: UploadFile | None = File(default=None),
        text: str | None = Form(default=None),
    ) -> Response:
        request_id = _request_id(request)
        started = time.monotonic()
        runtime = await _get_runtime(request)
        image_bytes: bytes | None = None
        image_meta: ImageInfo | None = None
        try:
            if image is not None and text is not None:
                raise InvalidEntries("provide exactly one of image or text")
            if image is None and text is None:
                raise InvalidEntries("either image or text must be provided")
            if image is not None:
                image_bytes = await image.read()
                if not image_bytes:
                    raise AdapterError(400, "empty_image", "image must not be empty")
                if len(image_bytes) > runtime.settings.max_image_bytes:
                    raise AdapterError(
                        413, "image_too_large", "image exceeds the configured size limit"
                    )
                image_meta = _image_info(image_bytes, image.content_type)

            routed = _parse_entries(entries, runtime.settings.model_alias)
            if routed.embedding is not None:
                embedding_result = await _embedding_result(
                    runtime,
                    routed.embedding,
                    text,
                    image_bytes,
                    image_meta,
                    request_id,
                )
                if routed.stock:
                    stock_response = await _post_stock(
                        runtime,
                        routed.stock,
                        image_bytes,
                        image.filename if image is not None else None,
                        image.content_type if image is not None else None,
                        text,
                        request_id,
                    )
                    if not stock_response.is_success:
                        raise AdapterError(
                            502,
                            "stock_ml_error",
                            "stock Immich ML rejected the request",
                            upstream_status=stock_response.status_code,
                        )
                    try:
                        merged = stock_response.json()
                    except ValueError as error:
                        raise AdapterError(
                            502, "invalid_stock_response", "stock Immich ML returned invalid JSON"
                        ) from error
                    if not isinstance(merged, dict):
                        raise AdapterError(
                            502, "invalid_stock_response", "stock Immich ML returned a non-object"
                        )
                    merged.update(embedding_result)
                    result: Response = JSONResponse(merged)
                else:
                    result = JSONResponse(embedding_result)
            else:
                stock_response = await _post_stock(
                    runtime,
                    routed.stock,
                    image_bytes,
                    image.filename if image is not None else None,
                    image.content_type if image is not None else None,
                    text,
                    request_id,
                )
                if not stock_response.is_success:
                    raise AdapterError(
                        502,
                        "stock_ml_error",
                        "stock Immich ML rejected the request",
                        upstream_status=stock_response.status_code,
                    )
                result = Response(
                    content=stock_response.content,
                    status_code=stock_response.status_code,
                    media_type="application/json",
                )
            _log(
                "request_complete",
                request_id=request_id,
                route="embeddinggemma" if routed.embedding is not None else "stock_ml",
                task=(routed.embedding or {}).get("task", "stock"),
                model_name=((routed.embedding or {}).get("entry") or {}).get("modelName", "stock"),
                selected_upstream="litellm" if routed.embedding is not None else "stock_ml",
                modality="image" if image_bytes is not None else "text",
                image_bytes=len(image_bytes) if image_bytes is not None else 0,
                total_latency_ms=round((time.monotonic() - started) * 1000, 2),
                http_status=result.status_code,
            )
            return result
        except AdapterError as error:
            _log(
                "request_error",
                request_id=request_id,
                route="embeddinggemma"
                if "routed" in locals() and routed.embedding is not None
                else "stock_ml",
                image_bytes=len(image_bytes) if image_bytes is not None else 0,
                total_latency_ms=round((time.monotonic() - started) * 1000, 2),
                http_status=error.status_code,
                error_code=error.code,
            )
            return _error_response(error, request_id)

    return application


app = create_app()
