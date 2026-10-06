import os
from dataclasses import dataclass


def _float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def _int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


@dataclass(frozen=True)
class Settings:
    stock_ml_base_url: str = "http://immich-machine-learning:3003"
    litellm_base_url: str = "http://litellm-proxy.llm-gateway.svc.cluster.local:4000"
    litellm_embedding_path: str = "/v1/embeddinggemma/embeddings"
    litellm_api_key: str = ""
    model_alias: str = "ViT-B-16-SigLIP2__webli"
    embedding_model: str = "embeddinggemma-2"
    upstream_timeout_seconds: float = 300.0
    max_image_bytes: int = 64 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            stock_ml_base_url=os.environ.get("STOCK_ML_BASE_URL", cls.stock_ml_base_url).rstrip(
                "/"
            ),
            litellm_base_url=os.environ.get("LITELLM_BASE_URL", cls.litellm_base_url).rstrip("/"),
            litellm_embedding_path=os.environ.get(
                "LITELLM_EMBEDDING_PATH", cls.litellm_embedding_path
            ),
            litellm_api_key=os.environ.get("LITELLM_API_KEY", ""),
            model_alias=os.environ.get("IMMICH_EMBEDDING_MODEL_ALIAS", cls.model_alias),
            embedding_model=os.environ.get("EMBEDDING_MODEL", cls.embedding_model),
            upstream_timeout_seconds=_float(
                "UPSTREAM_TIMEOUT_SECONDS", cls.upstream_timeout_seconds
            ),
            max_image_bytes=_int("MAX_IMAGE_BYTES", cls.max_image_bytes),
        )

    @property
    def litellm_embedding_url(self) -> str:
        return f"{self.litellm_base_url}/{self.litellm_embedding_path.lstrip('/')}"
