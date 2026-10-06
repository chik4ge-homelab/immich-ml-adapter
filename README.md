# immich-ml-adapter

`immich-ml-adapter` is a small protocol adapter for Immich v3.2.0. It exposes
the Immich Machine Learning API on port 3003 and keeps the stock
`immich-machine-learning` deployment as the fallback for every task that is
not explicitly assigned to EmbeddingGemma 2.

## Routing

The only EmbeddingGemma compatibility alias is
`ViT-B-16-SigLIP2__webli`. Its 768-dimensional Immich model-name entry is
defined by Immich Server v3.2.0; the alias is not the actual backend model.

```text
Immich /predict
    -> immich-ml-adapter
       -> stock immich-machine-learning:3003
       -> LiteLLM /v1/embeddinggemma/embeddings
          -> EmbeddingGemma backend (LiteLLM-only network path)
```

The adapter has no EmbeddingGemma backend URL. The only configured upstreams
are the stock ML URL and the LiteLLM URL. Network policy enforces the same
boundary.

`GET /ping` returns the Immich-compatible plain-text response `pong`.
`POST /predict` accepts Immich's multipart `entries` JSON plus exactly one
`text` or `image` field. Embedding responses are serialized as the stringified
JSON array required by Immich, for example `{"clip":"[0.1,-0.2]"}`.

## LiteLLM contract

The adapter uses the authenticated custom pass-through route
`/v1/embeddinggemma/embeddings`. It sends a generic OpenAI-like embedding body
to that route, including `input_type=query` for Immich text search and
`input_type=document` for Immich image assets. The API key is injected from a
Kubernetes Secret and is never logged.

LiteLLM also registers `embeddinggemma-2` as a native text embedding model so
other internal clients can compare the standard `/v1/embeddings` route. The
adapter uses pass-through for both modalities because LiteLLM's native
multimodal translation must be verified independently and cannot be assumed to
preserve image content blocks.

## Rollback

1. Change Immich's Smart Search model back to its previous model name. The
   adapter sends non-alias CLIP requests to stock ML.
2. Change `IMMICH_MACHINE_LEARNING_URL` back to
   `http://immich-machine-learning:3003` and leave the stock deployment in
   place.

Do not run a full Smart Search reindex until text/image correctness and a small
asset sample have been validated. EmbeddingGemma 2 occupies the same 768
dimensions as some older models but is a different vector space, so existing
embeddings must not be mixed with new ones.
