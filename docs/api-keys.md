# API keys

Only one key is ever needed: `GEMINI_API_KEY`. It is only required for
the LLM detection and annotation paths. The signal and ensemble
(`--mode fast`) detectors run fully local.

## Getting a key

1. Go to https://aistudio.google.com/apikey.
2. Create an API key (free tier is fine to start).
3. Paste it into `.env` at the repo root:

```
GEMINI_API_KEY=your-key-here
```

`./setup.sh` will prompt you for this on first run.

## Key resolution order

Scripts look up the key in this order:

1. `GEMINI_API_KEY` from the environment.
2. `GOOGLE_API_KEY` from the environment (legacy alias).
3. Variables loaded from `.env` at the repo root.
4. Variables loaded from `tools/shared/.env` (only annotation tools).

If none are set and an LLM path is invoked, the script exits with a
clear error.

## Free-tier limits

Gemini's free tier rate limits are tight. If you see HTTP 429:

- Reduce parallelism (`--parallel 1` in the annotation tools).
- Wait a minute and retry.
- Upgrade to a paid Gemini plan.

There is no other key. SetOptics does not phone home, does not collect
telemetry, and does not require any other paid service to run end-to-end.
