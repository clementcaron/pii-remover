# Agent Guidelines

## Behavioral Rules

These rules apply to every task in this project unless explicitly overridden.
Bias: caution over speed on non-trivial work. Use judgment on trivial tasks.

### Rule 1 — Think Before Coding
State assumptions explicitly. If uncertain, ask rather than guess.
Present multiple interpretations when ambiguity exists.
Push back when a simpler approach exists.
Stop when confused. Name what's unclear.

### Rule 2 — Simplicity First
Minimum code that solves the problem. Nothing speculative.
No features beyond what was asked. No abstractions for single-use code.
Test: would a senior engineer say this is overcomplicated? If yes, simplify.

### Rule 3 — Surgical Changes
Touch only what you must. Clean up only your own mess.
Don't "improve" adjacent code, comments, or formatting (only if not asked).
Don't refactor what isn't broken. Match existing style.

### Rule 4 — Goal-Driven Execution
Define success criteria. Loop until verified.
Don't follow steps. Define success and iterate (max iter: 5-7).
Strong success criteria let you loop independently.

### Rule 5 — Use the model only for judgment calls
Use me for: classification, drafting, summarization, extraction.
Do NOT use me for: routing, retries, deterministic transforms.
If code can answer, code answers.

### Rule 6 — Do not commit
I prefer to commit myself — give me the diff and I'll review it. Examples of good commit messages:
- `fix: Fix cursor invisibility at start of big fraction`
- `feat: Implement snap-to-nearest cursor slot algorithm`
- `feat: Add GitHub Actions workflows for CI and Android release automation`
- `misc: Make Android release signing conditional on keystore presence`
- `fix: Parse and review Gradle problems report for build diagnostics`

### Rule 7 — Surface conflicts, don't average them
If two patterns contradict, pick one (more recent / more tested).
Explain why. Flag the other for cleanup.
Don't blend conflicting patterns.

### Rule 8 — Read before you write
Before adding code, read exports, immediate callers, shared utilities.
"Looks orthogonal" is dangerous. If unsure why code is structured a way, ask.

### Rule 9 — Tests verify intent, not just behavior
Tests must encode WHY behavior matters, not just WHAT it does.
A test that can't fail when business logic changes is wrong.
Join with Rule 8: Write tests before implementing.

### Rule 10 — Checkpoint after every significant step
Summarize what was done, what's verified, what's left (in short).
Don't continue from a state you can't describe back.
If you lose track, stop and restate.

### Rule 11 — Match the codebase's conventions, even if you disagree
Conformance > taste inside the codebase.
If you genuinely think a convention is harmful, surface it. Don't fork silently.

### Rule 12 — Fail loud
"Completed" is wrong if anything was skipped silently.
"Tests pass" is wrong if any were skipped.
Default to surfacing uncertainty, not hiding it.

### Security
Never hardcode secrets or sensitive information in code or config files. Use environment variables or secure vaults. Never commit secrets to version control, even in "example" or "bak" files.

---

## Project Stack & Commands
- **Stack:** Python 3.13+, FastAPI, Uvicorn, HTTPX, gliner (v0.2.26), PyTorch (cu130)
- **Package Manager:** uv
- **Test:** `python3 scripts/test_anonymization.py`
- **Build:** `docker compose up -d --build`
- **Setup:** `./scripts/interactive_setup.sh`

---

## Directory Structure
- `src/api/`: FastAPI routes and authentication middleware.
- `src/core/`: Configuration and security-hardened logging.
- `src/engine/`: Detection engine logic (`NuNERAnonymizer`) with entity merging.
- `src/proxy/`: `StreamHandler` for SSE de-anonymization and request state management.
- `scripts/`: Setup and testing utilities.
- `docker/`: Dockerfile and container configuration.

---

## Project Context (PII Remover Proxy)

### 1. What the Project Does
The **PII Remover Proxy** is a reverse proxy functioning as an AI Gateway. It intercepts outbound requests to Large Language Models (OpenAI, Anthropic, Gemini) and performs **Zero-Shot Named Entity Recognition (NER)** to anonymize Personally Identifiable Information (PII). When the LLM streams or returns a response, the proxy intercepts it again to de-anonymize the synthetic tokens back into the original data before forwarding it to the end client.

### 2. What NOT to Do (The "Failure Log" & Guidelines)
- **Do NOT bake models into Docker images:** Keep models stored in the `hf_cache` Docker volume mapping to `/home/appuser/.cache`.
- **Do NOT hardcode CPU/GPU deps in the main run trace:** The project dynamically swaps to the PyTorch `cu130` wheel inside the Dockerfile only if `USE_GPU="true"` is passed.
- **Do NOT break SSE Streaming:** De-anonymization has a complex look-ahead buffer in `StreamHandler` to replace overlapping chunks.
- **Do NOT require root in container:** Keep `appuser` as the active execution user.
- **Do NOT strictly string-match mapped tokens:** LLMs often wrap or escape synthetic tokens using Markdown syntax (e.g. `[PHONE\_NUMBER\_1]`). Use `re.sub` allowing optional backslashes.
- **Do NOT inject unescaped replacement strings:** Use `json.dumps(original)[1:-1]` to securely escape newlines/quotes when swapping tokens in JSON streams.

### 3. Points to Upgrade (Production Roadmap)
1. **State Persistence (Redis):** Centralize ephemeral state mapping to an external store like Redis.
2. **Advanced Caching & Routing:** Implement request-response caching and fallback model routing.
3. **Enhanced Logging & Telemetry:** Tie into OpenTelemetry or Prometheus/Grafana for VRAM and latency metrics.
4. **Enhanced Entity Coverage:** Improve regex-fallback combined with NuNER to catch edge-case formats.
5. **UI & Middleware Standardization:** Ensure the proxy remains untethered from bespoke UI hacks (e.g., stripping `thinkingConfig`).

