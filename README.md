# Custom LLM Provider Inspector

Inspect, probe, benchmark, and analyze OpenAI-compatible API providers.

## Quick start

```bash
pip install -e .
provider-inspector

```
Or run directly:
```bash
python main.py

```
## Environment Variable Resolution
In the connection screen, you can enter environment variable names instead of manually pasting values.
You may enter:
 * A full Base URL
 * An environment variable ending with _URL (e.g., OPENAI_URL, MY_PROVIDER_URL)
 * $VARIABLE_NAME
If <NAME>_URL is provided, the inspector automatically looks for the corresponding <NAME>_KEY.
**Examples:**
 * OPENAI_URL  → looks for OPENAI_KEY
 * LOCAL_URL   → looks for LOCAL_KEY
 * MY_API_URL  → looks for MY_API_KEY (if present)
If no matching key is found, you will be prompted to enter one manually.
## Features
 * **Provider Summary** — connection info + model list
 * **Browse Models** — pick a model, run individual tests (chat, stream, tool, vision, JSON)
 * **Full Scan** — benchmark all models, export results as JSON
## Architecture
```
main.py              ← entry point
core/
  env.py             ← ~/.env resolver
  models.py          ← dataclasses (ModelInfo, CapabilityScan, etc.)
  api.py             ← APIClient (sync HTTP)
  scanner.py         ← provider detection, model refresh
  capabilities.py     ← quick capability check
  benchmark.py       ← test functions (chat, stream, tool, vision, json)
  session.py          ← app session state
ui/
  app.py             ← ProviderInspectorApp (state machine)
  screens.py         ← all screen renderers
  theme.py           ← Rich theme + style constants

```
