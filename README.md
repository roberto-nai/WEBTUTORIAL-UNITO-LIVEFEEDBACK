# Live Help

Flask module for the PHP tutorial project.

## Start

Use **Python 3.11**.

It is recommended to create and activate a virtual environment before installing dependencies:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:5050/live_help/<sessionID>
```

Compatibility URL (also accepted):

```text
http://127.0.0.1:5050/<sessionID>
```

## Environment variables

Create a local `.env` file in the project root (see  `.env.example`), then set your MySQL values:

```env
MYSQL_HOST=127.0.0.1
MYSQL_PORT=8889
MYSQL_USER=your_mysql_user
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=my_webtutorial
```

## Project structure

- `app.py` - Flask app and web routes.
- `feedback_strategy.py` - builds the feedback context used by the LLM.
- `llm_service.py` - generates the process-aware feedback text.
- `log_service.py` - builds the DFG and exports session logs.
- `ml_service.py` - extracts features and trains/uses the XGB model.
- `sql_service.py` - reads session data from MySQL.
- `requirements.txt` - Python dependencies.
- `README.md` - project documentation.
- `.gitignore` - ignores generated and local files.
- `.env.example` - example environment configuration.
- `prompts/` - prompt templates and feedback intent definitions.
  - `feedback_intents.json` - feedback intent catalogue.
  - `process_feedback_prompt_v1.json` - first prompt version.
  - `process_feedback_prompt_v2.json` - current prompt version.
- `templates/` - HTML templates for the web UI.
  - `index.html` - main live feedback page.
- `dfg/` - generated DFG PNG files at runtime.
- `logs/` - exported session CSV files at runtime.
- `models/` - training dataset and saved XGB artefacts.

## Current modules

- Live session visualisation in the browser
- PM4Py DFG from `events` and `quiz`
- Event log and quiz summary tables
- Basic live metrics
- XGB placeholder after the first 3 pages
- LLM feedback placeholder

## LLM used

- Runtime LLM service: **Ollama**: [https://ollama.com](https://ollama.com)
- Model: **Llama 3.2-3B**: [https://ollama.com/library/llama3.2](https://ollama.com/library/llama3.2)
