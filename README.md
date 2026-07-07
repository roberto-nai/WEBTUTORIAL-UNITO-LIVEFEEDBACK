# Live Feedback

Flask module for the Live Feedback

See also: [https://github.com/roberto-nai/AILEAP-2026](https://github.com/roberto-nai/AILEAP-2026)

## Start

Use **Python 3.11**.

It is recommended to create and activate a virtual environment before installing dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
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

See also: [https://github.com/roberto-nai/WEBTUTORIAL-UNITO](https://github.com/roberto-nai/WEBTUTORIAL-UNITO)

## Environment variables

Create a local `.env` file in the project root (see  `.env.example`), then set your MySQL values:

```env
MYSQL_HOST=127.0.0.1
MYSQL_PORT=8889
MYSQL_USER=your_mysql_user
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=your_db_name
```

Optional:

```env
HELP_REFRESH_SECONDS=30
ENABLE_LOCAL_LLM=1
OLLAMA_MODEL=llama3.2
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
- Event log table with default preview and expand/collapse controls
- Basic live metrics
- XGB prediction after the first 3 pages
- Process-aware LLM feedback (cached and sanitised)
- Feedback usefulness survey (1-5 stars) stored in `llm_survey`
- Generic error page with detailed traces logged in `app.log`

## Notes

- Event log preview rows are controlled server-side by `EVENT_LOG_PREVIEW_ROWS` in `app.py`.

## Database notes

Feedback rating persistence relies on the MySQL table `llm_survey`.

Current table structure used by the app:

```sql
CREATE TABLE `llm_survey` (
  `feedbackID` int(11) NOT NULL AUTO_INCREMENT,
  `sessionID` varchar(256) NOT NULL,
  `projectID` int(11) DEFAULT NULL,
  `rating` tinyint(1) NOT NULL,
  `feedbackIntent` enum('Encouragement','Review','Warning') NOT NULL,
  `predictedOutcome` varchar(32) DEFAULT NULL,
  `promptVersion` varchar(64) DEFAULT NULL,
  `modelVersion` varchar(64) DEFAULT NULL,
  `feedbackHash` char(64) DEFAULT NULL,
  `lastUpdate` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`feedbackID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
```

## LLM used

- Runtime LLM service: **Ollama**: [https://ollama.com](https://ollama.com)
- Model: **Llama 3.2-3B**: [https://ollama.com/library/llama3.2](https://ollama.com/library/llama3.2)
