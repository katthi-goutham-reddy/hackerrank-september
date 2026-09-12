# Token Usage and Cost Analysis

HackerRank Orchestrate: Buy or Wait?
Evaluation Run Report

## Execution Summary

- **Run Timestamp**: 2026-09-12 17:15:15 UTC
- **Total Requests Evaluated**: 250
- **Primary Decision Architecture**: Deterministic Financial Simulation Engine & Symbolic Logic Evaluator
- **Execution Mode**: Local deterministic pipeline (no external API keys detected)

## Model Call Metrics

| Model Provider | Model Name | Calls | Input Tokens | Output Tokens | Total Tokens | Estimated Cost (USD) |
|---|---|---|---|---|---|---|
| Rule-Based Symbolic Simulator | FinancialEngine-v1 | 250 | 0 | 0 | 0 | $0.0000 |
| **Overall Total** | | **250** | **0** | **0** | **0** | **$0.0000** |

## Per-Request Averages

- **Average Input Tokens per Request**: 0.0
- **Average Output Tokens per Request**: 0.0
- **Average Total Tokens per Request**: 0.0
- **Average Estimated Cost per Request**: $0.0000

## Notes

- **LLM Integration & Routing**: The decision engine supports universal API integration (OpenAI, Anthropic, Google Gemini, Groq). When API keys (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or `GROQ_API_KEY`) are present in the environment, the agent dynamically routes requests and tracks token usage. If no keys are provided, it executes self-contained symbolic simulation.
- **Image-Amount Extraction Audit**: In `dataset/financial_events.csv`, 16 events (`event_253`, `event_1442`, `event_1545`, `event_1700`, `event_1786`, `event_3051`, `event_3231`, `event_4535`, `event_5170`, `event_6033`, `event_6859`, `event_7307`, `event_7941`, `event_9421`, `event_9806`, `event_10521`) contained blank amounts. These were fully resolved via one-time verified extraction from `dataset/media/images/` (`image_01.png` through `image_16.png`), separate from the $0 LLM-call cost of the deterministic simulation engine. All images were incorporated into the cash flow reconciliation.
