# GPT Enhancement Documentation

## Purpose

The GPT Enhancement module integrates OpenAI GPT analysis with the existing greyhound racing prediction system. It enhances machine learning predictions with GPT-generated insights, providing narrative analysis, betting strategies, pattern analysis, and comprehensive reports.

## Prerequisites

- Python 3.8+
- OpenAI API key configured in an `.env` file
- Access to the `greyhound_racing_data.db` database

### Environment Variables

- `OPENAI_API_KEY`: The API key for authenticating requests to OpenAI's services.

## Typical Workflow

1. **Setup**: Ensure the OpenAI API key is configured and the database is available.
2. **Enhance Predictions**:
   - Use the `api/gpt/enhance_race` endpoint with the necessary parameters to enhance predictions.
   - Include the `race_file_path` in the POST request.
3. **Generate Insights**:
   - Use the `api/gpt/daily_insights` endpoint to generate daily insights.
   - Use the `api/gpt/comprehensive_report` endpoint to create reports.
4. **Monitor and Adjust**:
   - Regularly check the Flask logs for any anomalies or issues.

## Cost Awareness

The GPT API usage incurs token-based costs based on the amount of data processed:
- Input tokens: $0.03 per 1K tokens
- Output tokens: $0.06 per 1K tokens

Estimates suggest a cost between $0.15 - $0.30 per race.

Optimize usage by processing only necessary data and monitoring token usage via the `/api/gpt/status` endpoint.

---
