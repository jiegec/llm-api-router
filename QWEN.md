Create a LLM api router in this repo.
It should support both OpenAI and Anthropic API in separate endpoints.
It should support the vanilla openai and anthropic python API clients.
It should only change the model name according to the model mapping in JSON config.
It should not change the chat completion request.
It may parse the chat completion response to collect statistics. It may not change it.
Do not make assumptions about the chat completion request/response format, except for the ones that we use: model name and usage statistics. Just pass the json around.
For each endpoint, it supports multiple API backends, with some priority order.
When the high priority one failed or rate limited, use the next one automatically.
The router should be written in Python.
Manage your dependencies using poetry.
Write tests and github CI.
Git commit every time you have progress.
Write logs to files in json format.
Log includes request, response, statistics, retry attempts.
Cleanup unused code.
