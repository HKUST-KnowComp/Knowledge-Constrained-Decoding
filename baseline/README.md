OPENAI API References


# Completion API

* Parameters
  * NOTE: GPT3 uses the same tokenizer as GPT2

```python
def completion(
        model: str,
        prompt: str | list[str],
        suffix: str = None,
        max_tokens: int = 16,
        temperature: float = 1,  # between (0, 2),
        top_p: float = 1,
        n: int = 1,  # number of completions for each prompt)
        logprobs: int = 0,  # max 5,
        stop: str | list[str] = None,  # stop token, <=4,
        best_of: int = 1,
        logit_bias: dict[str, int] = None,
        presence_penalty: float = 0,  # between (-2, 2),
        frequency_penalty: float = 0,  # between (-2, 2),
)
```

* `presence_penalty: Optional[number] = 0`
  * Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.

* `frequency_penalty: Optional[number] = 0`
  * Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.



* Response

```json
{
  "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
  "object": "text_completion",
  "created": 1589478378,
  "model": "text-davinci-003",
  "choices": [
    {
      "text": "\n\nThis is indeed a test",
      "index": 0,
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 7,
    "total_tokens": 12
  }
}
```

# Chat API

* Parameters
```python
def completion(
        model: str,
        messages: list[{
                        "role": str = choice[system, user, assistant]
                        "content": str
                        "name": str = None
                        }],
        max_tokens: int = 16,
        temperature: float = 1,  # between (0, 2),
        top_p: float = 1,
        n: int = 1,  # number of completions for each prompt)
        stop: str | list[str] = None,  # stop token, <=4,
        best_of: int = 1,
        logit_bias: dict[str, int] = None,
        presence_penalty: float = 0,  # between (-2, 2),
        frequency_penalty: float = 0,  # between (-2, 2),
)
```
* Response
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "\n\nHello there, how may I assist you today?",
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "total_tokens": 21
  }
}
```
