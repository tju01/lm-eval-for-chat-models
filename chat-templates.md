# 1. Overview

Chat language models are fine-tuned with a [chat template](https://github.com/FastEval/FastEval/blob/main/docs/model-type.md).
This does not refer to the kind of chat templates that can be found in a library like [PromptSource](https://github.com/bigscience-workshop/promptsource).
Instead, it refers to the way that a conversation between a user and an assistant is transformed to a prompt string.
As an example, consider the following conversation:

```json
[
  { "role": "system", "content": "You are a helpful assistant." },
  { "role": "user", "content": "Can you give me some lorem ipsum?" },
  { "role": "assistant", "content": "Sure, here you go!\n\nLorem ipsum dolor sit amet [...]" },
  { "role": "user", "content": "Thanks! Some more please, it's not enough." },
]
```

Using the ChatML chat template, this conversation would be transformed into the following prompt string that would then be the input to the fine-tuned language model.

```
<|im_start>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Can you give me some lorem ipsum?<|im_end|>
<|im_start|>assistant
Sure, here you go!

Lorem ipsum dolor sit amet [...]<|im_end|>
<|im_start|>user
Thanks! Some more please, it's not enough.<|im_end|>
```

While the vast majority of chat models use a chat template that just involves some simple string concatenation like this, **[some models](https://huggingface.co/openchat/openchat#conversation-template) require working on a token level** due to the `tokenize(A) + tokenize(B) != tokenize(A + B)` problem.
A correct implementation also needs to take this into account.

# 2. Implementation

## 2.1 Option for user to specify the chat template

Add an option to specify the chat template in the `--model_args` flag.

## 2.2 Implementing the chat templates

Implement the code that maps the conversation to a prompt string or list of tokens. There are a some existing libraries, but I hadn't found one that is suitable for evaluation due to insufficient stability, correctness and system message support. The current implementations from FastEval could be moved into a separate library, but I will also look again to see if there is a good existing library for this by now.

https://huggingface.co/docs/transformers/main/en/chat_templating

```jinja
{% if messages[0]['role'] == 'system' %}
  {% set loop_messages = messages[1:] %}
  {% set system_message = messages[0]['content'] %}
{% elif true == true and not '<<SYS>>' in messages[0]['content'] %}
  {% set loop_messages = messages %}{% set system_message = 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\\n\\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\\'t know the answer to a question, please don\\'t share false information.' %}
{% else %}
  {% set loop_messages = messages %}
  {% set system_message = false %}
{% endif %}

{% for message in loop_messages %}
  {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
    {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
  {% endif %}
  {% if loop.index0 == 0 and system_message != false %}
    {% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}
  {% else %}
    {% set content = message['content'] %}
  {% endif %}
  {% if message['role'] == 'user' %}
    {{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}
  {% elif message['role'] == 'system' %}
    {{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}
  {% elif message['role'] == 'assistant' %}
    {{ ' '  + content.strip() + ' ' + eos_token }}
  {% endif %}
{% endfor %}
```

## 2.3 Adding methods for doing chat inference

Add one or multiple functions to the `LM` class for chat inference. The input to those methods would be a conversation instead of a prompt string. This is because this chat functionality can also work for other models like API models that directly use the conversation.

## 2.4 Using the chat templates in tasks

Use the templates in tasks. While it would be possible to add an option for using chat templates for every task, I'm not sure this provides much value. The template is most important for tasks that are specifically designed for chat language models like MT-Bench.
