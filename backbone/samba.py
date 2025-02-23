import copy
import os
import time

import numpy as np
import openai
from openai import OpenAI


def call_samba(
        messages,
        hist_messages=None,
        model_name="Meta-Llama-3.1-405B-Instruct",
        stop=None,
        temperature=0.0,
        top_p=1.0,
        max_tokens=None,
        time_limit_query=200,
        engine_flag=False,
        use_client=True
):
    key_file = "samba_keys_3.txt"
    if os.path.exists(key_file):
        with open(key_file) as f:
            gpt_api_key = f.read()
    # SAMBANOVA_API_URL = "https://fast-api.snova.ai/v1"
    SAMBANOVA_API_URL = "https://api.sambanova.ai/v1"

    if use_client:
        client = OpenAI(
            api_key=gpt_api_key,
            base_url=SAMBANOVA_API_URL,
        )
    else:
        openai.api_key = gpt_api_key
        openai.api_base = SAMBANOVA_API_URL

    messages = copy.deepcopy(messages)
    hist_messages = copy.deepcopy(hist_messages)
    if hist_messages is not None:
        messages = hist_messages + messages

    wait = 3
    call_time = 0
    save_message = True
    while call_time < 10000:
        try:
            if use_client:
                ans = client.chat.completions.create(model=model_name,
                                                     max_tokens=max_tokens,
                                                     stop=stop,
                                                     messages=messages,
                                                     temperature=temperature,
                                                     top_p=top_p,
                                                     stream=True,
                                                     # n=1)
                                                     )
            else:
                ans = openai.ChatCompletion.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    stop=stop,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True,
                    # n=1
                )
            if use_client:
                content = ""
                for chunk in ans:
                    content += chunk.choices[0].delta.content or ""
            else:
                content = ""
                for chunk in ans:
                    content += chunk.choices[0].delta.content or ""
            if content is None:
                print('Warning! Samba None response.')
                content = ''
            messages.append(
                {"role": "assistant", "content": content}
            )
            # time.sleep(10 + 5 * np.random.rand())
            return content, messages
        except Exception as e:
            print(e)
            wait = 10 + 5 * np.random.rand()
            wait = min(wait, 300)
            time.sleep(wait)
            # time.sleep(np.random.rand())
            wait += np.random.rand()
            call_time += 1
    raise RuntimeError("Failed to call samba")
