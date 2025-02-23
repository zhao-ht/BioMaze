pathway_cot_meta = '''You are an expert in biological pathways.'''


def get_basic_cot_prompt(dataset_name):
    return pathway_cot_meta


def get_cot_prompt(dataset_name, answer_type, enable_cot):
    prompt_dict = {}
    prompt_query = get_basic_cot_prompt(dataset_name).strip()
    if enable_cot:
        if answer_type in ['judge', 'choice', 'reasoning']:
            prompt_query += " Let's think step by step."
        else:
            raise ValueError(f"Invalid answer type: {answer_type}")
    else:
        if answer_type in ['judge']:
            prompt_query += " Answer the question with only 'Yes' or 'No'."
        elif answer_type in ['choice']:
            prompt_query += " Respond with only the choice label, such as 'A', 'B', etc."
        elif answer_type in ['reasoning']:
            pass
        else:
            raise ValueError(f"Invalid answer type: {answer_type}")
    prompt_dict['query'] = prompt_query

    if enable_cot:
        if answer_type in ['judge']:
            prompt_ans = "So, answer the question with only 'Yes' or 'No':"
        elif answer_type in ['choice']:
            prompt_ans = "So, answer the question with only the choice label, such as 'A', 'B', etc.:"
        elif answer_type in ['reasoning']:
            prompt_ans = None
        else:
            raise ValueError(f"Invalid answer type: {answer_type}")
    else:
        prompt_ans = None
    prompt_dict['ans'] = prompt_ans

    return prompt_dict
