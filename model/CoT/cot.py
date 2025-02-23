import random


class cot:
    def __init__(self, backbone_func, enable_cot, sys_prompt, prompt_dict, answer_type, examples, in_context_num,
                 temperature,
                 parsering_func=None, reasoning_only=False):
        self.backbone_func = backbone_func
        self.enable_cot = enable_cot
        self.sys_prompt = sys_prompt
        self.prompt_dict = prompt_dict
        self.answer_type = answer_type
        self.examples = examples
        self.in_context_num = in_context_num
        self.temperature = temperature
        self.parsering_func = parsering_func
        self.reasoning_only = reasoning_only

    def __call__(self, problem_text):
        print(problem_text)
        in_context_examples = random.sample(self.examples, self.in_context_num)
        examples = ''
        for pair in in_context_examples:
            examples += 'Question: {}\n\nAnswer: {}\n\n\n'.format(pair['Q'], pair['A'])
        prompt = self.prompt_dict['query'].strip().strip(
            '\n') + '\n\n\n{examples}Question: {question}\n\nAnswer: '
        query = prompt.format(examples=examples, question=problem_text)
        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": query}
        ]
        gen, messages = self.backbone_func(messages, temperature=self.temperature)
        print(gen)
        if self.answer_type != 'reasoning' and not self.reasoning_only:
            if self.enable_cot:
                messages.append({"role": "user", "content": self.prompt_dict['ans'].strip() + '\n' + problem_text})
                gen_ans, _ = self.backbone_func(messages, temperature=self.temperature)
                if self.parsering_func is not None:
                    res = self.parsering_func(gen_ans)
                else:
                    res = gen_ans
                print(gen_ans)
                print(res)
            else:
                gen_ans = gen
                if self.parsering_func is not None:
                    res = self.parsering_func(gen)
                else:
                    res = gen
        else:
            gen_ans = gen
            res = gen

        return {'res': res, "gen": {"gen": gen, 'gen_ans': gen_ans, 'messages': messages}}
