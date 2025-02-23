import copy
import random

from backbone.num_tokens import num_tokens_from_messages, num_tokens_string
from sup_func.sup_func import printc
from .graph_agent import graph_agent
from dataset.pathway_graph_env.pathway_graph_api_utils import length_control


class Graph_Agent_Result_Predictor:
    def __init__(self, backbone_func, model_name, cot_func, cot_merge_method, prompt_dict, in_context_examples,
                 answer_type,
                 answer_method,
                 remove_uncertainty,
                 uncertainty_query,
                 max_context_length,
                 max_length, temperature,
                 parsering_func,
                 ):
        self.backbone_func = backbone_func
        self.model_name = model_name
        self.cot_func = cot_func
        self.cot_merge_method = cot_merge_method
        self.prompt_dict = prompt_dict
        self.in_context_examples = in_context_examples
        self.answer_type = answer_type
        self.answer_method = answer_method
        self.remove_uncertainty = remove_uncertainty
        self.uncertainty_query = uncertainty_query
        self.max_context_length = max_context_length
        self.parsering_func = parsering_func
        self.max_length = max_length
        self.temperature = temperature

        assert self.cot_merge_method in ['uncertain', None]

    @staticmethod
    def clean_history(messages):
        messages = copy.deepcopy(messages)
        messages_clean = []
        for ind in range(1, len(messages), 2):
            assert messages[ind]["role"] == "assistant"
            assert messages[ind + 1]["role"] == "user"
            action_type, action, extra_info = graph_agent.action_parser(messages[ind]['content'])
            if action_type == 'action':
                messages_clean.append({"action": action, 'obs': messages[ind + 1]['content'].replace('\n\n', '\n')})
        return messages_clean

    @staticmethod
    def wrap_history(messages):
        message_str = "\n".join([(item['action'].strip() + '\n' + item['obs'].strip()).strip() for item in messages])
        return message_str

    @staticmethod
    def random_drop(lst):
        lst = copy.deepcopy(lst)
        index_to_remove = random.randint(0, len(lst) - 1)
        del lst[index_to_remove]
        return lst

    @staticmethod
    def uncertainty_response_detect(gen):
        # Uncertain is the standard uncertain response.
        uncertain_key_words = ['uncertain',
                               'cannot be determined',
                               'unclear', 'not clear', 'no clear', 'not directly clear', 'cannot determine',
                               'cannot infer', 'cannot answer',
                               'further research', 'further study', 'inconclusive',
                               'cannot be inferred', 'no specific',
                               # 'no direct information',
                               # 'no pathway', 'pathways do not', 'none of the pathway',
                               # 'no specific pathway', 'no specific pathways', 'no direct pathway',
                               # 'unfortunately', 'not possible',
                               ]
        conclusion_key_words = ['it is possible', 'may']
        detected = False
        for keyword in uncertain_key_words:
            if keyword.lower() in gen.lower():
                has_conclusion = False
                for con_keyword in conclusion_key_words:
                    if con_keyword.lower() in gen.lower():
                        has_conclusion = True
                if not has_conclusion:
                    printc('Uncertain sentence with {} detected.'.format(keyword), 'red')
                    detected = True
                    break
        return detected

    def uncertainty_response_judge(self, messages):
        messages = copy.deepcopy(messages)
        assert messages[-1]['role'] == 'assistant'

        if self.uncertainty_query:
            messages.append(
                {"role": "user", "content": self.prompt_dict['uncertain_judge'].strip()})
            gen_unc, _ = self.backbone_func(messages, temperature=self.temperature)
            printc(gen_unc, 'green')
            is_uncertain = self.uncertainty_response_detect(gen_unc)
            printc(is_uncertain, 'green')
            return is_uncertain, gen_unc
        else:
            return self.uncertainty_response_detect(messages[-1]['content']), messages[-1]['content']

    @staticmethod
    def remove_uncertain_sentence(response, depth=2):
        sentences = response.strip().rsplit('.')

        last_nonempty_part = -2 if sentences[-1] == '' else -1
        uncertain_flag = False
        res = response
        uncertain_res = ''
        if len(sentences) <= abs(last_nonempty_part):
            pass
        else:
            for ind in range(depth):
                if abs(last_nonempty_part) + ind >= len(sentences):
                    break
                last_sentence = sentences[last_nonempty_part - ind]
                detected = Graph_Agent_Result_Predictor.uncertainty_response_detect(last_sentence)
                if not detected:
                    break
                else:
                    uncertain = True
                    res = '.'.join(sentences[:(last_nonempty_part - ind)]) + '.'
                    uncertain_res = '.'.join(sentences[(last_nonempty_part - ind):]) + '.'
                    printc(res, 'cyan')
        return uncertain_flag, res, uncertain_res

    def history_cot(self, question, history):
        prompt = self.prompt_dict['query'].strip()
        for item in self.in_context_examples:
            prompt += '\n' + '\nQuestion: ' + item['Q'].strip() + "\nPathways: " + '\n' + item[
                'P'].strip() + '\nAnswer: ' + \
                      item[
                          'A'].strip()

        prompt_basic_length = num_tokens_string(prompt.strip() + question.strip(), self.model_name)
        exceed, reduced_history = length_control(history,
                                                 self.max_context_length - self.max_length - prompt_basic_length,
                                                 [question])

        prompt = prompt.strip() + '\n' + '\nQuestion: ' + question.strip() + "\nPathways: " + '\n' + graph_agent.observation_wrapper(
            reduced_history, 0).strip() + '\nAnswer: '
        messages = [
            {"role": "user", "content": prompt},
        ]

        max_token = self.max_context_length - num_tokens_from_messages(messages, self.model_name) - 150
        if self.answer_type != 'reasoning':
            max_token -= num_tokens_string(self.prompt_dict['ans'].strip() + '\n' + question.strip(), self.model_name)
        gen, messages = self.backbone_func(messages, temperature=self.temperature,
                                           max_tokens=max_token)

        print(graph_agent.observation_wrapper(
            reduced_history, 0))
        printc(gen, 'magenta')
        return gen, messages, reduced_history

    def cot_get_answer(self, question, gen, messages):
        messages = copy.deepcopy(messages)
        if self.answer_type != 'reasoning':
            if self.answer_method == 'conclusion':
                uncertain_flag, clean_gen, uncertain_sent = self.remove_uncertain_sentence(gen)
                if self.remove_uncertainty:
                    messages[-1]['content'] = clean_gen
                else:
                    messages[-1]['content'] = gen
                messages.append(
                    {"role": "user", "content": self.prompt_dict['ans'].strip() + '\n' + question.strip()})
                gen_ans, _ = self.backbone_func(messages, temperature=self.temperature)
                if self.parsering_func is not None:
                    res = self.parsering_func(gen_ans)
                else:
                    res = gen_ans
                printc(gen_ans, 'magenta')
                gen = {'gen': gen, 'clean_gen': clean_gen, 'uncertain_sent': uncertain_sent, 'gen_ans': gen_ans,
                       'messages': messages}
            else:
                assert self.answer_method == 'direct'
                if self.parsering_func is not None:
                    res = self.parsering_func(gen)
                else:
                    res = gen
                gen = {'gen': gen}
        else:
            res = gen
            gen = {'gen': gen, 'messages': messages}
        return {'res': res, "gen": gen}

    def __call__(self, question, history):
        # return dict {'res': res, "gen": gen}, where res is the string result, and gen is the dict of process record

        if self.cot_merge_method == 'uncertain':
            gen, messages, reduced_history = self.history_cot(question, history)
            uncertain, uncertain_cont = self.uncertainty_response_judge(messages)

            if not uncertain:
                res = self.cot_get_answer(question, gen, messages)
                res['gen']['uncertain'] = False
            else:
                cot_res_dict = self.cot_func(question)

                messages = cot_res_dict['gen']['messages'][0:2]
                cot_gen = cot_res_dict['gen']['gen']
                res = self.cot_get_answer(question, cot_gen, messages)

                res['gen']['gen'] = gen
                res['gen']['cot_gen'] = cot_gen
                res['gen']['cot_res'] = cot_res_dict['res']
                res['gen']['uncertain'] = True

            res['gen']['uncertain_cont'] = uncertain_cont
            res['gen']['reduced_history'] = reduced_history

        else:
            gen, messages, reduced_history = self.history_cot(question, history)
            uncertain, uncertain_cont = self.uncertainty_response_judge(messages)
            res = self.cot_get_answer(question, gen, messages)
            res['gen']['uncertain'] = uncertain
            res['gen']['uncertain_cont'] = uncertain_cont
        return res


def judge_parser(res):
    res = res.lower().strip().strip('.')
    if res == 'yes' or ('yes' in res and 'guess' in res):
        return 'Yes'
    elif res == 'no' or ('no' in res and 'guess' in res):
        return 'No'
    else:
        printc('Answer format error, return No as default: {}'.format(res), 'red')
        if 'yes' in res:
            return 'Yes'
        else:
            return 'No'


def get_graph_agent_answer_parser(answer_type):
    if answer_type == 'judge':
        return judge_parser
    else:
        return None
