import copy
import random
import re
from typing import Optional, Mapping, Any

from backbone.num_tokens import num_tokens_from_messages
from sup_func.sup_func import printc


def postprocess_fn(generation: str) -> str:
    generation = generation.lstrip()
    # Regex pattern to find "Answer:" or "Action:"
    pattern_answer_action = r"(Answer:|Action:)"
    matches_answer_action = list(re.finditer(pattern_answer_action, generation))

    # If no matches or only one match of Answer/Action, return the original string
    if len(matches_answer_action) <= 1:
        return generation

    # Get the index of the start of the second match of Answer/Action
    second_match_start = matches_answer_action[1].start()

    # Trim the string to end before the second match of Answer/Action
    trimmed_generation = generation[:second_match_start].strip()

    # Find the index of the end of the first "Action:"
    first_action_end_index = trimmed_generation.find("Action:") + len("Action:")
    # Check for the next occurrence of "End Action" after the first "Action:"
    end_action_index = trimmed_generation.find("End Action", first_action_end_index)
    # Determine where to trim the string
    if end_action_index != -1:
        # Trim the string to the determined index
        trimmed_generation = trimmed_generation[:end_action_index + len("End Action")].strip()

    # Check for the next occurrence of "Thought:" after the first "Action:"
    next_thought_index = trimmed_generation.find("Thought:", first_action_end_index)
    if next_thought_index != -1:
        trimmed_generation = trimmed_generation[:next_thought_index].strip()

    return trimmed_generation


def parse_generation(generation: str) -> Optional[Mapping[str, Any]]:
    if "Answer:" in generation and "Action:" in generation:
        return {
            "type": "invalid",
            "content": generation,
            "extra_info": "Invalid generation. Your output should contain either 'Action:' or 'Answer:', but not both."
        }

    if "Answer:" in generation:
        # find the first answer
        if generation.count("Answer:") > 1:
            # get the first answer
            answer = generation.split("Answer:")[1].strip()
            extra_info = "You have output more than one answer. Only the first answer will be used."
        else:
            answer = generation[generation.find("Answer:") + len("Answer:"):].strip()
            extra_info = None
        return {
            "type": "answer",
            "content": answer,
            "extra_info": extra_info
        }
    elif "Action:" in generation:
        if generation.count("Action:") > 1:
            action = generation.split("Action:")[1].lstrip()
            extra_info = "You have output more than one action. Only the first action will be used."
        else:
            action = generation[generation.find("Action:") + len("Action:"):].lstrip()
            extra_info = None

        if "End Action" in action:  # remove the "End Action" part
            action = action[:action.find("End Action")]

        return {
            "type": "action",
            "content": action,
            "extra_info": extra_info
        }
    else:
        return {
            "type": "invalid",
            "content": generation,
            "extra_info": "Invalid generation. Your output should contain either 'Action:' or 'Answer:'"
        }


class graph_agent:
    def __init__(
            self,
            backbone_func,
            model_name,
            env,
            max_steps,
            memory_size,
            init_prompt_dict,
            in_context_number=0,
            in_context_example=[],
            max_context_length=8192,
            temperature=0.0,
            allow_history_cut=False,
            chat_style=True,
            result_parser=None
    ):
        self.backbone_func = backbone_func
        self.model_name = model_name
        self.env = env

        self.max_steps = max_steps

        self.init_prompt_dict = init_prompt_dict
        self.in_context_number = in_context_number
        self.in_context_example = in_context_example
        self.split = {
            "example": [""],
            "text": [""],
            "rule": [""],
            "system_msg": [""],
            "instruction": [""],
            "goal": [""],
        }

        if memory_size is None:
            self.memory_size = max_steps + 10
        else:
            self.memory_size = memory_size

        self.max_context_length = max_context_length
        self.temperature = temperature

        self.allow_history_cut = allow_history_cut
        self.chat_style = chat_style
        self.result_parser = result_parser

    def make_instruction(self, goal):
        instruction = self.init_prompt_dict["instruction"]
        examples = random.sample(self.in_context_example, self.in_context_number)

        query = ""
        query += instruction.strip()
        memory = [
            {
                "role": "user",
                "content": query
            },
        ]
        if len(examples) > 0:
            memory[-1]['content'] += "\nHere are examples:\n"
            for example in examples:
                assert memory[-1]['role'] == 'user'
                memory[-1]['content'] += example[0].strip()
                for step in example[1:]:
                    assert isinstance(step, list)
                    assert len(step) == 2
                    memory.append({'role': 'assistant', 'content': step[0].strip()})
                    memory.append({'role': 'user', 'content': step[1].strip()})
                memory[-1][
                    'content'] += "\n\n\nNow, let's move on to the next task. The instructions for this task are the same as the previous ones.\n"
        assert memory[-1]['role'] == 'user'
        memory[-1][
            'content'] += "\nPlease explore pathways to find relevant information regarding the following question: " + goal.strip()

        return memory

    def get_instruction_with_history(
            self, memory
    ):
        messages = copy.deepcopy(memory)
        return messages

    def history_exceeded(self, memory):
        messages = self.get_instruction_with_history(memory)
        return num_tokens_from_messages(messages, self.model_name) >= self.max_context_length - 120

    def run(self, memory, temperature=0):
        messages = self.get_instruction_with_history(memory)

        human_do = False
        if not human_do:
            generation, _ = self.backbone_func(messages, temperature=temperature)

        action_type, action, extra_info = self.action_parser(generation)

        return generation, action_type, action, extra_info

    @staticmethod
    def action_parser(generation):
        # generation = postprocess_fn(generation)
        parsed = parse_generation(generation)
        action_type = parsed["type"]
        action = parsed["content"]
        if "extra_info" in parsed and parsed["extra_info"] is not None:
            extra_info = parsed["extra_info"]
        else:
            extra_info = ""
        return action_type, action, extra_info

    @staticmethod
    def observation_wrapper(obs_list, history_item_num):
        assert isinstance(obs_list, list)
        content = '\n'.join(['{}) {}'.format(ind, item) for ind, item in
                             zip(range(history_item_num,
                                       history_item_num + len(obs_list)),
                                 obs_list)])
        return content.strip()

    def execute_action(self, env, action_type, action, extra_info, len_pathway_history, raise_error=False):
        is_done = False

        if action_type == "action":
            execution_result, info = env.step(action)
            execution_result_str = env.res_dict_list_to_string(execution_result)
            content = self.observation_wrapper(execution_result_str, len_pathway_history)
            if info != '':
                content += info.strip() + '\n'
            if extra_info != "":
                content += "\n*Extra reminder: " + extra_info

        elif action_type == "answer":
            execution_result = []
            content = 'You finished the task.'
            is_done = True
        else:
            execution_result = []
            assert action_type == 'invalid'
            content = extra_info

        return env, execution_result, content, is_done

    def update(self, memory, generation, action_type, action, state):
        if self.chat_style:
            memory.append({
                "role": "assistant",
                "content": generation
            })
            memory.append({
                "role": "user",
                "content": state
            })
        else:
            memory.append((action_type, action))
            memory.append(("Observation", state))

        return memory

    @staticmethod
    def print_history(memory):
        printc("Instruction:\n" + memory[0]['content'], 'yellow')
        if len(memory) >= 1:
            for step in memory[1:]:
                if step['role'] == 'assistant':
                    printc('Action: ' + step['content'], 'green')
                else:
                    assert step['role'] == 'user'
                    printc('State: ' + step['content'], 'blue')

    def __call__(self, problem_input, **kwargs):
        env = self.env
        env.reset()

        goal = problem_input

        # if self.chat_style:
        memory = self.make_instruction(goal)
        steps = 0
        self.print_history(memory)

        done = False

        for step_id in range(self.max_steps):
            if (not self.allow_history_cut) and self.history_exceeded(memory):
                printc("Stopped due to context length limitation", 'red')
                pathway_history = env.summarize_history(key_words_list=[problem_input])
                final_result = self.result_parser(problem_input, env.res_dict_list_to_string(pathway_history,
                                                                                             key_words_list=[
                                                                                                 problem_input]))
                return {
                    "res": final_result['res'],
                    "gen": {
                        "steps": step_id + 1,
                        "history": memory,
                        "pathway_history": pathway_history,
                        "final_reason": final_result['gen'],
                        "log_infos": "context length limitation {} exceeded".format(self.max_context_length)
                    },
                }

            generation, action_type, action, extra_info = self.run(memory)
            printc("{} {}:{}".format(action_type, step_id, generation), 'green')
            env, execution_result, state, done = self.execute_action(
                env, action_type, action, extra_info, len(env._get_pathway_history())
            )
            printc("State {}: {}".format(step_id, state), 'blue')
            print(
                "Step {}: Is done: {}".format(
                    step_id, done
                )
            )

            memory = self.update(memory, generation, action_type, action, state)
            steps += 1

            if done:
                pathway_history = env.summarize_history(key_words_list=[problem_input])
                final_result = self.result_parser(problem_input, env.res_dict_list_to_string(pathway_history,
                                                                                             key_words_list=[
                                                                                                 problem_input]))
                return {
                    "res": final_result['res'],
                    "gen": {
                        "steps": step_id + 1,
                        "history": memory,
                        "pathway_history": pathway_history,
                        "final_reason": final_result['gen'],
                        "log_infos": "agent finished the task"
                    },
                }

        pathway_history = env.summarize_history(key_words_list=[problem_input])
        final_result = self.result_parser(problem_input, env.res_dict_list_to_string(pathway_history,
                                                                                     key_words_list=[
                                                                                         problem_input]))
        return {
            "res": final_result['res'],
            "gen": {
                "steps": self.max_steps,
                "history": memory,
                "pathway_history": pathway_history,
                "final_reason": final_result['gen'],
                "log_infos": "step limitation {} exceeded".format(self.max_steps)
            },
        }
