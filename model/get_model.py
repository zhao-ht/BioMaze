import json
import os

from backbone.num_tokens import get_max_context_length
from dataset.pathway_graph_env.pathway_graph_api_utils import load_hsa_graph
from dataset.pathway_graph_env.pathway_graph_env import Pathway_Graph
from model.Graph_Agent.graph_agent_prompt import get_graph_agent_result_parser_prompt
from model.ToG.tog_model import tog
from model.CoT import cot, get_cot_prompt
from model.CoK import cok
from model.Graph_Agent import graph_agent, get_graph_agent_instruction_dict, get_graph_agent_answer_parser, \
    Graph_Agent_Result_Predictor


def load_backbone(model_name, host=None, gpt_request=False):
    if "gpt" in model_name:
        from backbone import gpt
    elif model_name == "Meta-Llama-3.1-405B-Instruct":
        from backbone import samba
    else:
        from backbone import vllm

    if "gpt" in model_name:
        if gpt_request:
            backbone_func = lambda x, **kwargs: gpt.call_chat_gpt_request(
                x, model_name=model_name, **kwargs
            )
        else:
            backbone_func = lambda x, **kwargs: gpt.call_chat_gpt(
                x, model_name=model_name, **kwargs
            )
    elif model_name == "Meta-Llama-3.1-405B-Instruct":
        backbone_func = lambda x, **kwargs: samba.call_samba(
            x, model_name=model_name, **kwargs
        )
    else:
        if 'llama' in model_name.lower():
            stop = '<|eot_id|>'
        else:
            stop = None
        backbone_func = lambda x, **kwargs: vllm.call_vllm(
            x, model=model_name, host=host, stop=stop, **kwargs
        )

    return backbone_func


class model_loader:
    def __init__(self, args):
        self.planning_method = args.planning_method
        self.dataset_name = args.dataset_name

        # load backbone function
        self.backbone_func = load_backbone(
            args.model_name, args.host
        )

        # default result recoder
        self.result_recoder = None

        if args.planning_method in ['cot']:
            if args.eval_save_path is None:
                args.eval_save_path = f"eval_results/{args.planning_method}.{args.model_name}/{args.enable_cot}_{args.in_context_num}_{args.temperature}/{args.dataset_name}.{args.split_dataset_num}_split_{args.split_file}.{args.exp_id}.{args.distributed_id}.csv"

            parsering_func = get_graph_agent_answer_parser(args.answer_type)
            prompt_dict = get_cot_prompt(args.dataset_name, args.answer_type, args.enable_cot)
            with open(os.path.join('model', 'CoT', 'in_context_example',
                                   'in_context_example_{}_{}.json'.format(args.task_type,
                                                                          args.enable_cot))) as f:
                in_context_examples = json.load(f)
            self.itf = cot(backbone_func=self.backbone_func, enable_cot=args.enable_cot,
                           sys_prompt='',
                           prompt_dict=prompt_dict, answer_type=args.answer_type, examples=in_context_examples,
                           in_context_num=args.in_context_num, temperature=args.temperature,
                           parsering_func=parsering_func)

        elif args.planning_method in ['tog']:
            all_hsa_graph = load_hsa_graph(os.path.join('dataset', 'pathway_graph_env',
                                                        "all_hsa_graph.graphml"))
            with open(os.path.join('dataset', 'pathway_graph_env',
                                   'overall_entries.json'), 'r') as read_file:
                all_entry = json.load(read_file)

            # Result parser
            parsering_func = get_graph_agent_answer_parser(args.answer_type)
            prompt_dict = get_graph_agent_result_parser_prompt(args.dataset_name, args.answer_type)
            with open(os.path.join('model', 'Graph_Agent',
                                   'graph_agent_result_parser_example_{}.json'.format(args.task_type))) as f:
                in_context_examples = json.load(f)
            result_parser = Graph_Agent_Result_Predictor(backbone_func=self.backbone_func,
                                                         model_name=args.model_name,
                                                         cot_func=None,
                                                         cot_merge_method=None,
                                                         prompt_dict=prompt_dict,
                                                         in_context_examples=in_context_examples,
                                                         answer_type=args.answer_type,
                                                         answer_method=args.answer_method,
                                                         remove_uncertainty=args.remove_uncertainty,
                                                         uncertainty_query=False,
                                                         max_context_length=get_max_context_length(
                                                             args.model_name), max_length=1024,
                                                         temperature=args.temperature_reasoning,
                                                         parsering_func=parsering_func)

            self.itf = tog(backbone_func=self.backbone_func,
                           all_hsa_graph=all_hsa_graph,
                           all_entity=all_entry,
                           result_parser=result_parser,
                           answer_type=args.answer_type,
                           max_context_length=get_max_context_length(
                               args.model_name),
                           max_length=args.max_length,
                           temperature_exploration=args.temperature_exploration,
                           temperature_reasoning=args.temperature_reasoning,
                           width=args.width,
                           depth=args.depth,
                           remove_unnecessary_rel=args.remove_unnecessary_rel,
                           num_retain_entity=args.num_retain_entity,
                           prune_tools=args.prune_tools
                           )

        elif args.planning_method in ['cok']:
            if args.eval_save_path is None:
                args.eval_save_path = f"eval_results/{args.planning_method}.{args.model_name}/{args.max_pieces}_{args.in_context_num}_{args.temperature}/{args.dataset_name}.{args.split_dataset_num}_split_{args.split_file}.{args.exp_id}.{args.distributed_id}.csv"

            parsering_func = get_graph_agent_answer_parser(args.answer_type)
            prompt_dict = get_cot_prompt(args.dataset_name, args.answer_type, True)
            with open(os.path.join('model', 'CoT', 'in_context_example',
                                   'in_context_example_{}_{}.json'.format(args.task_type,
                                                                          True))) as f:
                in_context_examples = json.load(f)

            all_hsa_graph = load_hsa_graph(os.path.join('dataset', 'pathway_graph_env',
                                                        "all_hsa_graph.graphml"))
            with open(os.path.join('dataset', 'pathway_graph_env',
                                   'overall_entries.json'), 'r') as read_file:
                all_entry = json.load(read_file)

            self.itf = cok(backbone_func=self.backbone_func, model_name=args.model_name, all_hsa_graph=all_hsa_graph,
                           all_entity=all_entry,
                           sys_prompt='',
                           prompt_dict=prompt_dict, answer_type=args.answer_type,
                           max_pieces=args.max_pieces,
                           max_context_length=get_max_context_length(
                               args.model_name),
                           max_length=args.max_length,
                           examples=in_context_examples,
                           in_context_num=args.in_context_num, temperature=args.temperature,
                           parsering_func=parsering_func)


        elif args.planning_method in ['graph_agent']:
            if args.eval_save_path is None:
                args.eval_save_path = f"eval_results/{args.planning_method}.{args.model_name}/{args.answer_method}_{args.remove_uncertainty}_{args.uncertainty_query}_{args.temperature}_{args.cot_merge_method}_{args.max_steps}/{args.dataset_name}.{args.split_dataset_num}_split_{args.split_file}.{args.exp_id}.{args.distributed_id}.csv"

            # Result parser
            if args.cot_merge_method:
                parsering_func = get_graph_agent_answer_parser(args.answer_type)
                prompt_dict = get_cot_prompt(args.dataset_name, args.answer_type, True)
                cot_func = cot(backbone_func=self.backbone_func, enable_cot=True,
                               sys_prompt='',
                               prompt_dict=prompt_dict, answer_type=args.answer_type, examples=[],
                               in_context_num=0, temperature=args.temperature,
                               parsering_func=parsering_func)
            else:
                cot_func = None

            parsering_func = get_graph_agent_answer_parser(args.answer_type)
            prompt_dict = get_graph_agent_result_parser_prompt(args.dataset_name, args.answer_type)
            with open(os.path.join('model', 'Graph_Agent',
                                   'graph_agent_result_parser_example_{}.json'.format(args.task_type))) as f:
                in_context_examples = json.load(f)

            result_parser = Graph_Agent_Result_Predictor(backbone_func=self.backbone_func,
                                                         model_name=args.model_name,
                                                         cot_func=cot_func,
                                                         cot_merge_method=args.cot_merge_method,
                                                         prompt_dict=prompt_dict,
                                                         in_context_examples=in_context_examples,
                                                         answer_type=args.answer_type,
                                                         answer_method=args.answer_method,
                                                         remove_uncertainty=args.remove_uncertainty,
                                                         uncertainty_query=args.uncertainty_query,
                                                         max_context_length=get_max_context_length(
                                                             args.model_name), max_length=1024,
                                                         temperature=args.temperature,
                                                         parsering_func=parsering_func)

            # Main agent
            init_prompt_dict, in_context_examples = get_graph_agent_instruction_dict(args.answer_type)
            env = Pathway_Graph(
                hsa_graph_path=os.path.join('dataset', 'pathway_graph_env',
                                            "all_hsa_graph.graphml"),
                entry_path=os.path.join('dataset', 'pathway_graph_env',
                                        'overall_entries.json'))
            self.itf = graph_agent(
                backbone_func=self.backbone_func,
                model_name=args.model_name,
                env=env,
                max_steps=args.max_steps,
                memory_size=args.memory_size,
                init_prompt_dict=init_prompt_dict,
                in_context_number=len(in_context_examples),
                in_context_example=in_context_examples,
                max_context_length=get_max_context_length(
                    args.model_name),
                temperature=args.temperature,
                allow_history_cut=args.allow_history_cut,
                chat_style=True,
                result_parser=result_parser
            )
        else:
            raise ValueError(
                "Unimplied method name {}".format(args.planning_method))

    def learn(self, data, evaluator):

        raise ValueError(
            "Calling learning of unimplied method name {}".format(
                self.planning_method
            )
        )

    def __call__(self, data):
        print(data["input"])
        print(data['answer'])
        if self.planning_method in ["cot"]:
            return self.itf(data["input"])
        elif self.planning_method in ['tog']:
            return self.itf(data["input"])
        elif self.planning_method in ['cok']:
            return self.itf(data["input"])
        elif self.planning_method in ['graph_agent']:
            return self.itf(data["input"])
        else:
            raise ValueError(
                "Calling unimplied method name {}".format(self.planning_method)
            )
