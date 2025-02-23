import re

from dataset.pathway_graph_env.pathway_graph_api import get_pathway_api
from dataset.pathway_graph_env.pathway_graph_api_utils import ItemRetriever, get_entries_retriever_value
from .tog_pathway_func import *


def remove_repeat(input_list, eval_item=True):
    if eval_item:
        return [eval(item) for item in
                list(set([str(item) for item in input_list]))]
    else:
        return list(set([str(item) for item in input_list]))


def select_unseen(current, seen_set):
    unseen = [item for item in current if str(item) not in seen_set]
    seen_set.update([str(item) for item in current])
    return unseen, seen_set


def get_entries_descriptions_only_entity(all_entry):
    indexes = []
    for key in all_entry:
        if key.startswith('C') or key.startswith('K') or bool(re.match(r'^\d', key)):
            indexes.append(key)
    return get_entries_retriever_value(indexes, all_entry)


class tog:
    def __init__(self, backbone_func, all_hsa_graph, all_entity, result_parser, answer_type,
                 max_context_length,
                 max_length=512,
                 temperature_exploration=0.4, temperature_reasoning=0,
                 width=3, depth=3, remove_unnecessary_rel=True,
                 num_retain_entity=5, prune_tools="llm"):
        self.backbone_func = backbone_func
        self.all_hsa_graph = all_hsa_graph
        self.all_entity = all_entity
        self.search_engine = ItemRetriever(*get_entries_descriptions_only_entity(self.all_entity))

        self.result_parser = result_parser

        self.answer_type = answer_type

        # self.dataset = dataset
        self.max_length = max_length
        self.temperature_exploration = temperature_exploration
        self.temperature_reasoning = temperature_reasoning
        self.width = width
        self.depth = depth
        self.remove_unnecessary_rel = remove_unnecessary_rel

        self.num_retain_entity = num_retain_entity
        self.prune_tools = prune_tools

        self.max_context_length = max_context_length

    def __call__(self, question):

        printc(question, 'cyan')

        pass_steps = False

        all_entity_relations_list = []
        history = []
        all_searched_entities = set()
        all_seen_triples = set()

        pathway_apis = get_pathway_api(self.all_hsa_graph, self.all_entity, entity_search_engine=self.search_engine,
                                       relation_search_engine=None, node_search_engine=None,
                                       edge_search_engine=None,
                                       return_str=False,
                                       enable_raise_error=False)
        topic_entity, info = pathway_apis['search_entity']([question], topk=16)

        if not isinstance(topic_entity, str):
            search_entry_queue = [list(line_dict.keys())[0] for line_dict in topic_entity]
        else:
            search_entry_queue = []
        all_searched_entities.update([str(item) for item in search_entry_queue])

        if len(search_entry_queue) == 0:
            return generate_answer_pathway(question, all_entity_relations_list, 0, self)

        for depth in range(1, self.depth + 1):

            history.append(search_entry_queue)
            printc(str(search_entry_queue), 'blue')

            current_entity_relations_list = []
            for entity in search_entry_queue:
                retrieved_downstream_triples, info = pathway_apis['biopathway_next_step'](entity)
                retrieved_upstream_triples, info = pathway_apis['biopathway_previous_step'](entity)
                retrieved_triples = []
                if not isinstance(retrieved_downstream_triples, str):
                    retrieved_triples += retrieved_downstream_triples
                else:
                    history.append(retrieved_downstream_triples)
                    print(current_entity_relations_list)
                if not isinstance(retrieved_upstream_triples, str):
                    retrieved_triples += retrieved_upstream_triples
                else:
                    history.append(retrieved_upstream_triples)
                    print(current_entity_relations_list)
                current_entity_relations_list += retrieved_triples

            current_entity_relations_list = remove_repeat(current_entity_relations_list)
            current_entity_relations_list, all_seen_triples = select_unseen(current_entity_relations_list,
                                                                            all_seen_triples)

            history.append(current_entity_relations_list)
            printc('\n'.join([str(item) for item in current_entity_relations_list]), 'yellow')

            if len(current_entity_relations_list) == 0 and not pass_steps:
                return half_stop_pathway(question, all_entity_relations_list, depth, self)

            success, pruned_current_entity_relations_list = triple_prune(question, current_entity_relations_list,
                                                                         self)

            history.append(pruned_current_entity_relations_list)
            printc('\n'.join([str(item) for item in pruned_current_entity_relations_list]), 'green')

            all_entity_relations_list += pruned_current_entity_relations_list
            all_entity_relations_list = remove_repeat(all_entity_relations_list)

            next_search_entry_queue = []
            for triple in pruned_current_entity_relations_list:
                for ind in [0, 2]:
                    if isinstance(triple[ind]['ID'], list):
                        next_search_entry_queue += triple[ind]['ID']
                    else:
                        assert isinstance(triple[ind]['ID'], str)
                        next_search_entry_queue.append(triple[ind]['ID'])

            printc(str(next_search_entry_queue), 'green')
            next_search_entry_queue = remove_repeat(
                next_search_entry_queue, eval_item=False)
            next_search_entry_queue, all_searched_entities = select_unseen(next_search_entry_queue,
                                                                           all_searched_entities)
            search_entry_queue = next_search_entry_queue

            # if pass_steps and depth < self.depth and len(search_entry_queue) > 0:
            #     continue

            printc('\n\nQ: ' + question.strip() + "\nKnowledge Triplets: " + '\n'.join(
                [str(item) for item in all_entity_relations_list]), 'magenta')
            # if pass_steps:
            #     return {'res': 'No', 'gen': {}}

            if success:
                stop, results, ans, all_entity_relations_list_reduced = judge_stop_exploration(question,
                                                                                               all_entity_relations_list,
                                                                                               self)
                if stop:
                    printc("ToG stoped at depth %d." % depth, 'bright_green')
                    return {'res': ans, 'gen': {'response': results, 'pathway': all_entity_relations_list,
                                                'sampled_pathway': all_entity_relations_list_reduced, 'depth': depth}}
                    # return generate_answer_pathway(question, all_entity_relations_list, depth, self)
                else:
                    printc("depth %d still not find the answer." % depth, 'bright_green')
                    continue
            else:
                if pass_steps:
                    return {'res': 'No', 'gen': {}}

                return half_stop_pathway(question, all_entity_relations_list, depth, self)

        return generate_answer_pathway(question, all_entity_relations_list, depth, self)
