import json
import traceback

from dataset.pathway_graph_env.pathway_graph_api import get_pathway_api
from dataset.pathway_graph_env.pathway_graph_api_utils import load_hsa_graph, get_entries_retriever_value, \
    get_triples_retriever_value, get_nodes_retriever_value, get_edge_retriever_value, ItemRetriever, illustrate_triples, \
    compact_triple_to_str
from sup_func.sup_func import execute


class Pathway_Graph:
    def __init__(self, hsa_graph_path, entry_path, wrapper=''
                 ):

        all_hsa_graph = load_hsa_graph(hsa_graph_path)
        with open(entry_path, 'r') as read_file:
            all_entry = json.load(read_file)
        self.all_hsa_graph = all_hsa_graph
        self.all_entry = all_entry
        self.entity_search_engine = ItemRetriever(*get_entries_retriever_value(all_entry.keys(), all_entry))
        self.relation_search_engine = ItemRetriever(
            *get_triples_retriever_value(all_hsa_graph.edges, all_hsa_graph, all_entry))
        self.node_search_engine = ItemRetriever(*get_nodes_retriever_value(all_hsa_graph.nodes, all_entry))
        self.edge_search_engine = ItemRetriever(
            *get_edge_retriever_value(all_hsa_graph.edges, all_hsa_graph, all_entry))

        self.wrapper = wrapper

        self.reset()

    def _get_info(self):
        infos = dict()
        infos["goal"] = self.goal
        infos["states"] = self.states
        infos["actions"] = self.actions
        infos["history"] = self._get_history()
        infos["steps"] = self.steps
        infos["state"] = self.states[-1]
        return infos

    def _get_obs(self):
        assert isinstance(self.states[-1][0], list)
        states = self.states[-1][0]
        step_info = self.states[-1][1]
        previous_pathway_history = self._get_pathway_history(execlute_last=True)
        new_pathway = []
        for item in states:
            if item not in previous_pathway_history:
                new_pathway.append(item)
        if len(states) > 0 and len(new_pathway) == 0:
            step_info += '\nNo new pathways were found besides those previously seen.'
        assert previous_pathway_history + new_pathway == self._get_pathway_history()
        return (new_pathway, step_info)

    def _get_goal(self):
        return self.goal

    def _get_history(self):
        history = []
        assert len(self.states) == (len(self.actions) + 1)
        for state, action in zip(self.states[:-1], self.actions):
            history.append(state)
            history.append(action)
        history.append(self.states[-1])
        return history

    def _get_pathway_history(self, execlute_last=False):
        pathway_history = []
        if execlute_last:
            states_list = self.states[:-1]
        else:
            states_list = self.states
        for state in states_list:
            for item in state[0]:
                if item not in pathway_history:
                    pathway_history.append(item)
        return pathway_history

    def _get_api(self):
        tools_env = get_pathway_api(all_hsa_graph=self.all_hsa_graph, all_entry=self.all_entry,
                                    entity_search_engine=self.entity_search_engine,
                                    relation_search_engine=self.relation_search_engine,
                                    node_search_engine=self.node_search_engine,
                                    edge_search_engine=self.edge_search_engine,
                                    pathway_history=self._get_pathway_history(), return_str=False)
        return tools_env

    def reset(self):
        self.goal, self.init_info, = '', 'You can use the Biopathway database.'
        self.states = [([], self.init_info)]  # record a stream of states
        self.actions = []
        self.steps = 0

    def step(self, action, raise_error=False):

        code = "{}\n\nobs,info = {}".format(self.wrapper, action)
        global_vars = {}
        tools_env = self._get_api()
        for tool_name, tool_function in tools_env.items():
            global_vars[tool_name] = tool_function
        try:
            res = execute(code, time_limit_query=3000, global_env=global_vars)
            obs = res['obs']
            info = res['info']
            assert isinstance(obs, list)
        except Exception as e:
            if raise_error:
                raise e
            # traceback_str = traceback.format_exc()
            # obs = traceback_str
            obs = []
            formatted_exception = traceback.format_exception(type(e), e, e.__traceback__)
            filtered_exception = [line for line in formatted_exception if
                                  not (line.strip().startswith("File") or line.strip().startswith("Traceback"))]
            info = 'Error: ' + "".join(filtered_exception)

        self.update(action, (obs, info))
        return self._get_obs()

    def step_by_name(self, action_name, *args, **kwargs):
        tools_env = self._get_api()
        obs, info = tools_env[action_name](*args, **kwargs)
        self.update((action_name, args, kwargs), (obs, info))
        return self._get_obs()

    def update(self, action, state):
        # state is (obs, info). obs format must be list
        assert isinstance(state[0], list)
        self.steps += 1
        self.actions.append(action)
        self.states.append(state)

    def summarize_history(self, key_words_list, target_size=16):
        tools_env = self._get_api()
        obs, info = tools_env['summarize_history_subgraph'](key_words_list=key_words_list, target_size=target_size)
        return obs

    @staticmethod
    def res_dict_list_to_string(res_dict_list, key_words_list=[]):
        res_description_compact = [compact_triple_to_str(res_line, key_words_list=key_words_list)[1]
                                   for
                                   res_line in
                                   res_dict_list]
        return res_description_compact
