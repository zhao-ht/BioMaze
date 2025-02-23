import traceback

import networkx as nx
import numpy as np

from dataset.pathway_graph_env.pathway_graph_api_utils import graph_next_step, graph_previous_step, \
    retrieve_content_by_key_words, illustrate_entries, \
    illustrate_triples, illustrate_detailed_entity, length_control, graph_entity_N_hop_subgraph, \
    get_triples_retriever_value, \
    get_nodes_retriever_value, networkx_to_int64_array, subgraph_with_nodes_edges_subset, sort_graph_dfs, \
    get_edge_retriever_value, graph_statistics, compact_triple_to_str, graph_triple_N_hop_subgraph, recover_triple
from dataset.pathway_graph_env.pcst_retriever import pcst_retrieval, pcst_retrieval_size, plot_pcst_result


def get_pathway_api(all_hsa_graph, all_entry, entity_search_engine, relation_search_engine, node_search_engine,
                    edge_search_engine, pathway_history=[], return_str=False,
                    enable_raise_error=True, result_max_length=-1, do_visualization=False):
    def biopathway_next_step(entity_id, key_words_list=[], topk=8):
        res_triples = graph_next_step(all_hsa_graph, entity_id)
        res_description = illustrate_triples(res_triples, all_hsa_graph, all_entry)
        if len(res_description) > 0 and len(key_words_list) > 0:
            _, retriever_value = get_triples_retriever_value(res_triples, all_hsa_graph, all_entry)
            res_description, _, _ = retrieve_content_by_key_words(retriever_value, key_words_list, topk,
                                                                  custom_index=res_description)
            # res_description = [res_description[index] for index in res_index]
        if len(res_description) > 0:
            info = ''
            if return_str:
                res = [compact_triple_to_str(res_line, key_words_list=key_words_list)[1] for res_line in
                       res_description]
            else:
                res = res_description
            if result_max_length > 0:
                exceed, res = length_control(res, result_max_length)
                if exceed:
                    info = 'The result exceeds the maximum length allowed. Only display partial items. Please increase your keywords.'
        else:
            res = []
            info = 'No downstream was found for the given entity and keywords.'
        return res, info

    def biopathway_previous_step(entity_id, key_words_list=[], topk=8):
        res_triples = graph_previous_step(all_hsa_graph, entity_id)
        res_description = illustrate_triples(res_triples, all_hsa_graph, all_entry)
        if len(res_description) > 0 and len(key_words_list) > 0:
            _, retriever_value = get_triples_retriever_value(res_triples, all_hsa_graph, all_entry)
            res_description, _, _ = retrieve_content_by_key_words(retriever_value, key_words_list, topk,
                                                                  custom_index=res_description)
        if len(res_description) > 0:
            info = ''
            if return_str:
                res = [compact_triple_to_str(res_line, key_words_list=key_words_list)[1] for res_line in
                       res_description]
            else:
                res = res_description
            if result_max_length > 0:
                exceed, res = length_control(res, result_max_length)
                if exceed:
                    info = 'The result exceeds the maximum length allowed. Only display partial items. Please increase your keywords.'
        else:
            res = []
            info = 'No upstream was found for the given entity and keywords.'
        return res, info

    def biopathway_N_hop_step(entity_id, key_words_list=[], topk=8):
        res_triples, subgraph = graph_entity_N_hop_subgraph(all_hsa_graph, entity_id, 3)
        res_description = illustrate_triples(res_triples, all_hsa_graph, all_entry)
        if len(res_description) > 0 and len(key_words_list) > 0:
            _, retriever_value = get_triples_retriever_value(res_triples, all_hsa_graph, all_entry)
            res_description, _, _ = retrieve_content_by_key_words(retriever_value, key_words_list, topk,
                                                                  custom_index=res_description)
        if len(res_description) > 0:
            info = ''
            if return_str:
                res = [compact_triple_to_str(res_line, key_words_list=key_words_list)[1] for res_line in
                       res_description]
            else:
                res = res_description
            if result_max_length > 0:
                exceed, res = length_control(res, result_max_length)
                if exceed:
                    info = 'The result exceeds the maximum length allowed. Only display partial items. Please increase your keywords.'
        else:
            res = []
            info = 'No N hop triples was found for the given entity and keywords.'
        return res, info

    def retrieve_connected_subgraph(subgraph, key_words_list, target_size=16, num_clusters=1):
        sorted_nodes, sorted_edges, edges_array, node2id_mapping, id2node_mapping = networkx_to_int64_array(
            subgraph)

        _, node_retriever_value = get_nodes_retriever_value(sorted_nodes, all_entry)
        node_res_index, _, node_score = retrieve_content_by_key_words(node_retriever_value, key_words_list,
                                                                      len(sorted_nodes))
        n_prizes = np.zeros(len(sorted_nodes), dtype=float)
        np.put(n_prizes, node_res_index, node_score)

        _, edge_retriever_value = get_edge_retriever_value(sorted_edges, all_hsa_graph, all_entry)
        edge_res_index, _, edge_score = retrieve_content_by_key_words(edge_retriever_value, key_words_list,
                                                                      len(sorted_edges))
        e_prizes = np.zeros(len(sorted_edges), dtype=float)
        np.put(e_prizes, edge_res_index, edge_score)

        (pcst_nodes_index, pcst_edge_index), cost_mid = pcst_retrieval_size(edges_array, len(sorted_nodes),
                                                                            n_prizes, e_prizes,
                                                                            target_size, num_clusters=num_clusters)

        pcst_nodes = [id2node_mapping[node_id] for node_id in pcst_nodes_index]
        pcst_triples = [(id2node_mapping[n1], id2node_mapping[n2]) for n1, n2 in pcst_edge_index]
        pcst_triples = sort_graph_dfs(pcst_triples)
        res_description = illustrate_triples(pcst_triples, all_hsa_graph, all_entry)

        # test the pcst graph's connection
        pcst_subgraph = subgraph_with_nodes_edges_subset(subgraph, pcst_nodes, pcst_triples)
        if num_clusters == 1:
            assert nx.is_weakly_connected(pcst_subgraph)
        else:
            G_undirected = pcst_subgraph.to_undirected()
            num_components = len(list(nx.connected_components(G_undirected)))
            print('Number of clusters: ' + str(num_components))
        # visualize the original retrieve result and pcst graph
        if do_visualization:
            retrieve_nodes = [sorted_nodes[ind] for ind in node_res_index]
            retrieve_edges = [sorted_edges[ind] for ind in edge_res_index]
            retrieve_subgraph = subgraph_with_nodes_edges_subset(subgraph, retrieve_nodes, retrieve_edges)
            node_prizes = {sorted_nodes[id]: score for id, score in enumerate(n_prizes)}
            edge_prizes = {sorted_edges[id]: score for id, score in enumerate(e_prizes)}
            plot_pcst_result(retrieve_subgraph, pcst_subgraph, all_hsa_graph, all_entry, node_prizes, edge_prizes,
                             '{}'.format(key_words_list))

        return pcst_triples

    def search_biopathway_entity_N_hop_subgraph(entity_id, key_words_list=[], target_size=16):
        res_triples, subgraph = graph_entity_N_hop_subgraph(all_hsa_graph, entity_id, 3)
        res_description = illustrate_triples(res_triples, all_hsa_graph, all_entry)
        if len(res_description) > 0 and len(key_words_list) > 0:
            pcst_triples = retrieve_connected_subgraph(subgraph, key_words_list, target_size)
            res_description = illustrate_triples(pcst_triples, all_hsa_graph, all_entry)

        if len(res_description) > 0:
            info = ''
            if return_str:
                res = [compact_triple_to_str(res_line, key_words_list=key_words_list)[1] for res_line in
                       res_description]
            else:
                res = res_description
            if result_max_length > 0:
                exceed, res = length_control(res, result_max_length, key_words_list)
                if exceed:
                    info = 'The result exceeds the maximum length allowed. Only display partial items. Please increase your keywords.'
        else:
            res = []
            info = 'No subgraph was found for the given entity and keywords.'
        return res, info

    def search_biopathway_triple_N_hop_subgraph(history_line_id, key_words_list=[], target_size=16):
        res_line = pathway_history[history_line_id]
        triple = recover_triple(res_line)
        res_triples, subgraph = graph_triple_N_hop_subgraph(all_hsa_graph, triple, 3)
        res_description = illustrate_triples(res_triples, all_hsa_graph, all_entry)
        if len(res_description) > 0 and len(key_words_list) > 0:
            pcst_triples = retrieve_connected_subgraph(subgraph, key_words_list, target_size)
            res_description = illustrate_triples(pcst_triples, all_hsa_graph, all_entry)

        if len(res_description) > 0:
            info = ''
            if return_str:
                res = [compact_triple_to_str(res_line, key_words_list=key_words_list)[1] for res_line in
                       res_description]
            else:
                res = res_description
            if result_max_length > 0:
                exceed, res = length_control(res, result_max_length, key_words_list)
                if exceed:
                    info = 'The result exceeds the maximum length allowed. Only display partial items. Please increase your keywords.'
        else:
            res = []
            info = 'No subgraph was found for the given entity and keywords.'
        return res, info

    def search_biopathway_subgraph(key_words_list, target_size=16, topk=512):
        res_triples, engine_values, _ = relation_search_engine.retrieve(key_words_list, topk)
        subgraph = all_hsa_graph.subgraph(
            np.unique([edge[0] for edge in res_triples] + [edge[1] for edge in res_triples]))
        # For visual check
        stas = graph_statistics(subgraph)
        res_description = illustrate_triples(res_triples, all_hsa_graph, all_entry)
        if len(res_description) > 0 and len(key_words_list) > 0:
            pcst_triples = retrieve_connected_subgraph(subgraph, key_words_list, target_size)
            res_description = illustrate_triples(pcst_triples, all_hsa_graph, all_entry)

        if len(res_description) > 0:
            info = ''
            if return_str:
                res = [compact_triple_to_str(res_line, key_words_list=key_words_list)[1] for res_line in
                       res_description]
            else:
                res = res_description
            if result_max_length > 0:
                exceed, res = length_control(res, result_max_length, key_words_list)
                if exceed:
                    info = 'The result exceeds the maximum length allowed. Only display partial items. Please increase your keywords.'
        else:
            res = []
            info = 'No subgraph was found for given keywords.'
        return res, info

    def search_biopathway_subgraph_global(key_words_list, target_size=16, topk=128, topk_e=512):
        res_description = []
        if len(key_words_list) > 0:
            sorted_nodes, sorted_edges, edges_array, node2id_mapping, id2node_mapping = networkx_to_int64_array(
                all_hsa_graph)

            node_res_entity, _, node_score = node_search_engine.retrieve(key_words_list, topk)
            node_res_index = [node2id_mapping[node] for node in node_res_entity]
            n_prizes = np.zeros(len(sorted_nodes), dtype=float)
            np.put(n_prizes, node_res_index, node_score)

            edge_res_entity, _, edge_score = edge_search_engine.retrieve(key_words_list, topk_e)
            edge_res_index = [sorted_edges.index(edge) for edge in edge_res_entity]
            e_prizes = np.zeros(len(sorted_edges), dtype=float)
            np.put(e_prizes, edge_res_index, edge_score)

            (pcst_nodes_index, pcst_edge_index), cost_mid = pcst_retrieval_size(edges_array, len(sorted_nodes),
                                                                                n_prizes, e_prizes,
                                                                                target_size)

            pcst_nodes = [id2node_mapping[node_id] for node_id in pcst_nodes_index]
            pcst_triples = [(id2node_mapping[n1], id2node_mapping[n2]) for n1, n2 in pcst_edge_index]
            pcst_triples = sort_graph_dfs(pcst_triples)
            res_description = illustrate_triples(pcst_triples, all_hsa_graph, all_entry)

            # test the pcst graph's connection
            pcst_subgraph = subgraph_with_nodes_edges_subset(all_hsa_graph, pcst_nodes, pcst_triples)
            assert nx.is_weakly_connected(pcst_subgraph)

            # visualize the original retrieve result and pcst graph
            if do_visualization:
                retrieve_nodes = [sorted_nodes[ind] for ind in node_res_index]
                retrieve_edges = [sorted_edges[ind] for ind in edge_res_index]
                retrieve_subgraph = subgraph_with_nodes_edges_subset(all_hsa_graph, retrieve_nodes, retrieve_edges)
                node_prizes = {sorted_nodes[id]: score for id, score in enumerate(n_prizes)}
                edge_prizes = {sorted_edges[id]: score for id, score in enumerate(e_prizes)}
                plot_pcst_result(retrieve_subgraph, pcst_subgraph, all_hsa_graph, all_entry, node_prizes, edge_prizes,
                                 '{}'.format(key_words_list))

        if len(res_description) > 0:
            info = ''
            if return_str:
                res = [compact_triple_to_str(res_line, key_words_list=key_words_list)[1] for res_line in
                       res_description]
            else:
                res = res_description
            if result_max_length > 0:
                exceed, res = length_control(res, result_max_length, key_words_list)
                if exceed:
                    info = 'The result exceeds the maximum length allowed. Only display partial items. Please increase your keywords.'
        else:
            res = []
            info = 'No subgraph was found for given keywords.'
        return res, info

    def summarize_history_subgraph(key_words_list, target_size=16):
        if len(pathway_history) > 0:
            subgraph_triples = [recover_triple(res_line) for res_line in pathway_history]
            subgraph = all_hsa_graph.edge_subgraph(subgraph_triples).copy()
            # Convert to undirected graph
            G_undirected = subgraph.to_undirected()
            # Number of connected components
            num_components = max(1, len(list(nx.connected_components(G_undirected))))

            pcst_triples = retrieve_connected_subgraph(subgraph, key_words_list, target_size,
                                                       num_clusters=num_components)
            res_description = illustrate_triples(pcst_triples, all_hsa_graph, all_entry)
        else:
            res_description = []
        if len(res_description) > 0:
            info = ''
            if return_str:
                res = [compact_triple_to_str(res_line, key_words_list=key_words_list)[1] for res_line in
                       res_description]
            else:
                res = res_description
            if result_max_length > 0:
                exceed, res = length_control(res, result_max_length, key_words_list)
                if exceed:
                    info = 'The result exceeds the maximum length allowed. Only display partial items. Please increase your keywords.'
        else:
            res = []
            info = 'No subgraph was found for given keywords.'
        return res, info

    def search_entity(key_words_list, topk=8):
        res_indexs, engine_values, _ = entity_search_engine.retrieve(key_words_list, topk)
        res_description = illustrate_entries(res_indexs, all_entry)
        if len(res_description) > 0:
            info = ''
            if return_str:
                res = [compact_entity_to_str(res_line, key_words_list=key_words_list)[1] for res_line in
                       res_description]
            else:
                res = res_description
            if result_max_length > 0:
                exceed, res = length_control(res, result_max_length)
                if exceed:
                    info = 'The result exceeds the maximum length allowed. Only display partial items. Please increase your keywords.'
        else:
            res = []
            info = 'No item was found for the given keywords.'
        return res, info

    def search_relation(key_words_list, topk=8):
        res_triples, engine_values, _ = relation_search_engine.retrieve(key_words_list, topk)
        res_description = illustrate_triples(res_triples, all_hsa_graph, all_entry)
        if len(res_description) > 0:
            info = ''
            if return_str:
                res = [compact_triple_to_str(res_line, key_words_list=key_words_list)[1] for res_line in
                       res_description]
            else:
                res = res_description
            if result_max_length > 0:
                exceed, res = length_control(res, result_max_length)
                if exceed:
                    info = 'The result exceeds the maximum length allowed. Only display partial items. Please increase your keywords.'
        else:
            res = []
            info = 'No triples was found for the given keywords.'
        return res, info

    def detailed_information(entity_id):
        if entity_id not in all_entry:
            return [], 'Entity id {} is invalid.'.format(entity_id)
        res = illustrate_detailed_entity(entity_id, all_entry[entity_id])
        assert isinstance(res, dict)
        if return_str:
            res = compact_entity_to_str(res)
        return [res], ''

    # When using API individually, this wrapper simulates the error string report in pathway_graph environment
    def try_wrap_function(func):
        def wrapped_func(*args, **kwargs):
            if enable_raise_error:
                return func(*args, **kwargs)
            else:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    obs = []
                    formatted_exception = traceback.format_exception(type(e), e, e.__traceback__)
                    filtered_exception = [line for line in formatted_exception if
                                          not (line.strip().startswith("File") or line.strip().startswith("Traceback"))]
                    info = 'Error: ' + "".join(filtered_exception)

                    return obs, info

        return wrapped_func

    pathway_apis = {'biopathway_next_step': try_wrap_function(biopathway_next_step),
                    'biopathway_previous_step': try_wrap_function(biopathway_previous_step),
                    'biopathway_N_hop_step': try_wrap_function(biopathway_N_hop_step),
                    'summarize_history_subgraph': try_wrap_function(summarize_history_subgraph),
                    'search_biopathway_entity_N_hop_subgraph': try_wrap_function(
                        search_biopathway_entity_N_hop_subgraph),
                    'search_biopathway_triple_N_hop_subgraph': try_wrap_function(
                        search_biopathway_triple_N_hop_subgraph),
                    'search_biopathway_subgraph': try_wrap_function(search_biopathway_subgraph),
                    'search_biopathway_subgraph_global': try_wrap_function(search_biopathway_subgraph_global),
                    'search_entity': try_wrap_function(search_entity),
                    'search_relation': try_wrap_function(search_relation),
                    'detailed_information': try_wrap_function(detailed_information)}

    return pathway_apis
