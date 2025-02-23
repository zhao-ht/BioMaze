import copy
import json
import re

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backbone.num_tokens import num_tokens_string
from collections import Counter


def graph_statistics(graph_data):
    self_loop_edge = list(nx.selfloop_edges(graph_data))
    degree = dict(graph_data.degree())
    isolated = [node for node, degree in degree.items() if degree == 0]
    count = Counter([degree[key] for key in degree])
    weakly_connected_components = nx.weakly_connected_components(graph_data)
    component_sizes = [len(component) for component in weakly_connected_components]
    return len(graph_data.nodes), len(graph_data.edges), self_loop_edge, count, isolated, component_sizes


def clean_graph(graph_data):
    print('Remove self loop edge')
    self_loop_edge = list(nx.selfloop_edges(graph_data))
    for edge in self_loop_edge:
        print(edge)
    graph_data.remove_edges_from(self_loop_edge)

    print('Remove isolated nodes')
    degree = dict(graph_data.degree())
    isolated = [node for node, degree in degree.items() if degree == 0]
    print(len(isolated))
    graph_data.remove_nodes_from(isolated)

    return graph_data


# Basic functions
def load_hsa_graph(file_dr, clean_data=True, frozen=True):
    graph_data = nx.read_graphml(file_dr)
    # Convert the JSON string attributes back to dictionaries
    for u, data in graph_data.nodes(data=True):
        if 'graphics' in data:
            data['graphics'] = json.loads(data['graphics'])

    for u, v, data in graph_data.edges(data=True):
        if 'subtypes' in data:
            data['subtypes'] = json.loads(data['subtypes'])
        if 'hsa' in data:
            data['hsa'] = json.loads(data['hsa'])
        if 'pathway' in data:
            data['pathway'] = json.loads(data['pathway'])
    if clean_data:
        graph_data = clean_graph(graph_data)
    # ensure the graph data is read-only
    if frozen:
        graph_data = nx.freeze(graph_data)
    print(graph_statistics(graph_data))
    return graph_data


def node_match(graph, given_node, exact_match_first=False):
    given_node_pre = []
    for item in given_node:
        if ':' not in item:
            given_node_pre.append(add_pre(item))
        else:
            given_node_pre.append(item)

    matched_graph_node = []
    for node in graph.nodes:
        if exact_match_first and set(node.strip().split(' ')) == set(given_node_pre):
            return [node]
        if len(set(node.strip().split(' ')) & set(given_node_pre)) > 0:
            matched_graph_node.append(node)
    # check for node overlap
    # overlaped = False
    # if len(matched_graph_node) > 1:
    #     for i in range(len(matched_graph_node) - 1):
    #         for j in range(i + 1, len(matched_graph_node)):
    #             set1 = set(matched_graph_node[i].strip().split(' '))
    #             set2 = set(matched_graph_node[j].strip().split(' '))
    #             if set1.issubset(set2) or set2.issubset(set1):
    #                 print('warning: overlapping node matched for given node {}:\n  {}'.format(given_node_pre,
    #                                                                                           matched_graph_node))
    #                 overlaped = True
    #                 break
    #         if overlaped:
    #             break

    return matched_graph_node


def add_pre(later_name):
    if later_name.startswith('hsa'):
        full = 'path:{}'.format(later_name)
    elif later_name.startswith('map'):
        full = 'path:{}'.format(later_name)
    elif later_name.startswith('nt'):
        full = 'network:{}'.format(later_name)
    elif later_name.startswith('ko'):
        full = 'path:{}'.format(later_name)

    elif later_name.startswith("C"):
        full = 'cpd:{}'.format(later_name)
    elif later_name.startswith('K'):
        full = 'ko:{}'.format(later_name)
    elif later_name.startswith('G'):
        full = 'gl:{}'.format(later_name)
    elif later_name.startswith('D'):
        full = 'dr:{}'.format(later_name)
    # elif later_name.startswith('N') or later_name.startswith('H'):
    #     # not in hsa graph
    #     full = later_name
    elif bool(re.match(r'^\d', later_name)):
        full = 'hsa:{}'.format(later_name)
    else:
        raise ValueError('Warning: Invalid id {}'.format(later_name))
        full = later_name
    return full


# Graph basic functions

def networkx_to_int64_array(graph):
    sorted_nodes = list(graph.nodes)
    sorted_edges = list(graph.edges)
    node2id_mapping = {node: i for i, node in enumerate(sorted_nodes)}
    id2node_mapping = {i: node for i, node in enumerate(sorted_nodes)}
    int_edges = [(node2id_mapping[u], node2id_mapping[v]) for u, v in sorted_edges]
    int64_array = np.array(int_edges, dtype=np.int64)
    return sorted_nodes, sorted_edges, int64_array, node2id_mapping, id2node_mapping


def subgraph_with_nodes_edges_subset(graph, nodes, edges):
    origin_graph_edge_size = len(graph.edges)
    subgraph = graph.subgraph(nodes).copy()
    edges_out = []
    for edge in subgraph.edges:
        if edge not in edges:
            edges_out.append(edge)
    for edge in edges_out:
        subgraph.remove_edge(*edge)
    assert len(subgraph.edges) <= len(edges)
    assert len(graph.edges) == origin_graph_edge_size
    return subgraph


def custom_dfs_edges(G, source=None):
    visited_nodes = set()
    visited_edges = set()
    stack = [(source, iter(G[source]))]

    while stack:
        parent, children = stack[-1]
        try:
            child = next(children)
            edge = (parent, child)
            if edge not in visited_edges and child not in visited_nodes:
                visited_nodes.add(child)
                visited_edges.add(edge)
                yield edge
                stack.append((child, iter(G[child])))
            elif edge not in visited_edges:
                visited_edges.add(edge)
                yield edge
        except StopIteration:
            stack.pop()


def sort_graph_dfs(edges):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    # Perform DFS traversal and create a list of the edges in DFS order
    dfs_ordered_edges = []
    visited_edges = set()
    visited_nodes = set()
    dfs_edges_each_turn = []
    # Get the nodes sorted by in-degree
    sorted_nodes = sorted(G.nodes(), key=lambda node: G.in_degree(node))

    while sorted_nodes:
        node = sorted_nodes.pop(0)  # Get the next node with the smallest in-degree
        if node not in visited_nodes:
            visited_nodes.add(node)
        dfs_edges = list(custom_dfs_edges(G, source=node))
        dfs_edges_each_turn.append(dfs_edges)
        for edge in dfs_edges:
            if edge not in visited_edges:
                visited_edges.add(edge)
                dfs_ordered_edges.append(edge)
                visited_nodes.add(edge[0])
                visited_nodes.add(edge[1])

        # If all nodes have been visited, break the loop
        if len(dfs_ordered_edges) == len(edges):
            break
    assert len(dfs_ordered_edges) == len(edges)
    # sorted_edges = sorted(edges,
    #                       key=lambda edge: (dfs_ordered_edges.index(edge[1]), dfs_ordered_edges.index(edge[0])))
    return dfs_ordered_edges


# API steps utils functions
def edge_to_name(edge, all_hsa_graph, all_entry):
    name_dict = {}
    name_dict['Type'] = all_hsa_graph.edges[edge]['type']
    subtypes = [item['name'] for item in all_hsa_graph.edges[edge]['subtypes']]
    name_dict['Subtype'] = list(set(subtypes))
    name_dict['Biological_Process'] = {}
    for id in all_hsa_graph.edges[edge]['hsa']:
        name_dict['Biological_Process'][id] = all_entry[id]['main_entry']['NAME'][0].replace(
            '- Homo sapiens (human)', '').strip()
    for id in all_hsa_graph.edges[edge]['pathway']:
        name_dict['Biological_Process'][id] = all_entry[id]['main_entry']['NAME'][0].replace(
            '- Homo sapiens (human)', '').strip()
    return name_dict


def entity_to_name(entity, all_entry):
    if 'NAME' in all_entry[entity]['main_entry']:
        res = [item.replace('(RefSeq)', '').replace('- Homo sapiens (human)', '').strip() for item in
               all_entry[entity]['main_entry']['NAME']]
        if 'SYMBOL' in all_entry[entity]['main_entry']:
            assert isinstance(all_entry[entity]['main_entry']['SYMBOL'], list)
            if len(all_entry[entity]['main_entry']['SYMBOL']) == 1:
                res += [item.strip() for item in all_entry[entity]['main_entry']['SYMBOL'][0].split(',')]
            else:
                res += all_entry[entity]['main_entry']['SYMBOL']
    elif entity.startswith('G') and 'COMPOSITION' in all_entry[entity]['main_entry']:
        res = all_entry[entity]['main_entry']['COMPOSITION']
    else:
        assert entity in ['D09853']  # Corner Case
        return ['Urological agent']
    return res


def node_to_name(node, all_hsa_graph, all_entry):
    entity_list = remove_hsa_id_pre(node)
    if len(entity_list) == 1:
        return entity_to_name(entity_list[0], all_entry)
    else:
        name_in_graph = all_hsa_graph.nodes[node]['graphics']['name']
        # The case where C, K, G, and other entities are not named, but give the entity id in name
        if name_in_graph in all_entry:
            return entity_to_name(name_in_graph, all_entry)
        else:
            return [item.strip() for item in name_in_graph.strip('...').split(',')]


def remove_hsa_id_pre(index):
    res = []
    for item in index.split(' '):
        if ':' in item:
            item = item.split(':')[1]
            res.append(item)
    return res


# Search engine

def retriever_value_entity(later_name, entry):
    if later_name.startswith('hsa'):
        return abstract_dict(entry['main_entry'], ['NAME', 'DESCRIPTION'])
    elif later_name.startswith('map'):
        return abstract_dict(entry['main_entry'], ['NAME', 'DESCRIPTION', 'INCLUDING'])
    elif later_name.startswith('nt'):  # currently no nt
        return abstract_dict(entry['main_entry'], ['NAME'])
    elif later_name.startswith('ko'):
        return abstract_dict(entry['main_entry'], ['NAME', 'DESCRIPTION'])

    elif later_name.startswith('N'):
        return abstract_dict(entry['main_entry'],
                             ['NAME', 'CLASS'])
    elif later_name.startswith('M'):
        return abstract_dict(entry['main_entry'],
                             ['NAME', 'CLASS'])

    elif later_name.startswith("C"):
        return abstract_dict(entry['main_entry'], ['NAME', 'FORMULA', 'BRITE', 'COMMENT'])
    elif later_name.startswith("H"):
        return abstract_dict(entry['main_entry'],
                             ['NAME', 'SUBGROUP', 'DESCRIPTION', 'BRITE', 'PATHOGEN', 'COMMENT'])
    elif later_name.startswith('K'):
        return abstract_dict(entry['main_entry'], ['SYMBOL', 'NAME', 'BRITE', ])
    elif later_name.startswith('G'):
        return abstract_dict(entry['main_entry'], ['NAME', 'BRITE', 'COMMENT'])
    elif later_name.startswith('D'):
        return abstract_dict(entry['main_entry'],
                             ['NAME', 'FORMULA', 'SOURCE', 'CLASS', 'EFFICACY', 'COMMENT', 'BRITE', 'GENERIC', 'ABBR',
                              'COMPONENT'])

    elif bool(re.match(r'^\d', later_name)):
        return abstract_dict(entry['main_entry'],
                             ['SYMBOL', 'NAME', 'ORTHOLOGY', 'BRITE'])
    else:
        raise ValueError('not supported prefix for data {}'.format(later_name))


def get_entries_retriever_value(entry_list, all_entry):
    indexes = []
    descriptions = []
    for key in entry_list:
        indexes.append(key)
        descriptions.append(str(retriever_value_entity(key, all_entry[key])))
    return indexes, descriptions


def get_nodes_retriever_value(nodes_list, all_entry):
    indexes = []
    descriptions = []
    for node in nodes_list:
        indexes.append(node)
        full_info = []
        for entity in remove_hsa_id_pre(node):
            full_info.append(retriever_value_entity(entity, all_entry[entity]))
        descriptions.append(str(full_info))
    return indexes, descriptions


def get_triples_retriever_value(triple_list, whole_graph, all_entry):
    indexes = []
    all_full_info = []
    for line in triple_list:
        full_info = []
        for item in remove_hsa_id_pre(line[0]):
            full_info.append(retriever_value_entity(item, all_entry[item]))
        for item in remove_hsa_id_pre(line[1]):
            full_info.append(retriever_value_entity(item, all_entry[item]))

        full_info += [retriever_value_entity(id, all_entry[id]) for id in
                      whole_graph.edges[line[0], line[1]]['hsa']] + [
                         retriever_value_entity(id, all_entry[id]) for id in
                         whole_graph.edges[line[0], line[1]]['pathway']]
        full_info = str(full_info)
        indexes.append((line[0], line[1]))
        all_full_info.append(full_info)
    return indexes, all_full_info


def get_edge_retriever_value(triple_list, whole_graph, all_entry):
    indexes = []
    all_full_info = []
    for line in triple_list:
        full_info = []
        full_info += [retriever_value_entity(id, all_entry[id]) for id in
                      whole_graph.edges[line[0], line[1]]['hsa']] + [
                         retriever_value_entity(id, all_entry[id]) for id in
                         whole_graph.edges[line[0], line[1]]['pathway']]
        full_info = str(full_info)
        indexes.append((line[0], line[1]))
        all_full_info.append(full_info)
    return indexes, all_full_info


class ItemRetriever:
    def __init__(self, index, descriptions):
        self.index = index
        self.descriptions = descriptions

        self.vectorizer = TfidfVectorizer()
        self.sentence_vectors = self.vectorizer.fit_transform(descriptions)

    def retrieve(self, key_words_list, topk, include_zero_items=False):
        assert isinstance(key_words_list, list), 'key_words_list is not a list.'
        query = ' '.join(key_words_list)
        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(query_vector, self.sentence_vectors).flatten()
        top_indices = (-similarities).argsort()[:topk]

        res_index = [self.index[i] for i in top_indices if (similarities[i] > 0 or include_zero_items)]
        res_description = [self.descriptions[i] for i in top_indices if (similarities[i] > 0 or include_zero_items)]
        score = [similarities[i] for i in top_indices if (similarities[i] > 0 or include_zero_items)]
        return res_index, res_description, score


def retrieve_content_by_key_words(content_str, key_words_list, topk, include_zero_items=False, custom_index=None):
    if len(content_str) == 0:
        return [], [], []
    if custom_index is not None:
        res_index = custom_index
        assert len(res_index) == len(content_str)
    else:
        res_index = list(range(len(content_str)))
    retriever = ItemRetriever(res_index, content_str)
    res_index, res_content_str, score = retriever.retrieve(key_words_list, topk=topk,
                                                           include_zero_items=include_zero_items)
    return res_index, res_content_str, score


# Entity related function


def abstract_dict(full_dict, abs_keys):
    res_dict = {}
    for key in abs_keys:
        if key in full_dict:
            res_dict[key] = full_dict[key]
    return res_dict


def exclusive_abstract_dict(full_dict, exclude_keys):
    res_dict = {}
    for key in full_dict:
        if key not in exclude_keys:
            res_dict[key] = full_dict[key]
    return res_dict


# todo: improve entry abstract
def get_abstract_of_entry(later_name, entry):
    if later_name.startswith("C"):
        return abstract_dict(entry['main_entry'], ['ENTRY', 'NAME'])
    elif later_name.startswith("H"):
        return abstract_dict(entry['main_entry'],
                             ['ENTRY', 'NAME'])
    elif later_name.startswith('hsa'):
        return abstract_dict(entry['main_entry'], ['ENTRY', 'NAME'])
    elif later_name.startswith('map'):
        return abstract_dict(entry['main_entry'], ['ENTRY', 'NAME'])
    elif later_name.startswith('D'):
        return abstract_dict(entry['main_entry'],
                             ['ENTRY', 'NAME', 'FORMULA'])
    elif later_name.startswith('K'):
        return abstract_dict(entry['main_entry'], ['ENTRY', 'NAME'])
    elif later_name.startswith('G'):
        return abstract_dict(entry['main_entry'], ['ENTRY', 'NAME', 'COMPOSITION'])
    elif later_name.startswith('nt'):
        return abstract_dict(entry['main_entry'], ['ENTRY', 'NAME'])
    elif later_name.startswith('ko'):
        return abstract_dict(entry['main_entry'], ['ENTRY', 'NAME'])
    elif later_name.startswith('N'):
        return abstract_dict(entry['main_entry'], ['ENTRY', 'NAME'])
    elif bool(re.match(r'^\d', later_name)):
        return abstract_dict(entry['main_entry'],
                             ['ENTRY', 'SYMBOL'])
    else:
        raise ValueError('not supported prefix for data {}'.format(later_name))


# todo: improve entry detail
def illustrate_detailed_entity(later_name, entry):
    if later_name.startswith("C"):
        return abstract_dict(entry['main_entry'], ['ENTRY', 'NAME'])
    elif later_name.startswith("H"):
        return abstract_dict(entry['main_entry'],
                             ['ENTRY', 'NAME', 'DESCRIPTION', 'DIS_PATHWAY', 'PATHWAY', 'NETWORK', 'DRUG'])

    elif later_name.startswith('hsa'):
        return abstract_dict(entry['main_entry'], ['ENTRY', 'NAME', 'DESCRIPTION', 'PATHWAY_MAP', 'DRUG'])
    elif later_name.startswith('map'):
        return abstract_dict(entry['main_entry'], ['ENTRY', 'NAME'])
    elif later_name.startswith('D'):
        return abstract_dict(entry['main_entry'],
                             ['ENTRY', 'NAME', 'FORMULA', 'TYPE', 'CLASS', 'REMARK', 'DISEASE', 'TARGET', 'PATHWAY',
                              'INTERACTION'])
    elif later_name.startswith('K'):
        return abstract_dict(entry['main_entry'], ['ENTRY', 'NAME'])
    elif later_name.startswith('G'):
        return abstract_dict(entry['main_entry'], ['ENTRY', 'NAME'])
    elif later_name.startswith('nt'):
        return abstract_dict(entry['main_entry'], ['ENTRY', 'NAME'])
    elif later_name.startswith('ko'):
        return abstract_dict(entry['main_entry'], ['ENTRY', 'NAME'])
    elif later_name.startswith('N'):
        return abstract_dict(entry['main_entry'],
                             ['ENTRY', 'NAME', 'DEFINITION', 'EXPANDED', 'CLASS', 'PATHWAY', 'DISEASE', 'GENE',
                              'PERTURBANT'])
    elif bool(re.match(r'^\d', later_name)):
        return abstract_dict(entry['main_entry'],
                             ['ENTRY', 'SYMBOL', 'PATHWAY', 'NETWORK', 'ELEMENT', 'DISEASE', 'DRUG_TARGET'])
    else:
        raise ValueError('not supported prefix for data {}'.format(later_name))


# todo: improve entry illustration
def illustrate_entries(res_index, all_entry):
    res_dict_list = []
    for key in res_index:
        res_dict = {key: str(get_abstract_of_entry(key, all_entry[key]))}
        res_dict_list.append(res_dict)
    return res_dict_list


# Triples related function

def graph_next_step(whole_graph, entity):
    entity_match = node_match(whole_graph, [entity])
    if len(entity_match) == 0:
        raise ValueError('Entity id {} is invalid.'.format(entity))

    all_triples = []
    for node in entity_match:
        neighbors = list(whole_graph.neighbors(node))
        for neighbor in neighbors:
            all_triples.append((node, neighbor))

    all_triples = list(set(all_triples))
    return all_triples


def graph_previous_step(whole_graph, entity):
    entity_match = node_match(whole_graph, [entity])
    if len(entity_match) == 0:
        raise ValueError('Entity id {} is invalid.'.format(entity))

    all_triples = []
    for node in entity_match:
        predecessors = list(whole_graph.predecessors(node))
        for predecessor in predecessors:
            all_triples.append((predecessor, node))

    all_triples = list(set(all_triples))
    return all_triples


def graph_entity_N_hop_subgraph(whole_graph, entity, N_hop):
    entity_match = node_match(whole_graph, [entity])
    if len(entity_match) == 0:
        raise ValueError('Entity id {} is invalid.'.format(entity))

    all_neighbor_nodes = set()
    for node in entity_match:
        all_neighbor_nodes |= set(nx.single_source_shortest_path_length(whole_graph, node, cutoff=N_hop).keys())
        all_neighbor_nodes |= set(
            nx.single_source_shortest_path_length(whole_graph.reverse(), node, cutoff=N_hop).keys())
    subgraph = whole_graph.subgraph(all_neighbor_nodes)

    all_triples = list(subgraph.edges)
    return all_triples, subgraph


def graph_triple_N_hop_subgraph(whole_graph, triple, N_hop):
    assert triple in whole_graph.edges
    all_neighbor_nodes = set()
    for node in triple:
        all_neighbor_nodes |= set(nx.single_source_shortest_path_length(whole_graph, node, cutoff=N_hop).keys())
        all_neighbor_nodes |= set(
            nx.single_source_shortest_path_length(whole_graph.reverse(), node, cutoff=N_hop).keys())
    subgraph = whole_graph.subgraph(all_neighbor_nodes)

    all_triples = list(subgraph.edges)
    return all_triples, subgraph


def illustrate_triples(res_triples, whole_graph, all_entry):
    res_dict_list = []
    for line in res_triples:
        line_named = ({'ID': remove_hsa_id_pre(line[0]), 'Name': node_to_name(line[0], whole_graph, all_entry)},
                      edge_to_name([line[0], line[1]], whole_graph, all_entry),
                      {'ID': remove_hsa_id_pre(line[1]), 'Name': node_to_name(line[1], whole_graph, all_entry)})
        res_dict_list.append(line_named)
    return res_dict_list


def recover_triple(res_dict):
    triple = (' '.join([add_pre(item) for item in res_dict[0]['ID']]),
              ' '.join([add_pre(item) for item in res_dict[2]['ID']]))
    return triple


def compact_triple_to_str(triple_raw, max_length=150, key_words_list=[]):
    # Compact display of dict, list, and str, without symbols such as [], {}. ''
    # Remove certain unnecessary keys (just the key, not the value) such as ID, Subtype, and Type
    # Remove the entity ID value if necessary (Remap the entity ID to a small value and keep the map)
    # Limit the number of items in a list or dict
    def clean_str(string):
        return string.replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace(',', '').replace("'",
                                                                                                                   '').replace(
            '"', '').strip()

    def node_to_str(node):
        assert set(node.keys()) == {'ID', 'Name'}
        assert isinstance(node['ID'], list)
        assert isinstance(node['Name'], list)
        return clean_str(str(node['ID']) + ': ' + str(node['Name']))

    def edge_to_str(edge_dict):
        assert set(edge_dict.keys()).issubset({'Type', 'Subtype', 'Biological_Process'})
        edge_str = ''
        if 'Type' in edge_dict:
            assert isinstance(edge_dict['Type'], str)
            edge_str += edge_dict['Type'] + ' '
        if 'Subtype' in edge_dict:
            assert isinstance(edge_dict['Subtype'], list)
            edge_str += str(edge_dict['Subtype']) + ' '
        if 'Biological_Process' in edge_dict:
            assert isinstance(edge_dict['Biological_Process'], dict)
            edge_str += '| '
            edge_str += str(edge_dict['Biological_Process']) + ' '
        return clean_str(edge_str)

    def triple_to_str(triple):
        head = node_to_str(triple[0])
        tail = node_to_str(triple[2])
        edge = edge_to_str(triple[1])

        res = head.strip() + ' | ' + tail.strip() + ' | ' + edge.strip()
        res = clean_str(res)
        return res

    def length_clean(content):
        return num_tokens_string(clean_str(str(content)))

    exceed = False
    triple = copy.deepcopy(triple_raw)
    res = triple_to_str(triple)

    if num_tokens_string(res) > max_length:
        exceed = True

        sorted_biological_process_keys = list(triple[1]['Biological_Process'].keys())
        # Sort Name, subtype, and Biological_Process according to key_words_list if given
        if len(key_words_list) > 0:
            sorted_biological_process_keys, _, _ = retrieve_content_by_key_words(
                [triple[1]['Biological_Process'][key] for key in sorted_biological_process_keys], key_words_list,
                len(sorted_biological_process_keys), include_zero_items=True,
                custom_index=sorted_biological_process_keys)

            triple[1]['Subtype'], _, score = retrieve_content_by_key_words(triple[1]['Subtype'], key_words_list,
                                                                           len(triple[1]['Subtype']),
                                                                           include_zero_items=True,
                                                                           custom_index=triple[1]['Subtype'])

            triple[0]['Name'], _, score = retrieve_content_by_key_words(triple[0]['Name'], key_words_list,
                                                                        len(triple[0]['Name']),
                                                                        include_zero_items=True,
                                                                        custom_index=triple[0]['Name'])

            triple[2]['Name'], _, score = retrieve_content_by_key_words(triple[2]['Name'], key_words_list,
                                                                        len(triple[2]['Name']),
                                                                        include_zero_items=True,
                                                                        custom_index=triple[2]['Name'])

        # Reduce item: triple[1]['Biological_Process'], triple[1]['Subtype'], triple[0/2]['Name'], triple[0/2]['ID']

        while num_tokens_string(res) - 3 > max_length:
            reduced = False
            length_head = num_tokens_string(node_to_str(triple[0]))
            length_tail = num_tokens_string(node_to_str(triple[2]))
            length_edge = num_tokens_string(edge_to_str(triple[1]))
            if length_edge > max_length / 2:
                if length_clean(triple[1]['Biological_Process']) > max_length / 3:
                    if len(sorted_biological_process_keys) > 1:
                        sorted_biological_process_keys = sorted_biological_process_keys[:-1]
                        triple[1]['Biological_Process'] = {key: triple[1]['Biological_Process'][key] for key in
                                                           sorted_biological_process_keys}
                        reduced = True
                if length_clean(triple[1]['Subtype']) > max_length / 6:
                    if len(triple[1]['Subtype']) > 1:
                        triple[1]['Subtype'] = triple[1]['Subtype'][:-1]
                        reduced = True

            if length_head + length_tail > max_length / 2:
                if length_head > max_length / 4:
                    if length_clean(triple[0]['ID']) > max_length / 12:
                        if len(triple[0]['ID']) > 1:
                            triple[0]['ID'] = triple[0]['ID'][:-1]
                            reduced = True
                    if length_clean(triple[0]['Name']) > max_length / 6:
                        if len(triple[0]['Name']) > 1:
                            triple[0]['Name'] = triple[0]['Name'][:-1]
                            reduced = True
                if length_tail > max_length / 4:
                    if length_clean(triple[2]['ID']) > max_length / 12:
                        if len(triple[2]['ID']) > 1:
                            triple[2]['ID'] = triple[2]['ID'][:-1]
                            reduced = True
                    if length_clean(triple[2]['Name']) > max_length / 6:
                        if len(triple[2]['Name']) > 1:
                            triple[2]['Name'] = triple[2]['Name'][:-1]
                            reduced = True
            if not reduced:
                print('Warning! Triple length cannot be further reduced.')
                break

            res = triple_to_str(triple)

    return exceed, res


# Length control for all list result
def length_control(input_list, max_length, key_words_list=[]):
    exceed = False
    res = copy.deepcopy(input_list)
    num_of_tokens = num_tokens_string(str(res))
    if num_of_tokens > max_length:
        exceed = True
        # Sort res according to key_words_list if given
        if len(key_words_list) > 0:
            res, _, score = retrieve_content_by_key_words(
                [str(item) for item in res], key_words_list,
                len(res), include_zero_items=True, custom_index=res)

        while num_of_tokens > max_length:
            if len(res) > 1:
                res = res[:-1]
            else:
                raise ValueError("History can not be further reduced")
            num_of_tokens = num_tokens_string(str(res))
    return exceed, res
