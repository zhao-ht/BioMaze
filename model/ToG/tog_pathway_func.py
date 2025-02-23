import ast
import copy
import random
import re

import pandas as pd

from backbone.num_tokens import num_tokens_from_messages
from dataset.pathway_graph_env.pathway_graph_api_utils import edge_to_name, entity_to_name, compact_triple_to_str
from .utils import extract_answer, run_llm


#
# SPARQLPATH = "http://192.168.80.12:8890/sparql"  # depend on your own internal address and port, shown in Freebase folder's readme.md
#
# # pre-defined sparqls
# sparql_head_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ns:%s ?relation ?x .\n}"""
# sparql_tail_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ?x ?relation ns:%s .\n}"""
# sparql_tail_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\nns:%s ns:%s ?tailEntity .\n}"""
# sparql_head_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\n?tailEntity ns:%s ns:%s  .\n}"""
# sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n}"""
#
#
# def check_end_word(s):
#     words = [" ID", " code", " number", "instance of", "website", "URL", "inception", "image", " rate", " count"]
#     return any(s.endswith(word) for word in words)
#
#
# def abandon_rels(relation):
#     if relation == "type.object.type" or relation == "type.object.name" or relation.startswith(
#             "common.") or relation.startswith("freebase.") or "sameAs" in relation:
#         return True
#
#
# def execurte_sparql(sparql_query):
#     sparql = SPARQLWrapper(SPARQLPATH)
#     sparql.setQuery(sparql_query)
#     sparql.setReturnFormat(JSON)
#     results = sparql.query().convert()
#     return results["results"]["bindings"]
#
#
# def replace_relation_prefix(relations):
#     return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/", "") for relation in relations]
#
#
# def replace_entities_prefix(entities):
#     return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/", "") for entity in entities]
#
#
# def id2entity_name_or_type(entity_id):
#     sparql_query = sparql_id % (entity_id, entity_id)
#     sparql = SPARQLWrapper(SPARQLPATH)
#     sparql.setQuery(sparql_query)
#     sparql.setReturnFormat(JSON)
#     results = sparql.query().convert()
#     if len(results["results"]["bindings"]) == 0:
#         return "UnName_Entity"
#     else:
#         return results["results"]["bindings"][0]['tailEntity']['value']
#
#
# def clean_relations(string, entity_id, head_relations):
#     pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
#     relations = []
#     for match in re.finditer(pattern, string):
#         relation = match.group("relation").strip()
#         if ';' in relation:
#             continue
#         score = match.group("score")
#         if not relation or not score:
#             return False, "output uncompleted.."
#         try:
#             score = float(score)
#         except ValueError:
#             return False, "Invalid score"
#         if relation in head_relations:
#             relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
#         else:
#             relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
#     if not relations:
#         return False, "No relations found"
#     return True, relations
#
#
# def if_all_zero(topn_scores):
#     return all(score == 0 for score in topn_scores)
#
#
# def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations):
#     relations = []
#     if if_all_zero(topn_scores):
#         topn_scores = [float(1 / len(topn_scores))] * len(topn_scores)
#     i = 0
#     for relation in topn_relations:
#         if relation in head_relations:
#             relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True})
#         else:
#             relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False})
#         i += 1
#     return True, relations
#
#
# def construct_relation_prune_prompt(question, entity_name, total_relations, args):
#     return extract_relation_prompt % (
#         args.width, args.width) + question + '\nTopic Entity: ' + entity_name + '\nRelations: ' + '; '.join(
#         total_relations) + "\nA: "
#
#
# def construct_entity_score_prompt(question, relation, entity_candidates):
#     return score_entity_candidates_prompt.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '
#
#
# def relation_search_prune(entity_id, entity_name, pre_relations, pre_head, question, args):
#     sparql_relations_extract_head = sparql_head_relations % (entity_id)
#     head_relations = execurte_sparql(sparql_relations_extract_head)
#     head_relations = replace_relation_prefix(head_relations)
#
#     sparql_relations_extract_tail = sparql_tail_relations % (entity_id)
#     tail_relations = execurte_sparql(sparql_relations_extract_tail)
#     tail_relations = replace_relation_prefix(tail_relations)
#
#     if args.remove_unnecessary_rel:
#         head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
#         tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
#
#     if pre_head:
#         tail_relations = list(set(tail_relations) - set(pre_relations))
#     else:
#         head_relations = list(set(head_relations) - set(pre_relations))
#
#     head_relations = list(set(head_relations))
#     tail_relations = list(set(tail_relations))
#     total_relations = head_relations + tail_relations
#     total_relations.sort()  # make sure the order in prompt is always equal
#
#     if args.prune_tools == "llm":
#         prompt = construct_relation_prune_prompt(question, entity_name, total_relations, args)
#
#         result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
#         flag, retrieve_relations_with_scores = clean_relations(result, entity_id, head_relations)
#
#     elif args.prune_tools == "bm25":
#         topn_relations, topn_scores = compute_bm25_similarity(question, total_relations, args.width)
#         flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id,
#                                                                          head_relations)
#     else:
#         model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
#         topn_relations, topn_scores = retrieve_top_docs(question, total_relations, model, args.width)
#         flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id,
#                                                                          head_relations)
#
#     if flag:
#         return retrieve_relations_with_scores
#     else:
#         return []  # format error or too small max_length
#
#
# def entity_search(entity, relation, head=True):
#     if head:
#         tail_entities_extract = sparql_tail_entities_extract % (entity, relation)
#         entities = execurte_sparql(tail_entities_extract)
#     else:
#         head_entities_extract = sparql_head_entities_extract % (entity, relation)
#         entities = execurte_sparql(head_entities_extract)
#
#     entity_ids = replace_entities_prefix(entities)
#     new_entity = [entity for entity in entity_ids if entity.startswith("m.")]
#     return new_entity
#
#
# def entity_score(question, entity_candidates_id, score, relation, args):
#     entity_candidates = [id2entity_name_or_type(entity_id) for entity_id in entity_candidates_id]
#     if all_unknown_entity(entity_candidates):
#         return [1 / len(entity_candidates) * score] * len(entity_candidates), entity_candidates, entity_candidates_id
#     entity_candidates = del_unknown_entity(entity_candidates)
#     if len(entity_candidates) == 1:
#         return [score], entity_candidates, entity_candidates_id
#     if len(entity_candidates) == 0:
#         return [0.0], entity_candidates, entity_candidates_id
#
#     # make sure the id and entity are in the same order
#     zipped_lists = sorted(zip(entity_candidates, entity_candidates_id))
#     entity_candidates, entity_candidates_id = zip(*zipped_lists)
#     entity_candidates = list(entity_candidates)
#     entity_candidates_id = list(entity_candidates_id)
#     if args.prune_tools == "llm":
#         prompt = construct_entity_score_prompt(question, relation, entity_candidates)
#
#         result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
#         return [float(x) * score for x in
#                 clean_scores(result, entity_candidates)], entity_candidates, entity_candidates_id
#
#     elif args.prune_tools == "bm25":
#         topn_entities, topn_scores = compute_bm25_similarity(question, entity_candidates, args.width)
#     else:
#         model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
#         topn_entities, topn_scores = retrieve_top_docs(question, entity_candidates, model, args.width)
#     if if_all_zero(topn_scores):
#         topn_scores = [float(1 / len(topn_scores))] * len(topn_scores)
#     return [float(x) * score for x in topn_scores], topn_entities, entity_candidates_id
#
#
# def update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores,
#                    total_relations, total_entities_id, total_topic_entities, total_head):
#     if len(entity_candidates) == 0:
#         entity_candidates.append("[FINISH]")
#         entity_candidates_id = ["[FINISH_ID]"]
#     candidates_relation = [entity['relation']] * len(entity_candidates)
#     topic_entities = [entity['entity']] * len(entity_candidates)
#     head_num = [entity['head']] * len(entity_candidates)
#     total_candidates.extend(entity_candidates)
#     total_scores.extend(scores)
#     total_relations.extend(candidates_relation)
#     total_entities_id.extend(entity_candidates_id)
#     total_topic_entities.extend(topic_entities)
#     total_head.extend(head_num)
#     return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head
#
#
# def half_stop(question, cluster_chain_of_entities, depth, args):
#     print("No new knowledge added during search depth %d, stop searching." % depth)
#     answer = generate_answer(question, cluster_chain_of_entities, args)
#     save_2_jsonl(question, answer, cluster_chain_of_entities, file_name=args.dataset)
#
#
# def generate_answer(question, cluster_chain_of_entities, args):
#     prompt = answer_prompt + question + '\n'
#     chain_prompt = '\n'.join(
#         [', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
#     prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
#     result = run_llm(prompt, args.temperature_reasoning, args.max_length, args.openai_api_keys, args.LLM_type)
#     return result
#
#
# def entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores,
#                  args):
#     zipped = list(
#         zip(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores))
#     sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
#     sorted_entities_id, sorted_relations, sorted_candidates, sorted_topic_entities, sorted_head, sorted_scores = [x[0]
#                                                                                                                   for x
#                                                                                                                   in
#                                                                                                                   sorted_zipped], [
#         x[1] for x in sorted_zipped], [x[2] for x in sorted_zipped], [x[3] for x in sorted_zipped], [x[4] for x in
#                                                                                                      sorted_zipped], [
#         x[5] for x in sorted_zipped]
#
#     entities_id, relations, candidates, topics, heads, scores = sorted_entities_id[:args.width], sorted_relations[
#                                                                                                  :args.width], sorted_candidates[
#                                                                                                                :args.width], sorted_topic_entities[
#                                                                                                                              :args.width], sorted_head[
#                                                                                                                                            :args.width], sorted_scores[
#                                                                                                                                                          :args.width]
#     merged_list = list(zip(entities_id, relations, candidates, topics, heads, scores))
#     filtered_list = [(id, rel, ent, top, hea, score) for id, rel, ent, top, hea, score in merged_list if score != 0]
#     if len(filtered_list) == 0:
#         return False, [], [], [], []
#     entities_id, relations, candidates, tops, heads, scores = map(list, zip(*filtered_list))
#
#     tops = [id2entity_name_or_type(entity_id) for entity_id in tops]
#     cluster_chain_of_entities = [[(tops[i], relations[i], candidates[i]) for i in range(len(candidates))]]
#     return True, cluster_chain_of_entities, entities_id, relations, heads
#
#
# def reasoning(question, cluster_chain_of_entities, args):
#     prompt = prompt_evaluate + question
#     chain_prompt = '\n'.join(
#         [', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
#     prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
#
#     response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
#
#     result = extract_answer(response)
#     if if_true(result):
#         return True, response
#     else:
#         return False, response


def generate_pathway_triple_prune_prompt(all_hsa_graph):
    df = pd.read_csv(
        'dataset/pathway_cot/data_generation/reasoning_data/all_drug_disease_reasoning_parsed_mechanism_judge_merge.csv')
    df['aligned_ordered_subgraph_all'] = df['aligned_ordered_subgraph_all'].map(eval)
    related_edge_triples = []
    for edge in df.loc[45]['aligned_ordered_subgraph_all']['sorted_edges']:
        head = entity_to_name(edge[0], all_hsa_graph)
        tail = entity_to_name(edge[1], all_hsa_graph)
        relation = edge_to_name(edge, all_hsa_graph)
        related_edge_triples.append((head, relation, tail))
    unrelated_edge_triples = []
    for edge in df.loc[47]['aligned_ordered_subgraph_all']['sorted_edges']:
        head = entity_to_name(edge[0], all_hsa_graph)
        tail = entity_to_name(edge[1], all_hsa_graph)
        relation = edge_to_name(edge, all_hsa_graph)
        unrelated_edge_triples.append((head, relation, tail))
    unrelated_edge_triples = list(set(unrelated_edge_triples) - set(related_edge_triples))

    question = df.loc[45]['question']

    prompt = '''Please select the triples' sublist that related to the question.
    Question: {}
    Triples: {}
    Select: {}
    '''

    related_edge_triples_sampled = random.sample(related_edge_triples, 3)
    unrelated_edge_triples_sampled = random.sample(unrelated_edge_triples, 3)
    given_triples = random.sample(related_edge_triples_sampled + unrelated_edge_triples_sampled, 6)
    selected_triples = [item for item in given_triples if item in related_edge_triples]
    pathway_prompt = prompt.format(question, related_edge_triples_sampled + unrelated_edge_triples_sampled,
                                   related_edge_triples_sampled)

    return pathway_prompt


pathway_triple_prune_prompt = """Please select triples that related to the question, and return the list of selected triple ids. 
Question: Were IQ-DNA adducts formed after co-incubation of IQ with calf thymus DNA, hydrogen peroxide, and either bovine LPO or horseradish peroxidase (HRP)?
Triples: 0: ({'ID': ['10', '9'], 'Name': 'NAT2'}, {'Type': 'pathway', 'Subtype': ['activates or, through an enzymatic reaction, results in'], 'Related_Biological_Process': ['Chemical carcinogenesis - DNA adducts - Homo sapiens (human)', 'IQ to DNA adducts']}, {'ID': ['C20290'], 'Name': 'C20290'})
1: ({'ID': ['C16844', 'C00027'], 'Name': 'C16844...'}, {'Type': 'PCrel', 'Subtype': ['activation'], 'Related_Biological_Process': ['Cellular senescence - Homo sapiens (human)']}, {'ID': ['5606', '5608'], 'Name': 'MAP2K3'})
2: ({'ID': ['1544'], 'Name': 'CYP1A2'}, {'Type': 'ECrel', 'Subtype': ['has an relation of successive reactions with', 'compound'], 'Related_Biological_Process': ['Chemical carcinogenesis - DNA adducts - Homo sapiens (human)', 'PhIP to DNA adducts']}, {'ID': ['124907837', '445329', '6799', '6817', '6818'], 'Name': '124907837...'})
3: ({'ID': ['1544'], 'Name': 'CYP1A2'}, {'Type': 'ECrel', 'Subtype': ['compound', 'compound', 'compound', 'compound', 'compound', 'has an relation of successive reactions with'], 'Related_Biological_Process': ['Caffeine metabolism - Homo sapiens (human)', 'Chemical carcinogenesis - DNA adducts - Homo sapiens (human)', 'MeIQx to DNA adducts', '4-ABP to DNA adducts', 'IQ to DNA adducts', 'PhIP to DNA adducts']}, {'ID': ['10', '9'], 'Name': 'NAT2'})
Triple 0: ('NAT2', 'pathway', 'C20290') is related to the biological process of Chemical carcinogenesis - DNA adducts involving IQ to DNA adducts. Triple 3: ('CYP1A2', 'ECrel', 'NAT2') is relevant because it shows the relationship between the enzymes CYP1A2 and NAT2 in the context of Chemical carcinogenesis - DNA adducts, specifically mentioning the formation of IQ to DNA adducts, 4-ABP to DNA adducts, and PhIP to DNA adducts. These two triples provide information about the enzymes and processes involved in the formation of IQ-DNA adducts, which is the main focus of the question.
Select: [0, 3]

Question: Is VAMP-7 concentrated in the trans-Golgi network region of the cell, late endosomes, and transport vesicles that contain the mannose-6 phosphate receptor?
Triples: 0: ({'ID': ['hsa00030'], 'Name': 'Pentose phosphate pathway'}, {'Type': 'maplink', 'Subtype': ['compound'], 'Related_Biological_Process': ['Glycolysis / Gluconeogenesis - Homo sapiens (human)']}, {'ID': ['7167'], 'Name': 'TPI1'})
1: ({'ID': ['hsa00030'], 'Name': 'Pentose phosphate pathway'}, {'Type': 'maplink', 'Subtype': ['compound'], 'Related_Biological_Process': ['Glycolysis / Gluconeogenesis - Homo sapiens (human)']}, {'ID': ['2597', '26330'], 'Name': 'GAPDH'})
There are no triples in the provided list that relate to the question about VAMP-7 concentration in the trans-Golgi network region, late endosomes, and transport vesicles containing the mannose-6 phosphate receptor.
Select: []

Question: Does amino acid starvation rapidly increase the reactivity of the Ser(2448) phosphospecific antibody with mTOR?
Triples: 0: ({'ID': ['2475'], 'Name': 'MTOR'}, {'Type': 'GErel', 'Subtype': ['expression', 'indirect effect'], 'Related_Biological_Process': ['Th17 cell differentiation - Homo sapiens (human)']}, {'ID': ['3662'], 'Name': 'IRF4'})
1: ({'ID': ['2475'], 'Name': 'MTOR'}, {'Type': 'PPrel', 'Subtype': ['activation', 'indirect effect'], 'Related_Biological_Process': ['Central carbon metabolism in cancer - Homo sapiens (human)']}, {'ID': ['3091'], 'Name': 'HIF1A'})
2: ({'ID': ['2475'], 'Name': 'MTOR'}, {'Type': 'PPrel', 'Subtype': ['phosphorylation', 'inhibition'], 'Related_Biological_Process': ['Adipocytokine signaling pathway - Homo sapiens (human)', 'Type II diabetes mellitus - Homo sapiens (human)']}, {'ID': ['3667', '8471', '8660'], 'Name': 'IRS1'})
3: ({'ID': ['2475'], 'Name': 'MTOR'}, {'Type': 'PPrel', 'Subtype': ['activation', 'indirect effect'], 'Related_Biological_Process': ['Central carbon metabolism in cancer - Homo sapiens (human)']}, {'ID': ['8140'], 'Name': 'SLC7A5'})
4: ({'ID': ['2475'], 'Name': 'MTOR'}, {'Type': 'PPrel', 'Subtype': ['binding/association'], 'Related_Biological_Process': ['mTOR signaling pathway - Homo sapiens (human)']}, {'ID': ['253260'], 'Name': 'RICTOR'})
5: ({'ID': ['2475', '57521'], 'Name': 'MTOR'}, {'Type': 'PPrel', 'Subtype': ['inhibition'], 'Related_Biological_Process': ['Longevity regulating pathway - multiple species - Homo sapiens (human)']}, {'ID': ['1979'], 'Name': 'EIF4EBP2'})
The triple ({'ID': ['2475'], 'Name': 'MTOR'}, {'Type': 'PPrel', 'Subtype': ['phosphorylation', 'inhibition'], 'Related_Biological_Process': ['Adipocytokine signaling pathway - Homo sapiens (human)', 'Type II diabetes mellitus - Homo sapiens (human)']}, {'ID': ['3667', '8471', '8660'], 'Name': 'IRS1'}) is related to the reactivity of the Ser(2448) phosphospecific antibody with mTOR in the context of amino acid starvation. 
Select: [2]
"""

pathway_triple_prune_related_prompt = """Please select multiple triples that are related to the question's topic. Note that the triples you choose do not need to strictly answer the question, but they should be related to the topic, such as having a related entity in the question or belonging to a related process (upstream or downstream). Return the list of selected triple IDs in the format shown in the examples below:

Question: Were IQ-DNA adducts formed after co-incubation of IQ with calf thymus DNA, hydrogen peroxide, and either bovine LPO or horseradish peroxidase (HRP)?
Triples: 0: ({'ID': ['10', '9'], 'Name': 'NAT2'}, {'Type': 'pathway', 'Subtype': ['activates or, through an enzymatic reaction, results in'], 'Related_Biological_Process': ['Chemical carcinogenesis - DNA adducts - Homo sapiens (human)', 'IQ to DNA adducts']}, {'ID': ['C20290'], 'Name': 'C20290'})
1: ({'ID': ['C16844', 'C00027'], 'Name': 'C16844...'}, {'Type': 'PCrel', 'Subtype': ['activation'], 'Related_Biological_Process': ['Cellular senescence - Homo sapiens (human)']}, {'ID': ['5606', '5608'], 'Name': 'MAP2K3'})
2: ({'ID': ['1544'], 'Name': 'CYP1A2'}, {'Type': 'ECrel', 'Subtype': ['has an relation of successive reactions with', 'compound'], 'Related_Biological_Process': ['Chemical carcinogenesis - DNA adducts - Homo sapiens (human)', 'PhIP to DNA adducts']}, {'ID': ['124907837', '445329', '6799', '6817', '6818'], 'Name': '124907837...'})
3: ({'ID': ['1544'], 'Name': 'CYP1A2'}, {'Type': 'ECrel', 'Subtype': ['compound', 'compound', 'compound', 'compound', 'compound', 'has an relation of successive reactions with'], 'Related_Biological_Process': ['Caffeine metabolism - Homo sapiens (human)', 'Chemical carcinogenesis - DNA adducts - Homo sapiens (human)', 'MeIQx to DNA adducts', '4-ABP to DNA adducts', 'IQ to DNA adducts', 'PhIP to DNA adducts']}, {'ID': ['10', '9'], 'Name': 'NAT2'})
Select: [0, 3]

Question: Does amino acid starvation rapidly increase the reactivity of the Ser(2448) phosphospecific antibody with mTOR?
Triples: 0: ({'ID': ['2475'], 'Name': 'MTOR'}, {'Type': 'GErel', 'Subtype': ['expression', 'indirect effect'], 'Related_Biological_Process': ['Th17 cell differentiation - Homo sapiens (human)']}, {'ID': ['3662'], 'Name': 'IRF4'})
1: ({'ID': ['2475'], 'Name': 'MTOR'}, {'Type': 'PPrel', 'Subtype': ['activation', 'indirect effect'], 'Related_Biological_Process': ['Central carbon metabolism in cancer - Homo sapiens (human)']}, {'ID': ['3091'], 'Name': 'HIF1A'})
2: ({'ID': ['2475'], 'Name': 'MTOR'}, {'Type': 'PPrel', 'Subtype': ['phosphorylation', 'inhibition'], 'Related_Biological_Process': ['Adipocytokine signaling pathway - Homo sapiens (human)', 'Type II diabetes mellitus - Homo sapiens (human)']}, {'ID': ['3667', '8471', '8660'], 'Name': 'IRS1'})
3: ({'ID': ['2475'], 'Name': 'MTOR'}, {'Type': 'PPrel', 'Subtype': ['activation', 'indirect effect'], 'Related_Biological_Process': ['Central carbon metabolism in cancer - Homo sapiens (human)']}, {'ID': ['8140'], 'Name': 'SLC7A5'})
4: ({'ID': ['2475'], 'Name': 'MTOR'}, {'Type': 'PPrel', 'Subtype': ['binding/association'], 'Related_Biological_Process': ['mTOR signaling pathway - Homo sapiens (human)']}, {'ID': ['253260'], 'Name': 'RICTOR'})
5: ({'ID': ['2475', '57521'], 'Name': 'MTOR'}, {'Type': 'PPrel', 'Subtype': ['inhibition'], 'Related_Biological_Process': ['Longevity regulating pathway - multiple species - Homo sapiens (human)']}, {'ID': ['1979'], 'Name': 'EIF4EBP2'})
Select: [2, 4]

Now, select the triples for this case by output the selected triple IDs in the form of list [id1, id2, ...], following the example format.
"""

from sup_func.sup_func import printc


def parse_triple_prune_result(result, current_entity_relations_list):
    # Extract the list object substring using regular expressions

    # Extract the string after "Select:"
    # if 'Select:' not in result:
    #     printc(result, 'red')
    #     return False, []
    # select_index = result.find("Select:") + len("Select:")
    # select_string = result[select_index:].strip()

    # Parse the string into a Python object
    try:
        list_str = re.search('\[.*\]', result).group(0)
        parsed_result = ast.literal_eval(list_str)
        corresponding_items = []
        for ind in parsed_result:
            corresponding_items.append(current_entity_relations_list[ind])
        return True, corresponding_items
    except Exception as e:
        printc(result + '\n' + str(e), 'red')
        return False, []


def triple_prune(question, current_entity_relations_list, args):
    current_entity_relations_list_reduced = copy.deepcopy(current_entity_relations_list)
    if args.prune_tools == "llm":
        prompt = pathway_triple_prune_related_prompt.strip() + '\n\nQuestion: ' + str(
            question).strip() + '\nTriples: ' + '\n'.join(
            [str(ind) + ': ' + compact_triple_to_str(item, key_words_list=[question])[1] for ind, item in
             enumerate(
                 current_entity_relations_list_reduced)]) + '\nSelect (only output the selected triple IDs in the form of list [id1, id2, ...]): '
        messages = [
            {"role": "user", "content": prompt},
        ]
        while num_tokens_from_messages(messages) >= args.max_context_length - args.max_length:
            current_entity_relations_list_reduced = random.sample(current_entity_relations_list_reduced,
                                                                  len(current_entity_relations_list_reduced) // 2)
            prompt = pathway_triple_prune_related_prompt.strip() + '\n\nQuestion: ' + str(
                question).strip() + '\nTriples: ' + '\n'.join(
                [str(ind) + ': ' + compact_triple_to_str(item, key_words_list=[question])[1] for ind, item in
                 enumerate(
                     current_entity_relations_list_reduced)]) + '\nSelect: (only output the selected triple IDs in the form of list [id1, id2, ...])'
            messages = [
                {"role": "user", "content": prompt},
            ]

        result, _ = args.backbone_func(messages, temperature=args.temperature_exploration)
        return parse_triple_prune_result(result, current_entity_relations_list_reduced)

    elif args.prune_tools == "bm25":
        topn_entities, topn_scores = compute_bm25_similarity(question, entity_candidates, args.width)
    else:
        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
        topn_entities, topn_scores = retrieve_top_docs(question, entity_candidates, model, args.width)
    if if_all_zero(topn_scores):
        topn_scores = [float(1 / len(topn_scores))] * len(topn_scores)
    return [float(x) * score for x in topn_scores], topn_entities, entity_candidates_id


prompt_evaluate_stop_pathway = """Can you answer the question based on your knowledge? I will also provide you with some related facts. Answer by 'I can answer' or 'I cannot answer'. Return 'I can answer' if you already know the answer or can guess the answer, then answer the question.

Q: Does amino acid starvation rapidly increase the reactivity of the Ser(2448) phosphospecific antibody with mTOR?
Knowledge Triplets: ({'ID': ['hsa00330'], 'Name': 'Arginine and proline metabolism'}, {'Type': 'maplink', 'Subtype': ['compound'], 'Related_Biological_Process': ['D-Amino acid metabolism - Homo sapiens (human)']}, {'ID': ['1610'], 'Name': 'DAO'})
({'ID': ['C15972'], 'Name': 'C15972'}, {'Type': 'pathway', 'Subtype': ['binds to'], 'Related_Biological_Process': ['Valine, leucine and isoleucine degradation - Homo sapiens (human)', 'Branched-chain amino acids degradation 2']}, {'ID': ['593', '594'], 'Name': 'BCKDHA'})
({'ID': ['127124', '245973', '51382', '51606', '523', '525', '526', '528', '529', '534', '90423', '9296', '9550'], 'Name': 'ATP6V1G3'}, {'Type': 'PPrel', 'Subtype': ['activation'], 'Related_Biological_Process': ['mTOR signaling pathway - Homo sapiens (human)']}, {'ID': ['10542', '28956', '389541', '55004', '8649'], 'Name': 'LAMTOR5'})
({'ID': ['8528'], 'Name': 'DDO'}, {'Type': 'ECrel', 'Subtype': ['compound'], 'Related_Biological_Process': ['Alanine, aspartate and glutamate metabolism - Homo sapiens (human)']}, {'ID': ['137362', '2805', '2806'], 'Name': 'GOT1L1...'})
A: {I can answer}. The Ser(2448) phosphospecific antibody specifically recognizes mTOR when it is phosphorylated at this site. Therefore, when amino acid starvation leads to decreased phosphorylation at Ser(2448), the reactivity of the phosphospecific antibody with mTOR is also reduced. This is because there are fewer phosphorylated Ser(2448) sites for the antibody to bind to, leading to decreased reactivity. So, amino acid starvation rapidly attenuates, rather than increase the reactivity of the Ser(2448) phosphospecific antibody with mTOR.

Q: Did significantly elevated adduct levels persist in vinyl chloride-treated rat liver 14 days after cessation of exposure?
Knowledge Triplets: ({'ID': ['C02737'], 'Name': 'C02737'}, {'Type': 'pathway', 'Subtype': ['binds to'], 'Related_Biological_Process': ['Efferocytosis - Homo sapiens (human)', 'Exposure of phosphatidylserine to the outer leaflet']}, {'ID': ['121601', '196527', '203859', '338440', '50636', '63982'], 'Name': 'ANO4'})
({'ID': ['1555'], 'Name': 'CYP2B6'}, {'Type': 'ECrel', 'Subtype': ['compound', 'compound', 'compound'], 'Related_Biological_Process': ['Drug metabolism - cytochrome P450 - Homo sapiens (human)']}, {'ID': ['1576'], 'Name': 'CYP3A4'})
({'ID': ['1543', '1545', '1557', '1558', '1559', '1562', '1576'], 'Name': 'CYP1A1'}, {'Type': 'ECrel', 'Subtype': ['compound'], 'Related_Biological_Process': ['Chemical carcinogenesis - DNA adducts - Homo sapiens (human)']}, {'ID': ['119391', '221357', '27306', '2938', '2939', '2940', '2941', '2944', '2946', '2947', '2948', '2949', '2950', '2952', '2953', '373156', '4257', '4258', '4259', '653689', '9446'], 'Name': 'GSTO2'})
({'ID': ['1555', '1559', '1573'], 'Name': 'CYP2B6'}, {'Type': 'ECrel', 'Subtype': ['compound'], 'Related_Biological_Process': ['Arachidonic acid metabolism - Homo sapiens (human)']}, {'ID': ['246', '247'], 'Name': 'ALOX15'})
({'ID': ['1573'], 'Name': 'CYP2J2'}, {'Type': 'ECrel', 'Subtype': ['compound'], 'Related_Biological_Process': ['Linoleic acid metabolism - Homo sapiens (human)']}, {'ID': ['1571'], 'Name': 'CYP2E1'})
({'ID': ['1557'], 'Name': 'CYP2C19'}, {'Type': 'ECrel', 'Subtype': ['compound'], 'Related_Biological_Process': ['Arachidonic acid metabolism - Homo sapiens (human)']}, {'ID': ['242'], 'Name': 'ALOX12B'})
A: {I cannot answer}. I do not know the answer or even can not guess.
"""

get_answer_prompt_judge = 'Since you can answer this question, what is the answer to the question? Return only Yes or No.'

get_answer_prompt_reasoning = 'Since you can answer this question, what is the answer to the question? Output the answer.'


def judge_stop_exploration(question, all_entity_relations_list, args):
    all_entity_relations_list_reduced = copy.deepcopy(all_entity_relations_list)
    prompt = prompt_evaluate_stop_pathway.strip() + '\n\nQ: ' + question.strip() + "\nKnowledge Triplets: " + '\n'.join(
        [compact_triple_to_str(item, key_words_list=[question])[1] for item in
         all_entity_relations_list_reduced]) + '\nA: '
    messages = [
        {"role": "user", "content": prompt},
    ]
    while num_tokens_from_messages(messages) >= args.max_context_length - args.max_length:
        all_entity_relations_list_reduced = random.sample(all_entity_relations_list_reduced,
                                                          len(all_entity_relations_list_reduced) - 1)
        prompt = prompt_evaluate_stop_pathway.strip() + '\n\nQ: ' + question.strip() + "\nKnowledge Triplets: " + '\n'.join(
            [compact_triple_to_str(item, key_words_list=[question])[1] for item in
             all_entity_relations_list_reduced]) + '\nA: '
        messages = [
            {"role": "user", "content": prompt},
        ]

    response, _ = args.backbone_func(messages, temperature=args.temperature_reasoning)

    printc(response, 'magenta')

    stop_result = extract_answer(response)
    if stop_result.lower().strip().replace(" ", "").replace('.', '') == "icananswer" or 'I can answer' in response:
        if args.answer_type != 'reasoning':
            messages += [{"role": "assistant", "content": response}]
            messages += [{"role": "user", "content": get_answer_prompt_judge.strip() + '\n' + question.strip()}]
            answer, _ = args.backbone_func(messages, temperature=args.temperature_reasoning)
            printc(answer, 'magenta')
            if 'yes' in answer.lower():
                answer = 'Yes'
            else:
                answer = 'No'
            return True, response, answer, all_entity_relations_list_reduced
        else:
            if '}' in response:
                answer = response.split('}', 1)[1].lstrip('.').strip()
            else:
                answer = response.split('I can answer.', 1)[-1]
            printc(answer, 'magenta')
            if answer.strip() == '':
                messages += [{"role": "assistant", "content": response}]
                messages += [{"role": "user", "content": get_answer_prompt_reasoning.strip() + '\n' + question.strip()}]
                answer, _ = args.backbone_func(messages, temperature=args.temperature_reasoning)
                printc(answer, 'magenta')
            return True, response, answer, all_entity_relations_list_reduced
    else:
        return False, response, None, all_entity_relations_list_reduced


pathway_answer_prompt = """Answer the question with a final answer {Yes} or {No} (must be in {}). I will also provide you with some related facts. If there is no information related to the question in the facts provided, please answer the question based on your knowledge.

Q: Does amino acid starvation rapidly increase the reactivity of the Ser(2448) phosphospecific antibody with mTOR?
Knowledge Triplets: ({'ID': ['hsa00330'], 'Name': 'Arginine and proline metabolism'}, {'Type': 'maplink', 'Subtype': ['compound'], 'Related_Biological_Process': ['D-Amino acid metabolism - Homo sapiens (human)']}, {'ID': ['1610'], 'Name': 'DAO'})
({'ID': ['C15972'], 'Name': 'C15972'}, {'Type': 'pathway', 'Subtype': ['binds to'], 'Related_Biological_Process': ['Valine, leucine and isoleucine degradation - Homo sapiens (human)', 'Branched-chain amino acids degradation 2']}, {'ID': ['593', '594'], 'Name': 'BCKDHA'})
({'ID': ['127124', '245973', '51382', '51606', '523', '525', '526', '528', '529', '534', '90423', '9296', '9550'], 'Name': 'ATP6V1G3'}, {'Type': 'PPrel', 'Subtype': ['activation'], 'Related_Biological_Process': ['mTOR signaling pathway - Homo sapiens (human)']}, {'ID': ['10542', '28956', '389541', '55004', '8649'], 'Name': 'LAMTOR5'})
({'ID': ['8528'], 'Name': 'DDO'}, {'Type': 'ECrel', 'Subtype': ['compound'], 'Related_Biological_Process': ['Alanine, aspartate and glutamate metabolism - Homo sapiens (human)']}, {'ID': ['137362', '2805', '2806'], 'Name': 'GOT1L1...'})
A: The Ser(2448) phosphospecific antibody specifically recognizes mTOR when it is phosphorylated at this site. Therefore, when amino acid starvation leads to decreased phosphorylation at Ser(2448), the reactivity of the phosphospecific antibody with mTOR is also reduced. This is because there are fewer phosphorylated Ser(2448) sites for the antibody to bind to, leading to decreased reactivity. So, amino acid starvation rapidly attenuates, rather than increase the reactivity of the Ser(2448) phosphospecific antibody with mTOR. So the answer is {No}.

Q: Does APS associate with phosphotyrosines situated within the activation loop of the insulin receptor via the APS Src homology 2 domain?
Knowledge Triplets: ({'ID': ['4916'], 'Name': 'NTRK3'}, {'Type': 'PPrel', 'Subtype': ['activation', 'phosphorylation'], 'Related_Biological_Process': ['Neurotrophin signaling pathway - Homo sapiens (human)']}, {'ID': ['10603'], 'Name': 'SH2B2'})
({'ID': ['6714'], 'Name': 'SRC'}, {'Type': 'PPrel', 'Subtype': ['activation', 'phosphorylation'], 'Related_Biological_Process': ['Chemical carcinogenesis - reactive oxygen species - Homo sapiens (human)']}, {'ID': ['5580'], 'Name': 'PRKCD'})
({'ID': ['3170'], 'Name': 'FOXA2'}, {'Type': 'GErel', 'Subtype': ['expression'], 'Related_Biological_Process': ['Maturity onset diabetes of the young - Homo sapiens (human)']}, {'ID': ['3651'], 'Name': 'PDX1'})
({'ID': ['3651'], 'Name': 'PDX1'}, {'Type': 'GErel', 'Subtype': ['expression'], 'Related_Biological_Process': ['Maturity onset diabetes of the young - Homo sapiens (human)']}, {'ID': ['4825'], 'Name': 'NKX6-1'})
({'ID': ['10603'], 'Name': 'SH2B2'}, {'Type': 'PPrel', 'Subtype': ['activation', 'inhibition'], 'Related_Biological_Process': ['Insulin signaling pathway - Homo sapiens (human)']}, {'ID': ['867', '868'], 'Name': 'CBL'})
A: There is no information related to the question, So I need to answer it by myself. APS does associate with phosphotyrosines situated within the activation loop of the insulin receptor via the APS Src homology 2 domain. APS (adapter protein with pleckstrin homology and Src homology 2 domains) is a signaling adaptor protein that interacts with various receptor tyrosine kinases, including the insulin receptor. The Src homology 2 (SH2) domain of APS specifically recognizes and binds to phosphotyrosines within the activation loop of the insulin receptor, facilitating downstream signaling events. So the answer is {Yes}.
"""

get_final_answer_prompt = 'So, what is the answer to the question? Return only Yes or No.'


#
# def generate_answer_pathway(question, all_entity_relations_list, depth, args):
#     all_entity_relations_list_reduced = copy.deepcopy(all_entity_relations_list)
#     prompt = pathway_answer_prompt.strip() + '\nQ: ' + question.strip() + '\n' + "\nKnowledge Triplets: " + '\n'.join(
#         [compact_triple_to_str(item, key_words_list=[question])[1] for item in
#          all_entity_relations_list_reduced]) + '\nA: '
#     messages = [
#         {"role": "user", "content": prompt},
#     ]
#     while num_tokens_from_messages(messages) >= args.max_context_length - args.max_length:
#         all_entity_relations_list_reduced = random.sample(all_entity_relations_list_reduced,
#                                                           len(all_entity_relations_list_reduced) - 1)
#         prompt = pathway_answer_prompt.strip() + '\nQ: ' + question.strip() + '\n' + "\nKnowledge Triplets: " + '\n'.join(
#             [compact_triple_to_str(item, key_words_list=[question])[1] for item in
#              all_entity_relations_list_reduced]) + '\nA: '
#         messages = [
#             {"role": "user", "content": prompt},
#         ]
#
#     # all_result = []
#     # all_response = []
#     # for i in range(5):
#     response, _ = args.backbone_func(messages, temperature=args.temperature_reasoning)
#
#     printc(response, 'magenta')
#
#     messages += [{"role": "assistant", "content": response}]
#     print(num_tokens_from_messages(messages))
#     messages += [{"role": "user", "content": get_final_answer_prompt.strip() + '\n' + question.strip()}]
#     answer, _ = args.backbone_func(messages, temperature=args.temperature_reasoning)
#     printc(answer, 'magenta')
#     if 'yes' in answer.lower():
#         answer = 'Yes'
#     else:
#         answer = 'No'
#
#     return {'res': answer, 'gen': {'response': response, 'pathway': all_entity_relations_list,
#                                    'sampled_pathway': all_entity_relations_list_reduced, 'depth': depth}}


def generate_answer_pathway(question, all_entity_relations_list, depth, args):
    all_entity_relations_str_list = [compact_triple_to_str(item, key_words_list=[question])[1] for item in
                                     all_entity_relations_list]
    res_dict = args.result_parser(question, all_entity_relations_str_list)
    res_dict['gen']['depth'] = depth
    return res_dict


def half_stop_pathway(question, cluster_chain_of_entities, depth, args):
    printc("No new knowledge added during search depth %d, stop searching." % depth, 'bright_green')
    return generate_answer_pathway(question, cluster_chain_of_entities, depth, args)
