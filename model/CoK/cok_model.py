import copy
import random
import re
import math
from backbone.num_tokens import num_tokens_from_messages
from dataset.pathway_graph_env.pathway_graph_api import get_pathway_api
from dataset.pathway_graph_env.pathway_graph_api_utils import compact_triple_to_str
from dataset.pathway_graph_env.pathway_graph_api_utils import get_triples_retriever_value, get_nodes_retriever_value, \
    get_edge_retriever_value, ItemRetriever
from model.Graph_Agent import graph_agent
from sup_func.sup_func import printc
from backbone.num_tokens import num_tokens_from_messages, num_tokens_string
from dataset.pathway_graph_env.pathway_graph_api_utils import length_control

'''
Sentence: Notch is not necessary downstream of dishevelled in the R3 and R4 photoreceptors for them to adopt distinct fates.
Knowledge Triples:
182 3714: JAG1 AGS AGS1 AHD AWS CD339 CMT2HH DCHE HJ1 JAGL1 | 4851 4853 4854 4855: NOTCH1 AOS5 AOVD1 TAN1 hN1 | PPrel inhibition activation activate | hsa05224: Breast cancer hsa05200: Pathways in cancer hsa04330: Notch signaling pathway hsa01522: Endocrine resistance N00086: Notch signaling pathway N00087: NOTCH-overexpression to Notch signaling pathway
182 3714: JAG1 AGS AGS1 AHD AWS CD339 CMT2HH DCHE HJ1 JAGL1 | 4851 4853: NOTCH1 AOS5 AOVD1 TAN1 hN1 | PPrel activation activate | hsa04658: Th1 and Th2 cell differentiation hsa05224: Breast cancer N00087: NOTCH-overexpression to Notch signaling pathway
4851 4853 4854 4855: NOTCH1 AOS5 AOVD1 TAN1 hN1 | 11317 3516: RBPJL RBPL RBPSUHL SUHL | PPrel compound activate | hsa05224: Breast cancer hsa04330: Notch signaling pathway hsa05165: Human papillomavirus infection N00380: HPV E6 to Notch signaling pathway N00382: HPV E6 to Notch signaling pathway N00381: HPV E6 to Notch signaling pathway N00087: NOTCH-overexpression to Notch signaling pathway N00086: Notch signaling pathway
1855 1856 1857: DVL1 DRS2 DVL DVL1L1 DVL1P1 | 4851 4853 4854 4855: NOTCH1 AOS5 AOVD1 TAN1 hN1 | PPrel inhibition | hsa04330: Notch signaling pathway
5663 5664: PSEN1 ACNINV3 AD3 FAD PS-1 PS1 PSNL1 S182 | C22528: Notch intracellular domain; NICD | pathway activate | hsa04330: Notch signaling pathway N01478: Notch proteolytic activation
11317 3516: RBPJL RBPL RBPSUHL SUHL | 3280 388585: HES1 HES-1 HHL HRY bHLHb39 | GErel cause expression expression | N00087: NOTCH-overexpression to Notch signaling pathway N00380: HPV E6 to Notch signaling pathway N00382: HPV E6 to Notch signaling pathway N00381: HPV E6 to Notch signaling pathway hsa04330: Notch signaling pathway N00086: Notch signaling pathway N00225: EBV EBNA2 to RBP-Jk-mediated transcription hsa05169: Epstein-Barr virus infection hsa05224: Breast cancer
11317 3516: RBPJL RBPL RBPSUHL SUHL | 171558: pre T cell antigen receptor alpha PTCRA PT-ALPHA PTA | GErel expression | hsa04330: Notch signaling pathway
11317 3516: RBPJL RBPL RBPSUHL SUHL | 23462 23493 26508: HEY1 BHLHb31 CHF2 HERP2 HESR1 HRT-1 NERP2 OAF1 hHRT1 | GErel cause expression expression | hsa05224: Breast cancer hsa04330: Notch signaling pathway hsa05165: Human papillomavirus infection N00380: HPV E6 to Notch signaling pathway N00382: HPV E6 to Notch signaling pathway N00381: HPV E6 to Notch signaling pathway N00087: NOTCH-overexpression to Notch signaling pathway N00086: Notch signaling pathway
11317 3516: RBPJL RBPL RBPSUHL SUHL | 145873: mesoderm posterior bHLH transcription factor 2 MESP2 SCDO2 bHLHc6 | pathway cause expression | N01481: Notch-MESP2 signaling
342371 6310: ATXN1L BOAT BOAT1 | 11317 3516: RBPJL RBPL RBPSUHL SUHL | PPrel inhibition | hsa04330: Notch signaling pathway
441478: NOTCH regulated ankyrin repeat protein NRARP | 11317 3516: RBPJL RBPL RBPSUHL SUHL | PPrel inhibition | hsa04330: Notch signaling pathway
182: jagged canonical Notch ligand 1 JAG1 AGS AGS1 AHD AWS CD339 CMT2HH DCHE HJ1 JAGL1 | 4851 4853 4854 4855: NOTCH1 AOS5 AOVD1 TAN1 hN1 | PPrel activation activate | hsa05224: Breast cancer hsa04330: Notch signaling pathway hsa05165: Human papillomavirus infection N00086: Notch signaling pathway N00380: HPV E6 to Notch signaling pathway N00087: NOTCH-overexpression to Notch signaling pathway
C22528: Notch intracellular domain; NICD | 145873: mesoderm posterior bHLH transcription factor 2 MESP2 SCDO2 bHLHc6 | pathway cause expression | N01481: Notch-MESP2 signaling
145873: mesoderm posterior bHLH transcription factor 2 MESP2 SCDO2 bHLHc6 | 3955 4242 5986: LFNG SCDO3 | pathway cause expression | N01481: Notch-MESP2 signaling
57534 142678 9148 54492: (MIB NEUR) | 10683 28514 54567: DLL3 SCDO1 | pathway activate | N01479: Notch ligand ubiquitylation
57534 142678 9148 54492: (MIB NEUR) | 182 3714: JAG1 AGS AGS1 AHD AWS CD339 CMT2HH DCHE HJ1 JAGL1 | pathway activate | N01479: Notch ligand ubiquitylation
Corrected Sentence: Notch is necessary downstream of disheveled in the R3 and R4 photoreceptors for them to adopt distinct fates.
'''

reasoning_step_verify_prompt = '''
Revise the facts and conclusion of the sentence based on the given knowledge triples. 
If the knowledge triples do not indicate any errors in the sentence's facts or conclusion, you can leave the sentence unchanged by return a single 'Unchanged'

Sentence: The expression of the measles V protein does not promote alpha, beta, and gamma interferon-induced transcriptional responses.
Knowledge Triples: 
0) 7535: zeta chain of T cell receptor associated protein kinase 70 ZAP70 ADMIO2 IMD48 SRK STCD STD TZK ZAP-70 | 27040: linker for activation of T cells LAT IMD52 LAT1 pp36 | PPrel activate activation phosphorylation | hsa05235: PD-L1 expression and PD-1 checkpoint pathway in cancer hsa04658: Th1 and Th2 cell differentiation hsa05135: Yersinia infection hsa04020: Calcium signaling pathway hsa04064: NF-kappa B signaling pathway hsa04659: Th17 cell differentiation hsa04660: T cell receptor signaling pathway hsa04014: Ras signaling pathway
1) 356: Fas ligand FASLG ALPS1B APT1LG1 APTL CD178 CD95-L CD95L FASL TNFSF6 TNLG1A | 355: Fas cell surface death receptor FAS ALPS1A APO-1 APT1 CD95 FAS1 FASTM TNFRSF6 | PPrel activate activation binding/association | hsa05162: Measles hsa05170: Human immunodeficiency virus 1 infection hsa05168: Herpes simplex virus 1 infection hsa05022: Pathways of neurodegeneration - multiple diseases hsa05417: Lipid and atherosclerosis
2) C01245: D-myo-Inositol 145-trisphosphate; 1D-myo-Inositol 145-trisphosphate; Inositol 145-trisphosphate; Ins(145)P3 | C00076: Calcium cation; Ca2+; Calcium(2+); Calcium ion | PCrel activate activation indirect effect inhibition | hsa05417: Lipid and atherosclerosis hsa05235: PD-L1 expression and PD-1 checkpoint pathway in cancer hsa04925: Aldosterone synthesis and secretion N00027: Amplified EGFR to PLCG-CAMK signaling pathway N00029: Amplified PDGFR to PLCG-CAMK signaling pathway
3) C00039: DNA; DNAn; DNAn+1; (Deoxyribonucleotide)n; (Deoxyribonucleotide)m; (Deoxyribonucleotide)n+m; Deoxyribonucleic acid | 3428: interferon gamma inducible protein 16 IFI16 IFNGIP1 PYHIN2 | PCrel activation | hsa04623: Cytosolic DNA-sensing pathway
4) 64135: interferon induced with helicase C domain 1 IFIH1 AGS7 Hlcd IDDM19 IMD95 MDA-5 MDA5 RLR-2 SGMRT1 | 57506: mitochondrial antiviral signaling protein MAVS CARDIF IPS-1 IPS1 VISA | PPrel activate activation | hsa05162: Measles hsa05168: Herpes simplex virus 1 infection N00685: MV V to RIG-I-IRF7/3 signaling pathway hsa04622: RIG-I-like receptor signaling pathway hsa05171: Coronavirus disease - COVID-19 hsa05161: Hepatitis B hsa05164: Influenza A
Thought: The knowledge triples do not provide any information that contradicts the facts or conclusion of the sentence. So the sentence is left unchanged.
Corrected Sentence: Unchanged

Sentence: Notch is not necessary downstream of dishevelled in the R3 and R4 photoreceptors for them to adopt distinct fates.
Knowledge Triples:
0) 182 3714: JAG1 AGS AGS1 AHD AWS CD339 CMT2HH DCHE HJ1 JAGL1 | 4851 4853 4854 4855: NOTCH1 AOS5 AOVD1 TAN1 hN1 | PPrel inhibition activation activate | hsa05224: Breast cancer hsa05200: Pathways in cancer hsa04330: Notch signaling pathway hsa01522: Endocrine resistance N00086: Notch signaling pathway N00087: NOTCH-overexpression to Notch signaling pathway
1) 182 3714: JAG1 AGS AGS1 AHD AWS CD339 CMT2HH DCHE HJ1 JAGL1 | 4851 4853: NOTCH1 AOS5 AOVD1 TAN1 hN1 | PPrel activation activate | hsa04658: Th1 and Th2 cell differentiation hsa05224: Breast cancer N00087: NOTCH-overexpression to Notch signaling pathway
2) 4851 4853 4854 4855: NOTCH1 AOS5 AOVD1 TAN1 hN1 | 11317 3516: RBPJL RBPL RBPSUHL SUHL | PPrel compound activate | hsa05224: Breast cancer hsa04330: Notch signaling pathway hsa05165: Human papillomavirus infection N00380: HPV E6 to Notch signaling pathway N00382: HPV E6 to Notch signaling pathway N00381: HPV E6 to Notch signaling pathway N00087: NOTCH-overexpression to Notch signaling pathway N00086: Notch signaling pathway
3) 1855 1856 1857: DVL1 DRS2 DVL DVL1L1 DVL1P1 | 4851 4853 4854 4855: NOTCH1 AOS5 AOVD1 TAN1 hN1 | PPrel inhibition | hsa04330: Notch signaling pathway
4) 5663 5664: PSEN1 ACNINV3 AD3 FAD PS-1 PS1 PSNL1 S182 | C22528: Notch intracellular domain; NICD | pathway activate | hsa04330: Notch signaling pathway N01478: Notch proteolytic activation
5) 182: jagged canonical Notch ligand 1 JAG1 AGS AGS1 AHD AWS CD339 CMT2HH DCHE HJ1 JAGL1 | 4851 4853 4854 4855: NOTCH1 AOS5 AOVD1 TAN1 hN1 | PPrel activation activate | hsa05224: Breast cancer hsa04330: Notch signaling pathway hsa05165: Human papillomavirus infection N00086: Notch signaling pathway N00380: HPV E6 to Notch signaling pathway N00087: NOTCH-overexpression to Notch signaling pathway
Thought: As supported by pathway 5, it is shown that the activation of Notch1 (4851 4853 4854 4855) by Jagged1 (182) is involved in the Notch signaling pathway (hsa04330), breast cancer (hsa05224), and endocrine resistance (hsa01522). Therefore, the association between disheveled and Notch is crucial for the distinct fate determination of R3 and R4 photoreceptors. I will correct the conclusion in the sentence.
Corrected Sentence: Notch is necessary downstream of disheveled in the R3 and R4 photoreceptors for them to adopt distinct fates.
'''


class cok:
    def __init__(self, backbone_func, model_name, all_hsa_graph, all_entity,
                 sys_prompt, prompt_dict,
                 max_pieces,
                 answer_type,
                 max_context_length,
                 max_length,
                 examples,
                 in_context_num,
                 temperature,
                 parsering_func=None):
        self.backbone_func = backbone_func
        self.model_name = model_name
        self.all_hsa_graph = all_hsa_graph
        self.all_entity = all_entity

        self.triple_search_engine = ItemRetriever(
            *get_triples_retriever_value(self.all_hsa_graph.edges, self.all_hsa_graph, self.all_entity))
        self.node_search_engine = ItemRetriever(*get_nodes_retriever_value(all_hsa_graph.nodes, self.all_entity))
        self.edge_search_engine = ItemRetriever(
            *get_edge_retriever_value(all_hsa_graph.edges, all_hsa_graph, self.all_entity))

        self.sys_prompt = sys_prompt
        self.prompt_dict = prompt_dict
        self.max_pieces = max_pieces
        self.answer_type = answer_type
        self.examples = examples

        self.max_context_length = max_context_length
        self.max_length = max_length

        self.in_context_num = in_context_num
        self.temperature = temperature
        self.parsering_func = parsering_func

    def parse_steps(self, gen):
        basic_steps = [re.sub('No, |Yes, ', '', item).strip('.') + '.' for item in
                       gen.strip().replace('\n', ' ').split('. ') if
                       len(item) > 0]
        if len(basic_steps) <= self.max_pieces:
            return basic_steps
        else:
            merged_steps = []
            each_step_size_floor = math.floor(len(basic_steps) / self.max_pieces)
            remains = len(basic_steps) % self.max_pieces
            start = 0
            for ind in range(self.max_pieces):
                end = start + each_step_size_floor
                if ind < remains:
                    end += 1
                merged_steps.append(' '.join(basic_steps[start:end]))
                start = end
            assert start == len(basic_steps)
            return merged_steps

    def parse_new_step(self, response, origin_gen):
        if 'Unchanged' in response or 'unchanged' in response.lower():
            return origin_gen, False
        elif 'Corrected Sentence: ' in response:
            res = response.split('Corrected Sentence: ')[-1]
        else:
            res = response.split('\n')[-1]
        refined = not res.strip() == origin_gen.strip()
        return res, refined

    def refine_step(self, question, step_sentence):
        pathway_apis = get_pathway_api(self.all_hsa_graph, self.all_entity, entity_search_engine=None,
                                       relation_search_engine=self.triple_search_engine,
                                       node_search_engine=self.node_search_engine,
                                       edge_search_engine=self.edge_search_engine,
                                       return_str=True,
                                       enable_raise_error=False)

        # all_entity_relations_list, info = pathway_apis['search_relation']([step_sentence])
        all_entity_relations_list, info = pathway_apis['search_biopathway_subgraph_global']([question, step_sentence])
        if len(all_entity_relations_list) == 0:
            return step_sentence, '', False

        all_entity_relations_list_reduced = copy.deepcopy(all_entity_relations_list)
        prompt = reasoning_step_verify_prompt.strip() + '\n\nSentence: ' + step_sentence.strip()
        prompt_basic_length = num_tokens_string(prompt.strip())
        exceed, all_entity_relations_list_reduced = length_control(all_entity_relations_list_reduced,
                                                                   self.max_context_length - self.max_length - prompt_basic_length,
                                                                   [question])
        prompt = prompt.strip() + "\nKnowledge Triples:\n" + graph_agent.observation_wrapper(
            all_entity_relations_list_reduced, 0).strip() + '\nThought: '
        messages = [
            {"role": "user", "content": prompt},
        ]

        response, _ = self.backbone_func(messages, temperature=self.temperature)

        new_step_sentence, refined = self.parse_new_step(response, step_sentence)
        printc(prompt, 'blue')
        printc(response, 'green')
        printc(new_step_sentence, 'yellow')
        return new_step_sentence, response, refined

    def refine_all_steps(self, question, reasoning_process):
        reasoning_steps = self.parse_steps(reasoning_process)
        refined_steps = []
        refine_response_list = []
        refined_list = []
        for step in reasoning_steps:
            refined_step, refine_response, refined = self.refine_step(question, step)
            refined_steps.append(refined_step)
            refine_response_list.append(refine_response)
            refined_list.append(refined)
        refined_reasoning_steps = ' '.join(refined_steps)
        printc(reasoning_process, 'magenta')
        printc(refined_reasoning_steps, 'green')
        return refined_reasoning_steps, refine_response_list, refined_list

    def __call__(self, problem_text):
        # problem_text = 'Is Notch necessary downstream of dishevelled in the R3 and R4 photoreceptors for them to adopt distinct fates?'
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
        max_token = self.max_context_length - num_tokens_from_messages(messages, self.model_name) - 2500
        if self.answer_type != 'reasoning':
            max_token -= num_tokens_string(self.prompt_dict['ans'].strip() + '\n' + problem_text.strip(),
                                           self.model_name)
        gen, _ = self.backbone_func(messages, temperature=self.temperature, max_tokens=max_token)
        print(gen)
        refined_gen, refine_response_list, refined_list = self.refine_all_steps(problem_text, gen)
        print(refined_gen)
        if self.answer_type != 'reasoning':
            messages += [{"role": "assistant", "content": refined_gen}]
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
            res = gen

        return {'res': res, "gen": {"gen": gen, 'gen_ans': gen_ans, 'messages': messages,
                                    'refine_response_list': refine_response_list, 'refined_list': refined_list}}
