tool_descriptions_full = {
    "search_biopathway_subgraph_global": {
        "description": "This function retrieves biological pathways related to the given keywords. The argument key_words_list is a list of keyword strings. For example: ['Amivantamab', 'Non-small cell lung cancer']. Use this function to explore relevant biological pathways.",
        "fn_signature": "search_biopathway_subgraph_global(key_words_list: list, target_size=16: int, topk=128: int, topk_e=512: int)"
    },
    "search_biopathway_triple_N_hop_subgraph": {
        "description": "This function retrieves biological pathways related to the specified keywords in the vicinity of a previously seen pathway line. The argument history_line_id is an integer ID indicating a pathway line (which looks like 'ID) line content...') from previous interactions. The argument key_words_list is a list of keyword strings, such as ['Amivantamab', 'Non-small cell lung cancer']. Use this function to investigate nearby pathways if you suspect important pathways may be missing.",
        "fn_signature": "search_biopathway_triple_N_hop_subgraph(history_line_id: int, key_words_list: list, target_size=16: int)"
    }
}

tool_descriptions_selected = {}
for key in ["search_biopathway_subgraph_global", "search_biopathway_triple_N_hop_subgraph"]:
    tool_descriptions_selected[key] = tool_descriptions_full[key]


def get_instruction(answer_type) -> str:
    tool_desc = "You have access to the following tools:\n"
    for i, (tool_name, tool_description) in enumerate(tool_descriptions_selected.items()):
        tool_desc += f"[{i + 1}] {tool_description['fn_signature']}: {tool_description['description']}\n"
    res = tool_desc + "\n"

    res += "You can use the tools by outputing a line of Python code that invoke the tools.\n"
    res += "You should begin your tool invocation with 'Action:' and end it with 'End Action'.\n"
    res += "Example: 'Action:\ntool_name(argument_1)\nEnd Action'\n"
    res += "You can only invoke one tool at a time.\n"

    res = res + "\nNow, let's get started!\n\n"
    res = res + "You can optionally express your thoughts using natural language before your action. For example, 'Thought: I want to use tool_name to do something. Action: <your action to call tool_name> End Action'.\n"

    res = res + "Note that your output should always contain either 'Action:' or 'Answer:', but not both.\n"
    res = res + "When you are done, output the result using 'Answer: your answer'. You can also express your thought before the 'Answer:'.\n"
    if answer_type in ['judge']:
        res = res + "After 'Answer:', please ONLY output the answer (e.g., Yes or No), without any other text.\n"
    elif answer_type in ['choice']:
        res = res + "After 'Answer:', please ONLY output the answer (e.g., A, B, C, etc.), without any other text.\n"
    else:
        assert answer_type == 'reasoning'
    res += 'Note: Conduct at least two rounds of investigations related to the biological pathways before answering the question. For clarity, only previously unseen pathways will be shown to you in new turns.\n'
    return res


example2_new2 = ['''
Please explore pathways to find relevant information regarding the following question: APOBEC3G and APOBEC3F, members of the APOBEC family of cellular polynucleotide cytidine deaminases, are known to be potent inhibitors of HIV-1 infection. In the absence of the essential viral protein Vif, what is the effect of APOBEC3G/F on HIV-1 infection, specifically regarding the encapsidation of these proteins by budding virus particles and the subsequent editing of negative sense reverse transcripts in newly infected cells?
''',
                 [
                     '''
     Thought: To find information regarding the effect of APOBEC3G/F on HIV-1 infection, I can use the search_biopathway_subgraph_global tool to explore relevant biological pathways. I should convert all the question content into keywords to enhance retrieval quality. Action:
     search_biopathway_subgraph_global(['APOBEC3G and APOBEC3F, members of the APOBEC family of cellular polynucleotide cytidine deaminases, are known to be potent inhibitors of HIV-1 infection. In the absence of the essential viral protein Vif, what is the effect of APOBEC3G/F on HIV-1 infection, specifically regarding the encapsidation of these proteins by budding virus particles and the subsequent editing of negative sense reverse transcripts in newly infected cells?'])
     End Action
     ''',
                     '''
     ...
     6) K19258: HIV protein Tat tat | 581: BCL2 associated X apoptosis regulator BAX BCL2L4 | pathway activate | hsa05170: Human immunodeficiency virus 1 infection N00450: HIV Tat to intrinsic apoptotic pathway
     7) K22890: HIV-1 protein Vpu vpu | 4790 5970: NFKB1 CVID12 EBP-1 KBF1 NF-kB NF-kB1 NF-kappa-B1 NF-kappaB NF-kappabeta NFKB-p105 NFKB-p50 NFkappaB | pathway inhibit | hsa05170: Human immunodeficiency virus 1 infection N00437: HIV Vpu to TLR2/4-NFKB signaling pathway
     8) K22890: HIV-1 protein Vpu vpu | 3661: interferon regulatory factor 3 IRF3 IIAE7 | pathway inhibit | hsa05170: Human immunodeficiency virus 1 infection N00440: HIV Vpu/Vif/Vpr to cGAS-STING signaling pathway
     ...
     '''],
                 [
                     '''
     Thought: All listed pathways are connected to the broader context of HIV-1 infection. Pathway 7 examines how the viral protein Vpu interacts with immune signaling, which could provide insights into how viral proteins counteract APOBEC3 proteins. I will check the neighboring pathways of Pathway 7 by using the function search_biopathway_triple_N_hop_subgraph with history_line_id set to 7. The keywords are the same as previously used.
     Action:
     search_biopathway_triple_N_hop_subgraph(7, ['APOBEC3G and APOBEC3F, members of the APOBEC family of cellular polynucleotide cytidine deaminases, are known to be potent inhibitors of HIV-1 infection. In the absence of the essential viral protein Vif, what is the effect of APOBEC3G/F on HIV-1 infection, specifically regarding the encapsidation of these proteins by budding virus particles and the subsequent editing of negative sense reverse transcripts in newly infected cells?'])
     End Action
     ''',
                     '''
     ...
     16) 4790 5970: NFKB1 CVID12 EBP-1 KBF1 NF-kB NF-kB1 NF-kappa-B1 NF-kappaB NF-kappabeta NFKB-p105 NFKB-p50 NFkappaB | 2353 3725: FOS AP-1 C-FOS p55 | PPrel indirect effect activation | hsa05170: Human immunodeficiency virus 1 infection
     17) 2353 3725: FOS AP-1 C-FOS p55 | 7124: tumor necrosis factor TNF DIF TNF-alpha TNFA TNFSF2 TNLG1F | GErel expression | hsa05418: Fluid shear stress and atherosclerosis hsa05135: Yersinia infection hsa05170: Human immunodeficiency virus 1 infection hsa04620: Toll-like receptor signaling pathway hsa04660: T cell receptor signaling pathway hsa05132: Salmonella infection hsa05130: Pathogenic Escherichia coli infection hsa05142: Chagas disease
     18) 7124: tumor necrosis factor TNF DIF TNF-alpha TNFA TNFSF2 TNLG1F | 7133: TNF receptor superfamily member 1B TNFRSF1B CD120b TBPII TNF-R-II TNF-R75 TNFBR TNFR1B TNFR2 TNFR80 p75 p75TNFR | PPrel activate activation | hsa04060: Cytokine-cytokine receptor interaction hsa05170: Human immunodeficiency virus 1 infection hsa04668: TNF signaling pathway hsa04061: Viral protein interaction with cytokine and cytokine receptor hsa04920: Adipocytokine signaling pathway
     ...
     '''],
                 ['''
Thought: I have gathered sufficient pathway information related to the question. Now, I need to report the closely related pathway IDs.
Answer:
[7, 16, 18]
''',
                  '''
     You finished the task.
     ''']
                 ]

In_context_example = [example2_new2]


def get_graph_agent_instruction_dict(answer_type):
    return {'instruction': get_instruction(answer_type),
            'system_msg': "You are a master in planning."}, In_context_example


graph_agent_final_answer_prompt = """You are an expert in biological pathways. Infer the question's answer based on your knowledge. I will also provide you with some related pathways. Let's think step by step.
"""

graph_agent_final_answer_prompt_uncertain_judge = """So, have you gotten a final answer to the question? Return 'Yes' if you have a certain answer, or return 'Uncertain' if you do not have a certain final answer."""

get_judge_answer_prompt_conclusion = "So, answer the question with only 'Yes' or 'No':"


def get_graph_agent_result_parser_prompt(dataset_name, answer_type):
    prompt_dict = {}
    prompt_dict['query'] = graph_agent_final_answer_prompt
    prompt_dict['uncertain_judge'] = graph_agent_final_answer_prompt_uncertain_judge
    if answer_type == 'judge':
        prompt_dict['ans'] = get_judge_answer_prompt_conclusion

    return prompt_dict
