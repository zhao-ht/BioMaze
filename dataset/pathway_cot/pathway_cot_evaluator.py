import numpy as np

from backbone.gpt import call_chat_gpt
from backbone.num_tokens import num_tokens_from_messages
from backbone.samba import call_samba
from backbone.vllm import call_vllm
from sup_func.sup_func import printc


def pathway_no_evaluation_evaluator(model_output, data):
    return 1.0


def pathway_judge_evaluator(model_output, data):
    res_set = {'Yes', 'No'}
    if isinstance(model_output, str):
        if model_output not in res_set:
            return 0.0
        cor = model_output == data['answer']
        cor = float(cor)
        return cor
    else:
        assert isinstance(model_output, list)
        score_list = []
        for item in model_output:
            if item not in res_set:
                cor = 0.0
            else:
                cor = item == data['answer']
                cor = float(cor)
            score_list.append(cor)
        return np.mean(score_list)


def pathway_judge_block_evaluator(model_output, data):
    res_map_dict = {'Yes': 0, 'No': 1}
    if model_output not in res_map_dict:
        return 0.0
    cor = res_map_dict[model_output] == data['answer']
    cor = float(cor)
    return cor


def pathway_choice_evaluator(model_output, data):
    cor = (model_output == data['answer'])
    cor = float(cor)
    return cor


pathway_reasoning_answer_evaluator_prompt = '''I will provide you with a question, a standard answer, and a new answer to be evaluated. 
Your task is to evaluate the new answer based on whether it arrives at the same conclusion as the standard answer regarding the question, following the format for examples.
The new answer may include additional reasoning, intermediate steps, or related concepts, but these will not impact the evaluation. 
The final judgement should solely be based on whether the new answer aligns with the conclusion of the standard answer regarding the question.
The judgement will be either 'Correct' if it aligns, or 'Incorrect' if it doesn't.

Question: In pancreatic acinar cells, the secretory hormone cholecystokinin (CCK) triggers complex patterns of Ca2+ signals. Researchers investigated the Ca2+-mobilizing molecule nicotinic acid adenine dinucleotide phosphate (NAADP) and its effects on Ca2+ release. How does the sensitivity to NAADP compare to the sensitivity to cyclic adenosine 5'-diphosphate-ribose (cADPR) and inositol trisphosphate (InsP3) in these cells?
Standard Answer: Cells are significantly more sensitive to NAADP than to either cADPR or InsP3.
New Answer: It can be inferred that the sensitivity to NAADP in pancreatic acinar cells is comparable to the sensitivity to cADPR and InsP3. Both NAADP and cADPR are capable of activating pathways involved in Ca2+ release, such as the pancreatic secretion pathway (hsa04972) and the calcium signaling pathway (hsa04020). Additionally, InsP3 is also involved in the calcium signaling pathway. Therefore, the sensitivity to NAADP is likely to be similar to the sensitivity to cADPR and InsP3 in these cells.
Comment: The question asks about how the sensitivity to NAADP compares to the sensitivity to cyclic adenosine 5'-diphosphate-ribose (cADPR) and inositol trisphosphate (InsP3) in pancreatic acinar cells. For this question, the standard answer states that cells are significantly more sensitive to NAADP than to either cADPR or InsP3, while the new answer concludes that the sensitivity to NAADP is likely to be similar to the sensitivity to cADPR and InsP3. Therefore, the judgement is 'Incorrect'.
Judge: Incorrect

Question: Tax oncoprotein of human T-cell leukemia virus type 1 (HTLV-1) affects multiple regulatory processes of infected cells through activation and repression of specific transcription and also through modulation of functions of cell cycle regulators. It has been found that Tax binds to p16ink4a, a member of the INK4 family of cyclin-dependent kinase inhibitors, and counteracts its inhibitory activity, resulting in cell cycle progression. In addition to p16ink4a, what is the effect of Tax on p15ink4b?
Standard Answer: Tax can bind to p15ink4b and inactivate its function, restoring CDK4 kinase activity.
New Answer: Based on the pathways provided, it is not clear what the specific effect of Tax on p15ink4b is. There are no pathways directly indicating the interaction or modulation of p15ink4b by Tax. Therefore, further research or information is needed to determine the effect of Tax on p15ink4b.
Comment: The question asks about in addition to p16ink4a, what the effect of Tax on p15ink4b is. Regarding the question, the standard answer states that Tax can bind to p15ink4b and inactivate its function, restoring CDK4 kinase activity, while the new answer does not provide a conclusion regarding the effect of Tax on p15ink4b. Therefore, the judgement is 'Incorrect'. 
Judge: Incorrect

Question: A DDE-degrading bacterium, Janibacter sp. TYM3221, is able to grow on biphenyl and degrades 1,1-dichloro-2,2-bis(4-chlorophenyl)ethylene (DDE) via a meta-ring cleavage pathway. The bphAa gene, encoding a biphenyl dioxygenase large subunit, was previously demonstrated to be involved in the degradation of DDE in TYM3221. The bph gene cluster, containing orf2 and bphDAaAbAcAdBCST was cloned and characterized. What is the transcription pattern of the bph gene cluster in Janibacter sp. TYM3221?
Standard Answer: The bph gene cluster, containing orf2 and bphDAaAbAcAdBCST, is transcribed as an operon in Janibacter sp. TYM3221.
New Answer: The transcription pattern of the bph gene cluster in Janibacter sp. TYM3221 is likely to be inducible. This means that the expression of the genes in the cluster, including orf2 and bphDAaAbAcAdBCST, is upregulated in the presence of biphenyl or DDE as inducers. The bphAa gene, encoding the biphenyl dioxygenase large subunit, has been shown to be involved in the degradation of DDE. Therefore, it is expected that the transcription of the bph gene cluster is activated when biphenyl or DDE is present as a substrate for degradation. Further studies would be needed to determine the precise regulatory mechanisms and conditions that influence the transcription of the bph gene cluster in Janibacter sp. TYM3221.
Comment: The question asks for what the transcription pattern of the bph gene cluster in Janibacter sp. TYM3221 is. Toward the question, the standard answer states that the bph gene cluster is transcribed as an operon in Janibacter sp. TYM3221, while the new answer suggests that the transcription pattern of the bph gene cluster is likely to be inducible. The two answers do not contradict each other, but the new answer does not confirm that the bph gene cluster is transcribed as an operon. Therefore, the judgement is 'Incorrect'. 
Judge: Incorrect

Question: Opioids increase the firing of dopamine cells in the ventral tegmental area by presynaptic inhibition of GABA release. In slices taken from morphine-treated guinea pigs after washing out the morphine (withdrawn slices), what is the effect of a low concentration of a mu receptor agonist on the amplitude of the GABAB-mediated IPSP?
Standard Answer: In withdrawn slices from morphine-treated guinea pigs, a low concentration of a mu receptor agonist increases the amplitude of the GABAB IPSP.
New Answer: The opioids increase the firing of dopamine cells in the ventral tegmental area by presynaptic inhibition of GABA release. In the withdrawn slices taken from morphine-treated guinea pigs, a low concentration of a mu receptor agonist is likely to promote the association between beta-catenin and alpha-catenin. Therefore, the effect of a low concentration of a mu receptor agonist on the amplitude of the GABAB-mediated IPSP is likely to be an increase in amplitude.
Comment: The question query about the effect of a low concentration of a mu receptor agonist on the amplitude of the GABAB-mediated IPSP. The standard answer is that a low concentration of a mu receptor agonist increases the amplitude of the GABAB IPSP. Correspondingly, the new answer state that a low concentration of a mu receptor agonist increases the amplitude of the GABAB-mediated IPSP in withdrawn slices from morphine-treated guinea pigs, which is the same as the standard answer regarding the question. Therefore, the judgement is 'Correct'.
Judge: Correct

Question: The aryl hydrocarbon receptor (AHR) is a ligand-dependent transcription factor that plays a role in mediating the toxicity of dioxin. In the CD4+ T-cell lineage of mice, AHR expression is restricted to the T(H)17 cell subset, which is characterized by the production of the cytokine IL-17. Activation of AHR during T(H)17 cell development increases the proportion of T(H)17 T cells and their production of cytokines. However, CD4+ T cells from AHR-deficient mice can still develop T(H)17 cell responses. How do CD4+ T cells from AHR-deficient mice respond to AHR ligand?
Standard Answer: CD4+ T cells from AHR-deficient mice can develop T(H)17 cell responses, but fail to produce IL-22 when confronted with AHR ligand.
New Answer: Based on the pathways provided, it can be inferred that CD4+ T cells from AHR-deficient mice may still respond to AHR ligand. This is supported by pathway 12, which suggests that the aryl hydrocarbon receptor (AHR) can be activated by ligands such as kynurenate. Therefore, even in the absence of AHR expression, CD4+ T cells from AHR-deficient mice may still exhibit a response to AHR ligand.
Comment: The question asks about how CD4+ T cells from AHR-deficient mice respond to AHR ligands. For this question, the standard answer states that CD4+ T cells from AHR-deficient mice can develop T(H)17 cell responses, but fail to produce IL-22 when confronted with AHR ligand. The new answer, however, suggests that CD4+ T cells from AHR-deficient mice may still respond to AHR ligand, but does not mention anything about the production of IL-22. Therefore, the judgement is 'Incorrect'. 
Judge: Incorrect
'''


def pathway_reasoning_answer_evaluator_response_parser(response):
    if 'Judge:' not in response:
        printc('Judge format error {}'.format(response))
        if (
                ' correct' in response.lower() or ' "correct' in response.lower() or " 'correct" in response.lower()) and not (
                ' incorrect' in response.lower() or ' "incorrect' in response.lower() or " 'incorrect" in response.lower()):
            judge = True
        else:
            judge = False
    else:
        start_a = response.find('Judge:') + len('Judge:')
        answer = response[start_a:]
        if answer.strip().lower() == 'correct':
            judge = True
        else:
            judge = False
    return judge


def pathway_reasoning_answer_evaluator(model_output, data, evaluate_model='Meta-Llama-3.1-405B-Instruct'):
    query = pathway_reasoning_answer_evaluator_prompt.strip() + '\n\n' + 'Question: ' + data[
        'question'].strip() + '\n' + 'Standard Answer: ' + data[
                'answer'].strip() + '\n' + 'New Answer: ' + model_output.strip() + '\n' + 'Comment: '
    message = [
        {
            "role": "user",
            "content": query
        }
    ]
    if num_tokens_from_messages(message, evaluate_model) > 3800:
        return {'score': False, 'score_response': 'Out of Length'}
    if evaluate_model == 'Meta-Llama-3.1-405B-Instruct':
        response, _ = call_samba(message, model_name=evaluate_model)
    elif 'gpt' in evaluate_model:
        response, _ = call_chat_gpt(message, model_name=evaluate_model)
    else:
        response, _ = call_vllm(message, model=evaluate_model, host=8000, stop='<|eot_id|>')
    printc(response, 'yellow')
    res = pathway_reasoning_answer_evaluator_response_parser(response)
    return {'score': res, 'score_response': response}
