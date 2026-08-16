import numpy as np
from collections import Counter
from utils.file_tool import json_read_to_dict

def calculate_mic(i_pred, i_gold):
    i_pred = set(i_pred)
    i_gold = set(i_gold)
    if len(i_gold) == 0:
        return 0.0
    intersection = i_pred & i_gold
    return len(intersection) / len(i_gold)

def instance_mic(pred_tasks, gold_tasks):
    pred_set=set(pred_tasks)
    gold_set=set(gold_tasks)
    covered=len(pred_set & gold_set)
    total=len(gold_set)
    if total==0:
        return 1.0
    return covered/total

def jaccard_similarity(pred, gold):
    pred_set = set(pred)
    gold_set = set(gold)
    intersection = pred_set & gold_set
    union = pred_set | gold_set

    if len(union) == 0:
        return 1.0
    return len(intersection) / len(union)

def counter_jaccard(pred, gold):
    pred_counter = Counter(pred)
    gold_counter = Counter(gold)
    labels = set(pred_counter.keys()) | set(gold_counter.keys())
    intersection = 0
    union = 0
    for label in labels:
        intersection += min(pred_counter[label], gold_counter[label])
        union += max(pred_counter[label], gold_counter[label])
    if union == 0:
        return 1.0
    return intersection / union

if __name__ == '__main__':
    intent_answer_s = json_read_to_dict('../mult_agents_evaluation/mult_agent_answer_s.json')
    mic_s = []
    jaccard_s = []
    for intent_answer in intent_answer_s:
        intents = intent_answer['intents']
        mult_agents_intents = intent_answer['mult_agents_intents']
        for i in range(len(intents)):
            intent = intents[i]
            mult_agents_intent = mult_agents_intents[i]
            intents_list = [part.strip() for part in intent.split(',')]
            mult_agents_intents_list = [part.strip() for part in mult_agents_intent.split(",")]
            mic = instance_mic(mult_agents_intents_list, intents_list)
            jaccard = counter_jaccard(mult_agents_intents_list, intents_list)
            mic_s.append(mic)
            jaccard_s.append(jaccard)
    print(np.mean(mic_s))
    print(np.mean(jaccard_s))
