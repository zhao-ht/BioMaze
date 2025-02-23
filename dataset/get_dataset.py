import math
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from datasets import load_dataset

from dataset.pathway_cot import pathway_judge_evaluator, pathway_judge_block_evaluator, pathway_choice_evaluator, \
    pathway_no_evaluation_evaluator, pathway_reasoning_answer_evaluator


class Batch_dataset(Dataset):
    def __init__(self, dataset, batch_size=None):
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        if self.batch_size is None:
            return 1
        else:
            return math.ceil(len(self.dataset) / self.batch_size)

    def __getitem__(self, index):
        if index >= self.__len__():
            raise StopIteration
        if self.batch_size is None:
            item_list = [item for item in self.dataset]
        else:
            item_list = [
                item
                for item in self.dataset[
                            (index * self.batch_size): (
                                min(len(self.dataset), (index + 1) * self.batch_size)
                            )
                            ]
            ]
        return item_list


def preprocess_pathway_dataset(dataset_pd):
    def eval_nona(item):
        if pd.isna(item):
            return item
        else:
            return eval(item)

    for key in dataset_pd.columns:
        try:
            dataset_pd[key] = dataset_pd[key].apply(eval_nona)
        except Exception as e:
            pass

    return dataset_pd


def get_dataset_part(dataset, i, n):
    """
    Split a PyTorch dataset into n parts and get the i-th part.

    Parameters:
    dataset (Dataset): the original dataset.
    i (int): the index of the part to get.
    n (int): the total number of parts.

    Returns:
    Subset: the i-th part of the dataset.
    """
    length = len(dataset)
    indices = list(range(length))
    part_size = math.ceil(length / n)

    if i < n - 1:
        part_indices = indices[i * part_size: (i + 1) * part_size]
    elif i == n - 1:
        # The last part includes all remaining data points.
        part_indices = indices[i * part_size:]
    else:
        part_indices = []

    print('Distributed Testing on Index {}'.format(part_indices))
    return Subset(dataset, part_indices)


def get_dataset(args):
    data_cleaner = None

    ds = load_dataset("haitengzhao/BioMaze")
    if args.task_type == 'judge':
        dataset_pd = ds['TrueFalse'].to_pandas()
    else:
        dataset_pd = ds['Openended'].to_pandas()

    if args.subcategory is not None:
        dataset_pd = dataset_pd[dataset_pd['subcategory'] == args.subcategory]

    dataset_pd['input'] = dataset_pd['question']
    dataset_pd = dataset_pd.reset_index()

    dataset = [item for key, item in dataset_pd.to_dict(orient='index').items()]

    test_index_key = "question"

    if args.task_type == 'judge':
        evaluator = pathway_judge_evaluator
    else:
        if args.no_evaluation:
            evaluator = pathway_no_evaluation_evaluator
        else:
            evaluator = lambda model_output, data: pathway_reasoning_answer_evaluator(model_output, data,
                                                                                      evaluate_model=args.evaluate_model)

    dataset = {"train": None, "val": None, "test": dataset}

    # test set permutation or subset

    if args.random_permutation:
        random.shuffle(dataset['test'])

    if args.subset_balance or args.subset_number is not None:
        if args.subset_number is not None and not args.subset_balance:
            dataset['test'] = random.sample(dataset['test'], args.subset_number)
        else:
            negative_subset = []
            positive_subset = []
            for item in dataset['test']:
                if item['answer'] == 'Yes':
                    positive_subset.append(item)
                else:
                    negative_subset.append(item)
            if args.subset_number is not None and args.subset_balance:
                dataset['test'] = random.sample(negative_subset, args.subset_number // 2) + random.sample(
                    positive_subset, args.subset_number // 2)
                random.shuffle(dataset['test'])
            elif args.subset_number is None and args.subset_balance:
                subset_number_half = min(len(negative_subset), len(positive_subset))
                dataset['test'] = random.sample(negative_subset, subset_number_half) + random.sample(
                    positive_subset,
                    subset_number_half)
                random.shuffle(dataset['test'])

    return {
        "dataset": dataset,
        "evaluator": evaluator,
        "data_cleaner": data_cleaner,
        "test_index_key": test_index_key,
    }
