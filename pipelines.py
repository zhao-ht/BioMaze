import copy
import json
import os.path
import tempfile
import time
from threading import Thread, BoundedSemaphore, Lock

import pandas as pd
from torch.utils.data import Subset
from tqdm import tqdm

from dataset.get_dataset import get_dataset_part
from sup_func.sup_func import pd_concat_ignore2


def answer_eval(response, dataset, data, evaluator=None, model_result_recoder=None):
    result = copy.copy(data)

    if response["res"] is None:
        score = 0
    else:
        score = evaluator(response["res"], data)

    if isinstance(score, dict):
        assert 'score' in score
        for key in score.keys():
            result[key] = score[key]
        score = score['score']
    else:
        result["score"] = score

    if model_result_recoder is None:
        # default response record items
        result["res"] = response["res"]
        if isinstance(response["gen"], dict):
            for key in response["gen"].keys():
                result[key] = response["gen"][key]
        else:
            result["generation"] = response["gen"]
    else:
        # model specific response record
        result = model_result_recoder(response, result)

    return score, result


def save_result_jsonl(file_name, result):
    with open(file_name, "a") as f:
        f.write(json.dumps(result) + "\n")
        f.flush()


def save_result_pd(file_name, result, sort_columns=False):
    # File index is not used
    df = pd.DataFrame([[result[key]
                        for key in result.keys()]], columns=result.keys())
    if os.path.exists(file_name):
        combined_data = pd_concat_ignore2(
            pd.read_csv(file_name, index_col=0), df
        )
    else:
        combined_data = df
    if sort_columns:
        combined_data = (
            combined_data.sort_index(axis=1)
        )

    target_dir = os.path.dirname(file_name)
    if target_dir and not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir)
        except FileExistsError:
            # For multiprocess, makedir may happen at the same time
            assert os.path.exists(target_dir)
    # Write to a temporary file first
    target_dir = os.path.dirname(file_name)
    with tempfile.NamedTemporaryFile(dir=target_dir, delete=False, mode='w', suffix='.csv') as temp_file:
        temp_file_name = temp_file.name
        combined_data.to_csv(temp_file_name, header=True)

    # Replace the original file with the temporary file
    os.replace(temp_file_name, file_name)


def resume_result_jsonl(file_name):
    lines = open(file_name).readlines()
    num_skip_exps = len(lines)
    for id, data in enumerate(map(json.loads, lines)):
        if "score" not in data:
            print(id, data)
    scores = [data["score"] for data in map(json.loads, lines)]
    return scores, num_skip_exps


def resume_result_pd(file_name, executed_column):
    resume = pd.read_csv(file_name)
    if "score" in resume:
        scores = resume["score"].values.tolist()
    elif "success" in resume:
        scores = (resume["success"] > 0).values.tolist()
    else:
        print("Warnning! No score or success in resumed file. No score resumed")
        scores = []
    if executed_column in resume:
        executed_samples = []
        for sample_list in resume[executed_column].values.tolist():
            if not pd.isna(sample_list):
                try:
                    samples = eval(sample_list)
                    if not isinstance(samples, list):
                        executed_samples.append(str(samples))
                    else:
                        for item in eval(sample_list):
                            executed_samples.append(str(item))
                except:
                    executed_samples.append(str(sample_list))
        executed_samples = set(executed_samples)
        num_skip_exps = len(executed_samples)
    else:
        num_skip_exps = len(resume)
        executed_samples = set()
    return scores, num_skip_exps, executed_samples


def test_single_sample(
        data, model, args, file_name, evaluator, is_parallel=False, ignore_error=False
):
    global scores, f, pbar

    if is_parallel or ignore_error:  # ignore error to release the process of parallel
        try:
            response = model(data)
        except Exception as e:
            print(e)
            response = {
                "res": None,
                "gen": None,
                "error": "0_test_single_sample_{}".format(e),
            }
    else:
        response = model(data)

    if is_parallel:
        lock.acquire()

    score, result = answer_eval(
        response, args.dataset_name, data, evaluator, model.result_recoder
    )

    scores.append(score)
    pbar.set_description(f"Total Score : {100 * sum(scores) / len(scores)}")

    save_result_pd(file_name, result)

    if is_parallel:
        lock.release()
        pool.release()


pool = BoundedSemaphore(4)
lock = Lock()


def testing(dataloader, model, args):
    global scores, pbar

    trial = 0
    test_index_key = dataloader["test_index_key"]

    OUTPUT_PATH = (
        args.eval_save_path
        if args.eval_save_path is not None
        else f"eval_results/{args.planning_method}.{args.model_name}/{args.dataset_name}.{args.split_dataset_num}_split_{args.split_file}.{args.exp_id}.{args.distributed_id}.csv"
    )

    print("Saving testing to {}".format(OUTPUT_PATH))

    if args.resume_from_merge:
        parts = OUTPUT_PATH.split('.')
        resume_path = '.'.join(parts[:-2] + ['None'] + [parts[-1]])
    else:
        resume_path = OUTPUT_PATH
    if args.resume and os.path.exists(resume_path):
        print("Resuming testing from {}".format(resume_path))
        scores, num_skip_exps, executed_samples = resume_result_pd(
            resume_path, test_index_key
        )
    else:
        scores = []
        num_skip_exps = 0
        executed_samples = set()
        if os.path.exists(OUTPUT_PATH):
            raise ValueError(
                "Eval result file exists. Cannot start a new testing. Please rename the eval result file {} first.".format(
                    OUTPUT_PATH
                )
            )

    print("Executed_samples: {}".format(len(executed_samples)))

    dataset = dataloader["dataset"]["test"]
    if test_index_key == 'index':
        left_index = [index for index in range(len(dataset)) if str(index) not in executed_samples]
    else:
        left_index = [index for index, data in enumerate(dataset) if str(data[test_index_key]) not in executed_samples]
    dataset = Subset(dataset, left_index)

    if args.distributed_test:
        dataset = get_dataset_part(dataset, args.distributed_id, args.distributed_number)

    pbar = tqdm(dataset)
    threads = []
    for data in pbar:
        trial += 1
        if not args.parallel_test:

            if dataloader["data_cleaner"] is not None and (
                    not dataloader["data_cleaner"](data)
            ):
                print("Dirty Data! Skip")
                continue
            test_single_sample(
                data,
                model,
                args,
                OUTPUT_PATH,
                dataloader["evaluator"],
                ignore_error=args.ignore_error,
            )
        else:
            if dataloader["data_cleaner"] is not None and (
                    not dataloader["data_cleaner"](data)
            ):
                print("Dirty Data! Skip")
                continue
            pool.acquire()
            thread = Thread(
                target=test_single_sample,
                args=(
                    data,
                    model,
                    args,
                    OUTPUT_PATH,
                    dataloader["evaluator"],
                    True,
                    args.ignore_error,
                ),
            )
            threads.append(thread)
            thread.start()

    for thread in threads:
        thread.join(300)  # 5 min for each task
        if thread.is_alive():
            print("A job didn't finish within the time limit")

    print(f"Total Score : {100 * sum(scores) / len(scores)}")
    print("Testing finished")
