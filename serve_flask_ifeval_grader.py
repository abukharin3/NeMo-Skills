from functools import partial
from typing import Callable
import multiprocessing as mp
from enum import Enum

from flask import Flask, request, jsonify
import numpy as np

from nemo_skills.code_execution import math_grader
from instruction_following.instructions_registry import INSTRUCTION_DICT

app = Flask(__name__)

PROCESS_COUNT=32

class WorkerSignal(Enum):
    RUN=0
    QUIT=1


def extract_and_check(pred_sentence: str,
                      ground_truth: str,
                      extract_from_boxed:bool=True,
                      extract_regex:str=None,
                      **kwargs):
    try:
        pred_output = math_grader.extract_answer(pred_sentence, extract_from_boxed=extract_from_boxed, extract_regex=extract_regex)
        if isinstance(pred_output, str):
            pred_output = pred_output.replace("'''", r'\'\'\'')
            while pred_output.endswith('\\'):
                pred_output = pred_output[:-1]

        if isinstance(ground_truth, str):
            ground_truth = ground_truth.replace("'''", r'\'\'\'')
            while ground_truth.endswith('\\'):
                ground_truth = ground_truth[:-1]
        
        result = math_grader.math_equal(pred_output,
                                      ground_truth,
                                      include_percentage=kwargs.get("include_percentage", True),
                                      tolerance=kwargs.get("tolerance", 0.0001),
                                      timeout=kwargs.get("timeout", 10)) * 1.0
        print(pred_output, ground_truth, result)
        return result
    except Exception as e:
        print(f"extract_and_check had an error. {e} Skipping problem (returning incorrect)", flush=True)
        return False



def math_check_mp_worker(check_fn: Callable,
                         input_queue: mp.Queue,
                         output_queue: mp.Queue):
    while True:
        signal, idx, args = input_queue.get()
        if signal == WorkerSignal.RUN:
            check_results = check_fn(*args)
            output_queue.put((idx, check_results))
        else:
            return

submit_queue = mp.Queue()
result_queue = mp.Queue()
global_lock = mp.Lock()
print("QUEUES BUILT")
procs = [mp.Process(target=math_check_mp_worker, 
                           args=(extract_and_check, submit_queue, result_queue))
                           for _ in range(PROCESS_COUNT)]
print("PROCS BUILT")
workers = []
for p in procs:
    print("STARTING PROC")
    p.start()
    workers.append(p)


@app.route('/ifeval_grader', methods=['POST'])
def math_grader():
    """
    Endpoint to evaluate the "response" and "answer".
    Expects a JSON payload with "prompt","pred_responses" and "args" (batched).
    """
    try:
        with global_lock:
            # Parse JSON data from the request
            data = request.get_json()
            responses = data.get("pred_responses")
            args = data.get("args")
            prompts = data.get("prompts")

            if len(responses) != len(args):
                return jsonify({"error": "responses and args must have the same batch size"}), 400
            print(f"r {len(responses)}, a {len(args)}", flush=True)


            scores = []
            for i in range(len(responses)):
                prompt = prompts[i]
                args_i = args[i]
                response = responses[i]
                response = response.split("</think>")[-1]
                # Validate inputs
                try:
                    task_args = args_i
                    instruction_list = task_args["instruction_id_list"]
                    is_following_list = []

                    for index, instruction_id in enumerate(instruction_list):
                        try:
                            instruction_cls = INSTRUCTION_DICT[instruction_id]
                            instruction = instruction_cls(instruction_id)

                            kwargs = (
                                task_args["instruction_kwargs"][index]
                                if task_args["instruction_kwargs"][index] is not None
                                else {}
                            )
                            instruction.build_description(**kwargs)
                            instruction_args = instruction.get_instruction_args()
                            if instruction_args and "prompt" in instruction_args:
                                instruction.build_description(prompt=prompt)

                            if response.strip() and instruction.check_following(response):
                                is_following_list.append(True)
                            else:
                                is_following_list.append(False)
                        except Exception as e:
                            print(f"Error in instruction_following_rewards: {e}, task: {args}")
                            scores.append(0)
                    low, high = 0, 1
                    correctness = sum(is_following_list) / len(is_following_list)
                    score = low + (high - low) * correctness
                    scores.append(score)
                except Exception as e:
                    print(f"Error in instruction_following_rewards: {e}")
                    scores.append(0)

            rewards = np.array(scores)

            output_dict = {
                "rewards": rewards.reshape((rewards.shape[0], 1)).tolist(),
            }

            # Return the result as JSON
            return jsonify(output_dict)
    
    except Exception as e:
        app.logger.error("An error occurred: %s", str(e), exc_info=True)
        print("ERROR", str(e), flush=True)

        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5568, debug=True)
