from functools import partial
from typing import Callable
import multiprocessing as mp
from enum import Enum

from flask import Flask, request, jsonify
import numpy as np

from nemo_skills.code_execution import math_grader

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


@app.route('/math_grader', methods=['POST'])
def math_grader():
    """
    Endpoint to evaluate the "response" and "answer".
    Expects a JSON payload with "pred_responses" and "ground_truths" (batched).
    """
    try:
        with global_lock:
            # Parse JSON data from the request
            data = request.get_json()
            responses = data.get("pred_responses")
            answers = data.get("ground_truths")

            # Validate inputs
            if responses is None or answers is None:
                return jsonify({"error": "Both 'response' and 'answer' must be provided."}), 400
            if len(responses) != len(answers):
                return jsonify({"error": "responses and answers must have the same batch size"}), 400
            print(f"r {len(responses)}, a {len(answers)}", flush=True)
                
            print(f"s {submit_queue.qsize()}, o {result_queue.qsize()}")
            for inst_idx, (pred, gt) in enumerate(zip(responses, answers)):
                print(f"adding inst {inst_idx}", flush=True)
                submit_queue.put((WorkerSignal.RUN, inst_idx, (pred, gt)))
            
            rewards = np.zeros(len(responses))
            
            for _ in range(len(responses)):
                inst_idx, reward = result_queue.get()
                print(f"got inst {inst_idx}", flush=True)
                rewards[inst_idx] = reward

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
    app.run(host='0.0.0.0', port=5567, debug=True)
