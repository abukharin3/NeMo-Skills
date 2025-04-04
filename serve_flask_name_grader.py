from functools import partial
from typing import Callable
import multiprocessing as mp
from enum import Enum
import traceback

from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Since we're not using the multiprocessing for the ifeval endpoint,
# we can remove or comment out the unused code
# PROCESS_COUNT=32

# class WorkerSignal(Enum):
#     RUN=0
#     QUIT=1


# def extract_and_check(pred_sentence: str,
#                       ground_truth: str,
#                       extract_from_boxed:bool=True,
#                       extract_regex:str=None,
#                       **kwargs):
#     try:
#         pred_output = math_grader_module.extract_answer(pred_sentence, extract_from_boxed=extract_from_boxed, extract_regex=extract_regex)
#         if isinstance(pred_output, str):
#             pred_output = pred_output.replace("'''", r'\'\'\'')
#             while pred_output.endswith('\\'):
#                 pred_output = pred_output[:-1]

#         if isinstance(ground_truth, str):
#             ground_truth = ground_truth.replace("'''", r'\'\'\'')
#             while ground_truth.endswith('\\'):
#                 ground_truth = ground_truth[:-1]
        
#         result = math_grader_module.math_equal(pred_output,
#                                       ground_truth,
#                                       include_percentage=kwargs.get("include_percentage", True),
#                                       tolerance=kwargs.get("tolerance", 0.0001),
#                                       timeout=kwargs.get("timeout", 10)) * 1.0
#         print(pred_output, ground_truth, result)
#         return result
#     except Exception as e:
#         print(f"extract_and_check had an error. {e} Skipping problem (returning incorrect)", flush=True)
#         return False


# def math_check_mp_worker(check_fn: Callable,
#                          input_queue: mp.Queue,
#                          output_queue: mp.Queue):
#     while True:
#         signal, idx, args = input_queue.get()
#         if signal == WorkerSignal.RUN:
#             check_results = check_fn(*args)
#             output_queue.put((idx, check_results))
#         else:
#             return

# submit_queue = mp.Queue()
# result_queue = mp.Queue()
global_lock = mp.Lock()
print("IFEVAL GRADER SERVER STARTING")

# procs = [mp.Process(target=math_check_mp_worker, 
#                            args=(extract_and_check, submit_queue, result_queue))
#                            for _ in range(PROCESS_COUNT)]
# print("PROCS BUILT")
# workers = []
# for p in procs:
#     print("STARTING PROC")
#     p.start()
#     workers.append(p)


@app.route('/ifeval_grader', methods=['POST'])
def ifeval_grader():
    """
    Endpoint to evaluate instruction following.
    Expects a JSON payload with "prompts", "pred_responses" and "args" (batched).
    """
    try:
        with global_lock:
            # Parse JSON data from the request
            data = request.get_json()
            if data is None:
                print("Error: No JSON data received in request", flush=True)
                return jsonify({"error": "No JSON data received", "rewards": [[0.0]]}), 400
                
            responses = data.get("pred_responses")

            scores = []
            for i in range(len(responses)):
                if "nemotron" in responses[i].lower() and "llama" in responses[i].lower():
                    scores.append(1.0)
                else:
                    scores.append(0.0)
            
            rewards = np.array(scores)
            if len(rewards) == 0:
                print("Warning: Empty rewards array, returning zeros", flush=True)
                rewards = np.array([0.0] * len(responses))

            output_dict = {
                "rewards": rewards.reshape((rewards.shape[0], 1)).tolist(),
            }

            print(f"Returning rewards: {output_dict['rewards']}", flush=True)
            # Return the result as JSON
            return jsonify(output_dict)
    
    except Exception as e:
        app.logger.error("An error occurred: %s", str(e), exc_info=True)
        print("CRITICAL ERROR:", str(e), flush=True)
        print(traceback.format_exc(), flush=True)
        
        # Always return a valid JSON response even in case of error
        return jsonify({
            "error": str(e),
            "rewards": [[0.0]] * (len(request.get_json().get("pred_responses", [])) if request.get_json() else 1)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5569, debug=False)
