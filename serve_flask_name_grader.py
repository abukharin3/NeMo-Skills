from functools import partial
from typing import Callable
import multiprocessing as mp
from enum import Enum
import traceback

from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)


global_lock = mp.Lock()
print("NAME GRADER SERVER STARTING")


@app.route('/name_grader', methods=['POST'])
def name_grader():
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
