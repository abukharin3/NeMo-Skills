# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

# Copyright (c) 2023 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright (c) 2021 Dan Hendrycks
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
This logic is largely copied from the Hendrycks' MATH release (math_equivalence), and borrowed from:
- https://github.com/microsoft/ToRA/blob/main/src/eval/grader.py
- https://github.com/microsoft/ProphetNet/tree/master/CRITIC
- https://github.com/openai/prm800k
"""


import contextlib
import re
import signal
from math import isclose
from typing import Union

from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr


def _fix_fracs(string):
    # replacing all extra spaces
    while "\\frac " in string:
        string = string.replace("\\frac ", "\\frac")
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    if "_" in x:
        # Due to base
        x = x.split("_")[0]
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _remove_right_units(expr):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text" in expr:
        try:
            splits = re.split(r"\\text\s*{\s*", expr)
            # print(splits)
            assert len(splits) == 2 and splits[0] not in ("", "(")
            return splits[0]
        except AssertionError:
            pass

    if "\\text{" in expr:
        return re.sub(r"\\text{([^}]+)}", r"\1", expr)
    elif "\\mbox{" in expr:
        splits = expr.split("\\mbox{")
        assert len(splits) == 2
        return splits[0]
    else:
        return expr


def _process_and_or_inside_text(string):
    string = re.sub(r"\s*\\text{\s*(or|and)\s*}\s*", ",", string)
    string = re.sub(r",\s*,", ",", string)
    return string


def _remove_left_and_right(expr):
    """Remove the right and left latex commands."""
    expr = re.sub(r"\\left", "", expr)
    expr = re.sub(r"\\right", "", expr)
    return expr


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\s*\w+)", r"\\sqrt{\1}", string)
    return _string


def _fix_interval(expr):
    """Fix interval expression."""
    if "\\in " in expr:
        return expr.split("\\in ")[1].strip()

    return expr


def _inject_implicit_mixed_fraction(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 \\frac{3}{4} => 7+3/4
    """
    p1 = re.compile(r"(\d+) *\\frac{(\d+)}{(\d+)}")

    def replacer(match):
        whole_part = match.group(1)
        numerator = match.group(2)
        denominator = match.group(3)

        if whole_part:
            return f"{whole_part} + {numerator}/{denominator}"
        else:
            return f"{numerator}/{denominator}"

    step = p1.sub(replacer, step)
    return step


def normalize_answer_string(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    expr = _remove_left_and_right(expr)
    expr = _process_and_or_inside_text(expr)
    expr = _remove_right_units(expr)
    expr = _fix_interval(expr)
    
    # Handle "and" and text separators in answers
    expr = re.sub(r'\\text{\s*and\s*}', ',', expr)
    expr = re.sub(r'\\text', '', expr)
    
    # Handle "x=" prefix in answers
    expr = re.sub(r'^x\s*=\s*', '', expr)
    
    # Handle grade notation (e.g., 12^{th} grade -> 12)
    expr = re.sub(r'(\d+)\^{\\mathrm{th}}', r'\1', expr)
    expr = re.sub(r'(\d+)\^{th}', r'\1', expr)
    expr = re.sub(r'(\d+)th', r'\1', expr)
    expr = re.sub(r'\\text{\s*grade\s*}', '', expr)
    expr = re.sub(r'grade', '', expr)
    
    # Handle backslash before negative sign and normalize spaces
    expr = expr.replace("\\-", "-")
    # Remove trailing backslashes and normalize spaces
    while expr.endswith('\\'):
        expr = expr[:-1]
    expr = re.sub(r'\s+', ' ', expr).strip()  # Normalize multiple spaces to single space

    for surround_str in ["\\\\text", "\\\\mathrm", "\\\\mathcal", "\\\\textbf", "\\\\textit"]:
        expr = expr.replace(surround_str, "")
        pattern = f"^{surround_str}" + "\{(?P<text>.+?)\}$"
        m = re.search(pattern, expr)
        if m is not None:
            expr = m.group("text")

    expr = expr.replace("\!", "")
    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace("^{\\circ}", "")

    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
        "p.m.",
        "PM",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)

    if "day" in expr:
        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        weekday_expressed = False
        for day in days:
            if day in expr:
                weekday_expressed = True
                break

        if not weekday_expressed:
            expr = re.sub(f"day(s)?", "", expr)

    expr = re.sub(f"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = _fix_sqrt(expr)

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    expr = _fix_fracs(expr)

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = _inject_implicit_mixed_fraction(expr)
    expr = expr.replace(" ", "")

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def is_digit(s):
    try:
        if "{,}" in str(s):
            num = float(str(s).replace("{,}", ""))
            return True, num

        num = float(str(s).replace(",", ""))
        return True, num
    except ValueError:
        return False, None


def normalize(answer) -> str:
    # checking if answer is $<number> and removing $ in that case to compare
    if isinstance(answer, str) and bool(re.match(r'\$\d+(\.\d+)?', answer)):
        return answer[1:]

    # checking if answer is <number>% or <number>\\% and removing %
    if isinstance(answer, str) and (
        bool(re.match(r'^\d+(\.\d+)?%$', answer)) or bool(re.match(r'^\d+(\.\d+)?\\%$', answer))
    ):
        return answer.replace("\\%", "").replace("%", "")

    return answer


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    tolerance: float = 1e-4,
    timeout: float = 10.0,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    
    prediction = normalize(prediction)
    reference = normalize(reference)
    

    # another round of normalization
    prediction = normalize_answer_string(prediction)
    reference = normalize_answer_string(reference)
    

    if isinstance(prediction, str) and len(prediction) > 1000:  # handling weird corner-cases
        prediction = prediction[:1000]

    # Handle plus-minus notation (±) by expanding into explicit lists
    if isinstance(reference, str) and "\\pm" in reference:
        # If reference is already a list/set, expand any \pm elements within it
        if ((reference.startswith('{') or reference.startswith('\\{')) and 
            (reference.endswith('}') or reference.endswith('\\}'))):
            
            # Extract elements from reference
            ref_content = reference.replace('\\{', '{').replace('\\}', '}')
            ref_elements = [elem.strip() for elem in ref_content[ref_content.find('{')+1:ref_content.rfind('}')].split(',')]
            
            # Process reference elements, expanding any \pm notation
            expanded_ref_elements = []
            for elem in ref_elements:
                if "\\pm" in elem:
                    pm_match = re.search(r'(.*?)\\pm(.*)', elem)
                    if pm_match:
                        before = pm_match.group(1).strip()
                        after = pm_match.group(2).strip()
                        expanded_ref_elements.append(f"{before}+{after}")
                        expanded_ref_elements.append(f"{before}-{after}")
                else:
                    expanded_ref_elements.append(elem)
            
            # Create a new reference with expanded elements
            reference = "{" + ", ".join(expanded_ref_elements) + "}"
        
        # If reference is a single expression with \pm, convert to a list with both versions
        else:
            pm_match = re.search(r'(.*?)\\pm(.*)', reference)
            if pm_match:
                before_pm = pm_match.group(1).strip()
                after_pm = pm_match.group(2).strip()
                
                # Create the plus and minus versions
                plus_version = f"{before_pm}+{after_pm}"
                minus_version = f"{before_pm}-{after_pm}"
                
                # Replace the reference with a list containing both versions
                reference = "{" + plus_version + ", " + minus_version + "}"
    
    # Similarly handle \pm in prediction
    if isinstance(prediction, str) and "\\pm" in prediction:
        # If prediction is already a list/set, expand any \pm elements within it
        if ((prediction.startswith('{') or prediction.startswith('\\{')) and 
            (prediction.endswith('}') or prediction.endswith('\\}'))):
            
            # Extract elements from prediction
            pred_content = prediction.replace('\\{', '{').replace('\\}', '}')
            pred_elements = [elem.strip() for elem in pred_content[pred_content.find('{')+1:pred_content.rfind('}')].split(',')]
            
            # Process prediction elements, expanding any \pm notation
            expanded_pred_elements = []
            for elem in pred_elements:
                if "\\pm" in elem:
                    pm_match = re.search(r'(.*?)\\pm(.*)', elem)
                    if pm_match:
                        before = pm_match.group(1).strip()
                        after = pm_match.group(2).strip()
                        expanded_pred_elements.append(f"{before}+{after}")
                        expanded_pred_elements.append(f"{before}-{after}")
                else:
                    expanded_pred_elements.append(elem)
            
            # Create a new prediction with expanded elements
            prediction = "{" + ", ".join(expanded_pred_elements) + "}"
        
        # If prediction is a single expression with \pm, convert to a list with both versions
        else:
            pm_match = re.search(r'(.*?)\\pm(.*)', prediction)
            if pm_match:
                before_pm = pm_match.group(1).strip()
                after_pm = pm_match.group(2).strip()
                
                # Create the plus and minus versions
                plus_version = f"{before_pm}+{after_pm}"
                minus_version = f"{before_pm}-{after_pm}"
                
                # Replace the prediction with a list containing both versions
                prediction = "{" + plus_version + ", " + minus_version + "}"

    # Handle case where prediction is a list but reference is a single value
    # In this case, compare only the last element of the prediction
    # if isinstance(prediction, str) and "," in prediction and "," not in str(reference):
    #     pred_parts = [item.strip() for item in prediction.split(",")]
    #     # Take the last element of the prediction
    #     prediction = pred_parts[-1]

    # For comma-separated lists, compare sorted lists
    if isinstance(prediction, str) and isinstance(reference, str) and "," in prediction and "," in reference:
        # Extract elements, ignoring brackets
        pred_items = [item.strip() for item in prediction.replace("{", "").replace("}", "").replace("\\", "").split(',')]
        ref_items = [item.strip() for item in reference.replace("{", "").replace("}", "").replace("\\", "").split(',')]
        
        # Sort both lists to handle different orders
        pred_items_sorted = sorted(pred_items)
        ref_items_sorted = sorted(ref_items)
        
        # Try direct string comparison of sorted lists
        if pred_items_sorted == ref_items_sorted:
            return True
        

    # 0. string comparison
    if isinstance(prediction, str) and isinstance(reference, str):
        if prediction.strip().lower() == reference.strip().lower():
            return True
        if prediction.replace(" ", "") == reference.replace(" ", ""):
            return True

    try:  # 1. numerical equal
        if is_digit(prediction)[0] and is_digit(reference)[0]:
            prediction = is_digit(prediction)[1]
            reference = is_digit(reference)[1]
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if isclose(item, prediction, rel_tol=tolerance):
                        return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## deal with [], (), {}
    prediction = format_intervals(prediction)

    pred_str, ref_str = prediction, reference
    if (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")) or (
        prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str == ref_str:
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if (
        prediction
        and reference
        and prediction[0] in "(["
        and prediction[-1] in ")]"
        and reference[0] in "(["
        and reference[-1] in ")]"
        and prediction[0] == reference[0]
        and prediction[-1] == reference[-1]
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(pred_pt, ref_pt, include_percentage, tolerance)
                    for pred_pt, ref_pt in zip(pred_parts, ref_parts)
                ]
            ):
                return True

    # if we have point == tuple of values
    if prediction.startswith("Point") and reference[0] == "(" and reference[-1] == ")":
        pred_parts = prediction[prediction.find("(") + 1 : -1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(pred_pt, ref_pt, include_percentage, tolerance)
                    for pred_pt, ref_pt in zip(pred_parts, ref_parts)
                ]
            ):
                return True

    # Handle matrix/vector comparisons better
    if (("\\begin{pmatrix}" in reference or "\\begin{bmatrix}" in reference) and 
        ("\\begin{pmatrix}" in prediction or "\\begin{bmatrix}" in prediction)):
        # Extract matrix elements
        ref_pattern = r"\\begin\{[pb]matrix\}(.*?)\\end\{[pb]matrix\}"
        pred_pattern = r"\\begin\{[pb]matrix\}(.*?)\\end\{[pb]matrix\}"
        
        ref_match = re.search(ref_pattern, reference, re.DOTALL)
        pred_match = re.search(pred_pattern, prediction, re.DOTALL)
        
        if ref_match and pred_match:
            ref_content = ref_match.group(1).strip()
            pred_content = pred_match.group(1).strip()
            
            # Split by rows and columns
            ref_rows = [row.strip() for row in ref_content.split("\\\\")]
            pred_rows = [row.strip() for row in pred_content.split("\\\\")]
            
            if len(ref_rows) == len(pred_rows):
                all_equal = True
                for i in range(len(ref_rows)):
                    ref_cols = [col.strip() for col in ref_rows[i].split("&")]
                    pred_cols = [col.strip() for col in pred_rows[i].split("&")]
                    
                    if len(ref_cols) != len(pred_cols):
                        all_equal = False
                        break
                    
                    for j in range(len(ref_cols)):
                        if not math_equal(ref_cols[j], pred_cols[j], include_percentage, tolerance):
                            all_equal = False
                            break
                    
                    if not all_equal:
                        break
                
                if all_equal:
                    return True

    # if reference is a matrix
    if reference.startswith("\\begin{pmatrix}") and prediction.startswith("Matrix"):
        try:
            pred_matrix = parse_expr(prediction)
            ref_matrix_items = reference.split()[1:-1:2]
            if len(pred_matrix) == len(ref_matrix_items):
                if all(
                    [
                        math_equal(ref, pred, include_percentage, tolerance)
                        for ref, pred in zip(ref_matrix_items, pred_matrix)
                    ]
                ):
                    return True
        except Exception:
            pass

    return symbolic_equal(prediction, reference, tolerance, timeout)


def symbolic_equal(a, b, tolerance, timeout=10.0):
    def _parse(s):
        for f in [parse_expr, parse_latex]:
            try:
                with time_limit(timeout):
                    return f(s)
            except Exception:
                pass
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        with time_limit(timeout):
            if simplify(a - b) == 0:
                return True
    except Exception:
        pass

    try:
        with time_limit(timeout):
            if isclose(N(a), N(b), rel_tol=tolerance):
                return True
    except Exception:
        pass
    return False


def extract_answer(string, extract_from_boxed: bool = True, extract_regex: str = r"The final answer is (.+)$):"):
    """Extract Answer String from \\boxed expression."""
    # Handle multiple boxed answers
    boxed_answers = []
    
    # Find all occurrences of \boxed{...} or \fbox{...}
    start_idx = 0
    while start_idx < len(string):
        idx = string.find("\\boxed", start_idx)
        if idx < 0:
            idx = string.find("\\fbox", start_idx)
            if idx < 0:
                break
        
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1
        
        if right_brace_idx is not None:
            boxed_content = string[idx:right_brace_idx + 1]
            left = "\\boxed{"
            if boxed_content.startswith(left) and boxed_content.endswith("}"):
                boxed_answers.append(boxed_content[len(left):-1])
            
        start_idx = i + 1 if right_brace_idx is not None else len(string)
    
    # If we found multiple boxed answers, join them with commas
    if len(boxed_answers) > 1:
        return ", ".join(boxed_answers)
    elif len(boxed_answers) == 1:
        return boxed_answers[0]
    
    # Original single boxed answer extraction logic as fallback
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    if retval:
        left = "\\boxed{"
        try:
            assert retval[: len(left)] == left
            assert retval[-1] == "}"
            return retval[len(left) : -1]
        except AssertionError:
            return None

    return None


class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def format_intervals(prediction):
    patterns = {
        "Interval(": r"^Interval\((.*)\)$",
        "Interval.Ropen(": r"^Interval\.Ropen\((.*)\)$",
        "Interval.Lopen(": r"^Interval\.Lopen\((.*)\)$",
        "Interval.open(": r"^Interval\.open\((.*)\)$",
    }

    for key, pattern in patterns.items():
        match = re.match(pattern, prediction)
        if match:
            inner_content = match.group(1)

            if key == "Interval(":  # Intarval(a, b) == [a, b]
                return f"[{inner_content}]"
            elif key == "Interval.Ropen(":  # Intarval.Ropen(a, b) == [a, b)
                return f"[{inner_content})"
            elif key == "Interval.Lopen(":  # Intarval.Lopen(a, b) == (a, b]
                return f"({inner_content}]"
            elif key == "Interval.open(":  # Intarval.open(a, b) == (a, b)
                return f"({inner_content})"

    return prediction
