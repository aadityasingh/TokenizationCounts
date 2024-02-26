# TokenizationCounts

## Overview

This is our code for the paper (Tokenization counts:  the impact of tokenization on arithmetic in frontier LLMs)[https://arxiv.org/abs/2402.14903]. It contains all files needed to reproduce the figures in the main paper, as well as raw results CSV's in `results/`. Please direct any questions and/or correspondence to aaditya.singh.21@ucl.ac.uk.

## Codebase structure

Overall, codebase has a few key components:

### `main.py:` Main backbone for running prompting experiments on GPT. 
- All sampling is random seeded through JAX for maximum reproducibility.
- Code is very functionally programmed -- the idea is that the functions in main.py can be used to construct various experiments. Each experiment then gets its own run_*.py file which imports everything from main. We made this choice since runs are easily reproducible/no issues of forgotten configs.
- Support functionality for sampling addition problems of a given length signature in `sample_problem_with_lengths` (e.g., `lengths = [7,7,8]` means a 7 digit + 7 digit -> 8 digit problem). 
- We build samplers on top of this that sample problems with various constraints in cyclic order.
    - `make_all_digit_length_controlled_sample_problem` cycles through a list of length constraints of dimension 3 (so both addends and the answer are constrained). This was used for our later experiments once we realized answer length matters a lot.
    - `make_digit_controlled_sample_problem` accepts a `min_dig` count and `max_dig` count for each of the 2 addends. Then, it cycles through all combinations (e.g., if `min_dig=7, max_dig=9`, this would be 9 possible combinations) and generates problems.
- We build corresponding sampler to sample *shots* for each of these problem. There are few variants of these, for the different experiment settings
    - `make_all_digit_length_same_sample_shots` samples shots with the exact same digit lengths as a given problem. Addends and answer are constrained (analogous to `make_all_digit_length_controlled_sample_problem`)
    - `make_digit_same_sample_shots` is similar to `make_all_digit_length_same_sample_shots`, but only uses input addend lengths. Thus, it corresopnds to `make_digit_controlled_sample_problem`
- Next, we have various functions that format numbers and addition problems
    - `chunk_with_separator` breaks up a number either L2R or R2L using a separator. For example, comma_r2l can be achieved by doing `chunk_with_separator(str(num), 3, 'r2l', ',')`
    - `make_format_number` builds a format function (maps number -> str) which wraps around `chunk_with_separator` for the conditions we care about. See `SEPARATOR_NAMES` for the separators that are supported by `make_format_number`
    - `make_format_addition_problem` builds a problem formatter (maps string1, string2 -> problem string) and takes into account
- `make_safe_api_call` builds a safe api call function and wraps the retry logic/parameters
- `dummy_api_call_fn` is a function with the same signature (input and output) as safe api call fns, except it doesn't actually query a model. Useful for confirming experiment setups without wasting time/money
- `stringify_shot`/`destringify_shot` used for storing shot information in csv's -- never used for actual model querying
- `jsonify_logprobs` similarly stores a string version of the response logprobs in csv's.
- `run_addition_experiment` A wrapper function that accepts various first order functions and parameters and actually conducts an experiment. See docstring for more information

### Experiments: `run_*.py`

These files were used to generate the individual csv's. Each run results in a dated csv file with all the necessary info placed in `results/{date}_{name}.csv`. Some files were used to create multiple csv's, since we parallelized over models (for example). Some files are also a bit older, so haven't been updated to support the new OpenAI API -- the `*_new_models.py` files have the correct syntax for this updated API.

Below, we detail all the columns that'll exist in these data frames as they're saved:
- `problem_index`: Identifier for the speficic problem -- useful for `groupby` if we want to compare different formats/conditions/shot etc. on the same problem
- `flip_problem`: Whether or not the problem was flipped (e.g. if it's written as a+b, was it fed in as a+b or b+a). Note, if `flip_problem` is True, the saved `num1, num2` will be flipped given the index as well.
- `flip_shots`: same as `flip_problem`, but for shots
- `num1, num2`: the addends for the problem
- `problem_msg`: The actual query string used (will take into account delimiter, thinking tokens etc.)
- `problem_tokens`: Length of `problem_msg` in tokens
- `response_repeat`: Should basically always be 0 for our current experiments. This tells us which index of the response it is, if we use `n > 1` when sampling from the API (which I don't think we ever ended up doing)
- `response`: The actual raw model response string
- `logprobs_json`: A jsonified version of the top 5 tokens + logprobs for each output token. Created from the OpenAI API response logprobs using `main.py:jsonify_logprobs`
- `prompt_tokens, completion_tokens`: The total number of tokens in the prompt + completion, directly from the response message, used for calculating costs posthoc if needed/checking if max length (which I think we default to 10) was hit etc.
- `shots`: A comma-separated string ith all relevant shot information (output from `main.py:stringify_shot` run on each shot). `main.py:destringify_shot` can be used to recover shots
- `model`: Model that was run on, OpenAI identifier.
- `system_prompt`: The actual system prompt used. Most notebooks convert this to `['custom', 'original']` in the field `system_prompt_type` -- see below.
- `problem_number_format`: Format string used for the numbers in the problem. Can be passed to `main.py:make_format_number` to get a format function
- `answer_number_format`: Format string. Can be passed to `main.py:make_format_number` to get a format function
- `repeat_style`: What format things were repeated in. Defaults to None (since most experiments don't involve repeats)
- `num_spaces`: How many spaces in between operators. This was used for `nosep+1, nosep+2` conditions for thinking token control
- `num_trailing_spaces`: How many spaces after the `=` sign. This was used for `nosep+2` to add another token
- `num_shots`: How many shots were used
- `problem_rng_key`: Original RNG seed used to run this experiment. In combination with `problem_index`, this should exactly reproduce experiments.
- `shots_rng_key`: Original RNG seed used to get shots for this experiment. In combination with the exact problems, shot function, and `problem_index`, should reproduce the shots used.

### Analyzing: `utils.py` + `paper_figures.ipynb`

`paper_figures.ipynb` can be used to reproduce all figures in our (paper)[https://arxiv.org/abs/2402.14903]. `utils.py` contains all our error analysis code and is heavily used by `paper_figures.ipynb`. Our analysis pipeline contains of a post processing step (relying on `utils.post_process_df` which relies on `utils.post_process_row`) and then an error analysis step (`utils.add_error_analysis` which relies on `analyze_row_error`). The error analysis step assumes a tokenization direction, so care should be taken when passing dataframes containing both L2R and R2L tokenized rows. Examples can be found in `paper_figures.ipynb`.

Post-processing, columns added:
- Facts about the problem:
	- `true_answer`: Correct ans, `num1+num2`
	- `formatted_true_answer`: answer format function applied to `true_answer`, useful for checking exact match of format and answer
	- `len1, len2`: Digit lengths of `num1, num2`
	- `len_true_ans`: Digit length of `true_answer`
	- `len_match_both`: True if both `len1, len2` equal `len_true_ans`
	- `len_match`: True if either of `len1, len2` equal `len_true_ans`
	- `num_carries`: How many total carries there were in the problem
	- `max_carry_chain`: The length of the longest chain of carries
- Repeat related:
	- `repeat_problem_number_format`: if `repeat_style` in df, will set this to be the correct (needed this step to convert when `repeat_style = 'same'`)
	- `repeat_valid`: Check if the response contains a repeat with two addends
	- `repeat_detailed_info`: A list of two dictionaries with raw processing of repeats. These individual variables do get distilled into the rest of the dataset entries, just including here for clarity. Containing the following fields for each repeat:
		- `padded`: True if model added spaces next to this number
		- `extracted_formatted`: String repeat including formatting
		- `extraneous`: True if model added other text to the number (beyond the expected formatting)
		- `number`: An int with the actual number repeat
		- `constructed_formatted`: The properly formatted number given the extraction
		- `correct_copy`: True if the number was correct (independent of format)
		- `correct_format`: True if the correct formatting was used (independent of number being correct!)
	- `repeat_padded`: True if either repeat contained padding
	- `repeat_extraneous`: True if either repeat added other text to the number (beyond the expected formatting)
	- `repeat_num1, repeat_num2`: The extracted repeated numbers
	- `repeat_copy_num1_match, repeat_copy_num2_match`: Whether or not the extracted repeated numbers matched the correct numbers
	- `repeat_copy_match`: True if both numbers matched (divorced from format)
	- `repeat_format_match`: True if both repeats used the correct format (again divorced from number correctness)
- Answer related:
	- `padded_answer`: True if the answer contained extraneous padding
	- `extracted_formatted_answer`: String of model response, including formatting. None if not valid
	- `valid_answer`: True if we were able to extract an answer. Typically False when model starts using text
	- `answer`: int of model response, extracted by removing all formatting
	- `constructed_formatted_answer`: model answer, reformatted to use the expected formatting (based on `answer_number_format`)
	- `extraneous_response`: True if there was extraneous text in model response
	- `format_match`: True if model answered in right format (even if it got wrong answer)
	- `answer_match`: True if model answered correctly (even if in wrong format)
	- `formatted_answer_match`: True if both `format_match` and `answer_match`

Error analysis, columns added:
- `len_ans`: The length of the model answer
- `correct_ans_len`: Checks if the model answered with a number of the correct length. Useful since a lot of the error analysis won't make sense if this is false.
- `abs_error_mag`: abs(true_answer - answer). Useful for checking off by one errors quickly (since just looking at digit error magnitudes can be misleading due to carries!)
- `only_off_by_one`: True if `abs_error_mag` is a power of 10
- `only_off_by_one_digit_num`: If `only_off_by_one` is true, this is the number of the digit, counting from the left, that the off by one error is at
- `digit_{1-L}_error_mag`: Error magnitude at a given digit, left-to-right. For example, error between 1239 and 1240 would be `[0, 0, 1, 9]`
- `digit_{1-L}_error`: Boolean if there's an error at this digit
- `digit_{i}_within_{t}token_error_mag`: Same as earlier error mag, but distinguished into tokens. Note here that `tokenize_dir` defaults to `l2r`
- `digit_{i}_within_{t}token_error`: Same as earlier error boolean, but distinguished into tokens. Note here that `tokenize_dir` defaults to `l2r`
- `token_{i}_error`: True if there's an error in the given token
- Token frequency-related error checking:
	- `token_{i}_error_more_frequent`: Can be True, False or NaN/None. If there's no error in this token, will be NaN/None. If the error made makes it such that the model response is a more frequent token (per `cl100k_base` merge ranks), then True. Otherwise False.
	- `token_{i}_match_most_frequent_off_by_one_last_digit`: Whether or not the most frequent token out of this token and this token +/- 1 was chosen by the model (so this should on average be 1/3). This is independent of whether or not the model got the problem right, but it can be combined with accuracy to get some cool stats.
	- `token_{i}_match_most_frequent_off_first_digit`: Same as above, except it compares all possible digit substitutions (so this should on average be 1/10) for the first digit of the token. 
	- `token_{i}_picked_rank_off_by_one_last_digit`: The rank (either 0, 1, or 2) of the model response out of the true answer token, and the true answer token +/- 1. If model response is not in any of these, returns None
	- `token_{i}_picked_rank_off_first_digit`: The rank (in 0..9 inclusive) of the model response out of the true answer token with any first digit. If model response is not in any of these, returns None.
- Logprobs related metrics:
	- `valid_logprobs_metrics`: True if we were able to successfully extract logprobs for all tokens, else False
	- `token_{i}_logprob`: Logprob of model response for the given token
	- `token_{i}_logprob_top_minus_second`: Gap between model's most likely output token and second most likely token
	- `token_{i}_true_answer_logprob`: If not None, the logprob of the correct token for this position
	- `token_{i}_true_answer_rank`: If not None, the rank of the correct token for this position (1 to 5, inclusive)
	- `token_{i}_lower_entropy`: Lower bound on entropy for the given token. Calculated by using the fact that all remaining tokens have probability less than that of the 5th most likely token. Specifically, $$- \left(\sum_{i=1}^5 p_i \log\left(p_i\right) + \left(1-\sum_{i=1}^5 p_i\right) \cdot \log\left(p_5\right)\right).$$
	- `token_{i}_top5_entropy`: Entropy over the top 5 tokens (after renormalizing to have total probability 1)
	- `answer_logprob`: Total logprob of the generated answer (only counting outputted number tokens)
	- `increasing_response_logprobs`: Whether or not logprob of the response is monotonically increasing over output number tokens
	- `decreasing_response_logprobs`: Whether or not logprob of the response is monotonically decreasing over output number tokens
	- `increasing_response_top5_entropy`: Whether or not `top5_entropy` of the response is monotonically increasing over output number tokens
	- `decreasing_response_top5_entropy`: Whether or not `top5_entropy` of the response is monotonically decreasing over output number tokens
	- `increasing_response_lower_entropy`: Whether or not `lower_entropy` of the response is monotonically increasing over output number tokens
	- `decreasing_response_lower_entropy`: Whether or not `lower_entropy` of the response is monotonically decreasing over output number tokens
- `num_carry_in_errors`: Number of errors when carrying in a 1
- `num_carry_out_errors`: Number of errors when carrying out a 1
- `dropped_digit_error`: True if model got the right answer as if it dropped a digit (calculated by an exhaustive check over all digit drops)
- Left alignment error checking, mostly used for single digit experiments:
	- `left_alignment_wrong_answer`: Answer as if model added inputs from the length, used in `left_alignment_error` and stored to allow for comparative coloring in error analysis notebooks.
	- `left_alignment_error`: True if model seems to have added the inputs from the left, i.e. zero-padding the shorter number up to length of the longer number and then adding them produces the model's answer. If inputs are same length, this is automaticaly False.
	- `{left, right}_chunked_error`: Checks if model seems to have treated the inputs as lists separated by their separators, and then added them element-wise. If both inputs have same number of chunks, left vs right doesn't matter. If one number has fewer chunks, then we can choose to align the lists from the left or the right. left_chunked_error checks for alignment from the left, and right_chunked_error from the right.

### Miscellaneous

`gpt_token_paper_figures.ipynb` + `claude_token_paper_figures.ipynb` were used to create the figures about GPT-3, GPT-3.5, and Claude's tokenizers in the paper. We separated this from `paper_figures.ipynb` as it does not rely on any of the experiments we ran and simply relies on the tokenizers provided by the respective creators of these models. The files also contain some additional investigation of these tokenizers (which we left in as we thought it may be interesting), so they're a bit less clean.

`utils.pretty_print` provides a nice tool for visualizing errors and carries for individual problems. We found this very useful when looking for error patterns.

We also include a fun python command line game (`tokenization_game.py`) we made where you can test your ability to rank tokens in the GPT-3.5 tokenizer. Enjoy :)
