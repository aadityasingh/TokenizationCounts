import os
from itertools import product
import pdb
import argparse
import pandas as pd
import re
from datetime import datetime
import json
import time
from types import SimpleNamespace

import numpy as np
import jax

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
import tiktoken

PRICES = {'gpt-4': {'prompt_tokens': .03/1000,
                    'completion_tokens': .06/1000},
          'gpt-3.5-turbo': {'prompt_tokens': .0015/1000,
                            'completion_tokens': .002/1000},
          # Model not supported anymore...
          # 'gpt-4-0314': {'prompt_tokens': .03/1000,
          #                'completion_tokens': .06/1000},
          'gpt-4-0613': {'prompt_tokens': .03/1000,
                         'completion_tokens': .06/1000},
          'gpt-4-1106-preview': {'prompt_tokens': .01/1000,
                                 'completion_tokens': .03/1000},
          'gpt-3.5-turbo-1106': {'prompt_tokens': .0015/1000,
                                 'completion_tokens': .002/1000},
          'gpt-3.5-turbo-0613': {'prompt_tokens': .0015/1000,
                                 'completion_tokens': .002/1000},
          'gpt-3.5-turbo-0301': {'prompt_tokens': .0015/1000,
                                 'completion_tokens': .002/1000}}

KEY_BRANCH = int(1e6)
SEPARATOR_NAMES = {'comma': ',',
                   'space': ' ',
                   'dec': '.',
                   'hashtag': '#',
                   'dollar': '$',
                   'foo': 'foo'}


def sample_digit_of_length(key, l):
    return int(jax.random.randint(key, minval=10**(l-1), maxval=10**l, shape=(1,)))


def sample_problem_with_lengths(key, lengths):
    '''
    key: random key to sample with
    length: list of 3 lengths.
    '''
    assert len(lengths) == 3
    assert (lengths[0] >= lengths[2] - 1) or (lengths[1] >= lengths[2] - 1)
    
    retval = [0,0]
    key_ans, key_short = jax.random.split(key)

    if lengths[0] < lengths[1]:
        short_ind = 0
    else:
        short_ind = 1
    long_ind = 1-short_ind

    # We may need to constrain shorter number in some cases.
    # e.g. [7,7,7], the shorter number must be less than 9000000
    # so that the answer can be 7 digits
    max_answer = 10**lengths[2]
    min_longer = 10**(lengths[long_ind]-1)
    max_shorter = max_answer - min_longer

    retval[short_ind] = int(jax.random.randint(key, minval=10**(lengths[short_ind]-1), 
                                                    maxval=min(10**lengths[short_ind], max_shorter), 
                                                    shape=(1,)))


    min_answer = max(10**(lengths[2]-1), retval[short_ind]+min_longer)
    max_answer = min(10**lengths[long_ind]+retval[short_ind], 10**lengths[2])

    answer = int(jax.random.randint(key_ans, minval=min_answer, maxval=max_answer, shape=(1,)))

    retval[long_ind] = answer - retval[short_ind]
    assert len(str(retval[long_ind])) == lengths[long_ind]
    assert (retval[0] >= 0) and (retval[1] > 0)

    return retval


def make_all_digit_length_controlled_sample_problem(key, all_lengths):
    '''
    all_lengths: _ x 3 array of problem lengths
    '''
    keys = jax.random.split(key, KEY_BRANCH)
    def sample(i):
        return sample_problem_with_lengths(keys[i], all_lengths[i%len(all_lengths)])
    return sample


def make_all_digit_length_same_sample_shots(key):
    keys = jax.random.split(key, KEY_BRANCH)
    def sample_shots(i, problem, num_shots):
        lengths = [len(str(n)) for n in problem] + [len(str(sum(problem)))]
        this_keys = jax.random.split(keys[i], num_shots)
        shots = [sample_problem_with_lengths(k, lengths) for k in this_keys]
        return shots
    return sample_shots


def make_digit_controlled_sample_problem(key, min_dig, max_dig):
    range_len = max_dig - min_dig
    temp = np.broadcast_to(np.arange(min_dig, max_dig), (range_len, range_len))
    all_digit_pairs = np.vstack([temp.T.reshape(-1), temp.reshape(-1)]).T
    keys = jax.random.split(key, KEY_BRANCH)
    def sample(i):
        this_keys = jax.random.split(keys[i], 2)
        return [sample_digit_of_length(k, l) for k, l in zip(this_keys, all_digit_pairs[i % all_digit_pairs.shape[0]])]
    return sample


def make_digit_same_sample_shots(key):
    keys = jax.random.split(key, KEY_BRANCH)
    def sample_shots(i, problem, num_shots):
        lengths = [len(str(n)) for n in problem]
        this_keys = jax.random.split(keys[i], num_shots)
        shots = []
        for ind in range(num_shots):
            problem_keys = jax.random.split(this_keys[ind], 2)
            shots.append([sample_digit_of_length(pk, l) for pk, l in zip(problem_keys, lengths)])
        return shots
    return sample_shots


def chunk_with_separator(string, chunk_size=3, separator='', direction='l2r'):
    if direction == 'l2r':
        chunks = [string[i:i+chunk_size] for i in range(0, len(string), chunk_size)]
        return separator.join(chunks)
    elif direction == 'r2l':
        return chunk_with_separator(string[::-1], chunk_size=chunk_size, separator=separator[::-1], direction='l2r')[::-1]


def make_format_number(padding='nosep', chunk_size=3):
    def format_number(n):
        pad_pieces = padding.split('_')
        if len(pad_pieces) == 2:
            return chunk_with_separator(str(n), 
                                        chunk_size=chunk_size, 
                                        separator=SEPARATOR_NAMES[pad_pieces[0]], 
                                        direction=pad_pieces[1])
        else:
            if padding == 'nosep':
                return str(n)
    return format_number


def make_format_addition_problem(
    num_spaces=0, 
    num_trailing_spaces=0,
    with_equals=True
):
    def format_addition_problem(s1, s2):
        spacing = ' '*num_spaces
        end_spacing = ' '*num_trailing_spaces
        query = s1 + spacing + '+' + spacing + s2
        if with_equals:
            query = query + spacing + '='
        return query + end_spacing
    return format_addition_problem


def make_safe_openai_api_call(client, waiting_interval_s=[2,9], max_api_tries_per_q=10):
    def api_call(
        model, 
        messages, 
        temperature=0, 
        max_tokens=10, 
        n=1, 
        logprobs=True,
        top_logprobs=5,
        seed=0,
        verbose=False,
        **kwargs):
        try_count = 0
        success = False
        if verbose:
            print("Submitted the following:")
            print('\n'.join(['{}: {}'.format(m['role'], m['content']) for m in messages]))
        while success is False and try_count < max_api_tries_per_q:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                    logprobs=True,
                    top_logprobs=5,
                    seed=0,
                    **kwargs,
                )
                success = True
            except Exception as e:
                # try to wait a few seconds and attempt again if hitting api error
                # try max_api_tries times, and if still failing then output "MODEL_ERROR"
                t = np.random.randint(waiting_interval_s[0], waiting_interval_s[1])
                try_count += 1
                print(f'Hit API error. Waiting {t}s and trying again. Try {try_count} of {max_api_tries_per_q}.')
                time.sleep(t) # waits random 2-9s
        if verbose:
            print("Received:")
            for c in range(n):
                print('({} of {}) {}: {}'.format(c+1, n, response.choices[c].message.role, response.choices[c].message.content))
        return response, success
    return api_call


def dummy_api_call_fn(
    model, 
    messages, 
    temperature=0, 
    max_tokens=10, 
    n=1, 
    verbose=False,
    **kwargs):
    '''Dummy API call function used for getting meta-experiment costs.'''
    if verbose:
        print("Dummied the following:")
        print('\n'.join(['{}: {}'.format(m['role'], m['content']) for m in messages]))

    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(role='', content=''), 
                                                        logprobs=SimpleNamespace(content=[]))]*n,
                                usage=SimpleNamespace(prompt_tokens=num_tokens_from_messages(messages),
                                                        completion_tokens=max_tokens))
    return response, True


def stringify_shot(problem):
    return '{}+{}={}'.format(problem[0], problem[1], sum(problem))


def destringify_shot(s):
    pieces = s.split('+')
    num1 = int(pieces[0])
    num2 = int(pieces[1].split('=')[0])
    ans = int(pieces[1].split('=')[1])
    assert num1 + num2 == ans
    return [num1, num2]


def jsonify_logprobs(logprobs_content):
    '''
    This function takes in the content of logprobs from a response and creates a json with all the info

    We can then dump this json in the csv. A bit messy, we'll want to figure out what we want to do with
    logprobs and then update with post proc probs.
    '''
    retval = []
    for token_element in logprobs_content:
        retval.append({'token': token_element.token, 
                        'logprob': token_element.logprob, 
                        'top_5_logprobs': [{'token': t.token, 'logprob': t.logprob} 
                                            for t in token_element.top_logprobs]
                        })
    return retval


# copied from https://platform.openai.com/docs/guides/chat/introduction
def num_tokens_from_messages(messages):
    """Returns the number of tokens used by a list of messages."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens -= 1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens 


FLIP_CONDS = [{'flip_problem': False, 'flip_shots': False},
                {'flip_problem': True, 'flip_shots': False},
                {'flip_problem': False, 'flip_shots': True},
                {'flip_problem': True, 'flip_shots': True}]


def run_addition_experiment(
    name, 
    model,
    api_call_fn,
    sample_problem_fn,
    format_problem_number_fn,
    format_answer_number_fn,
    format_problem_fn,
    format_problem_number_repeat_fn=None,
    format_problem_repeat_fn=None,
    sample_shots_for_problem_fn=None,
    problem_range=[0,45],
    flip_def=[True, False, False, False],
    num_shots=0,
    system_prompt='You are a helpful assistant.',
    prompt=None,
    ckpt_every=None,
    ckpt_path='results/',
    temperature=0, 
    max_tokens=10, 
    num_model_outputs=1, 
    verbose=False,
    pause_after_cost_estimate=False):
    '''
    name: name for exp
    model: model name
    api_call_fn: model, messages, gpt_api_kwargs -> response, success
                    The point of this function is to abstract away retry logic
    sample_problem_fn: i -> x,y
                        Function that returns a problem given an index, reproducibly
    format_problem_number_fn: int -> str
                                Formats the problem addends
    format_answer_number_fn: int -> str
                                Like format_problem_number_fn, but used to format answer number
                                in few-shot examples
    format_problem_fn: str, str -> str
                        Takes in two strings for the addends, and returns an addition problem string
    format_problem_number_repeat_fn: int -> str
                        Like format_problem_number_fn, but used to format model repeat of query
    format_problem_repeat_fn: str, str -> str
                        Like format_problem_fn, but used to format model repeat of query
    sample_shots_for_problem_fn: if specified, i, problem, num_shots -> list of len num_shots, with each
                                element [x,y]
    problem_range: range of indices to use on sample_problem_fn to generate problems. Inclusive->exclusive
    flip_def: boolean list of length four
                Which flip conditions to run, indexing into flip_conds. Per
                discussion, most likely just want [T,F,F,F], [T,T,F,F], [T,T,T,T],
                and maybeeee [T,F,F,T].
    num_shots: int, number of few shot prompts
    system_prompt: str, System prompt for LLM
    prompt: Prompt to being message with. If not specified, defaults no prompt
    ckpt_every: How often to dump checkpoints of results. If more than length of problem range,
                will just checkpoint at the end. 
    temperature: Temperature to pass to model
    max_tokens: Max tokens in output per api call
    num_model_outputs: choices paramter fed in api call -- for temperature 0, no reason to do >1
    verbose: bool, for printing out stuff
    pause_after_cost_estimate: Whether or not to pause after printing cost estimate
    '''
    cost_multiplier = sum(flip_def)
    flip_conds_to_use = [fc for j, fc in enumerate(FLIP_CONDS) if flip_def[j]]

    repeat_problems = False
    if format_problem_repeat_fn:
        assert format_problem_number_repeat_fn
        repeat_problems = True

    start_time_str = datetime.now().strftime("%Y%m%d%H%M%S")

    # Assemble preamble
    preamble_msg = [{"role": "system", "content": system_prompt}]
    if prompt is not None:
        preamble_msg.append({"role": "user", "content": prompt})

    get_query_from_problem = lambda p: format_problem_fn(*[format_problem_number_fn(number) for number in p])
    if repeat_problems:
        get_query_repeat_from_problem = lambda p: format_problem_repeat_fn(*[format_problem_number_repeat_fn(number) for number in p])

    dfs = []


    def add_subdf(ind, problem, shots, flip_cond, api_response):
        for c in range(num_model_outputs):
            dfs.append(pd.DataFrame({
                'problem_index': ind,
                'flip_problem': flip_cond['flip_problem'],
                'flip_shots': flip_cond['flip_shots'],
                'num1': problem[0], 
                'num2': problem[1],
                'problem_msg': get_query_from_problem(problem),
                'problem_tokens': len(tiktoken.get_encoding('cl100k_base').encode(get_query_from_problem(problem))),
                'response_repeat': c,
                'response': api_response.choices[0].message.content,
                'logprobs_json': json.dumps(jsonify_logprobs(api_response.choices[c].logprobs.content)),
                'prompt_tokens': api_response.usage.prompt_tokens,
                'completion_tokens': api_response.usage.completion_tokens,
                'shots':','.join([stringify_shot(shot) for shot in shots]),},
                index=[0]))


    def get_problem_query_info_from_index(i, flip_problem=False, flip_shots=False):
        problem = sample_problem_fn(i)
        shots_msg = []
        if sample_shots_for_problem_fn is not None:
            all_shots = sample_shots_for_problem_fn(i, problem, num_shots=num_shots)
            for shot in all_shots:
                shot_answer = sum(shot)
                if flip_shots:
                    shot = shot[::-1]
                shots_msg.append({"role": "user", "content": get_query_from_problem(shot)})
                if repeat_problems:
                    shots_msg.append({"role": "assistant",
                                      "content": get_query_repeat_from_problem(shot)+format_answer_number_fn(shot_answer)})    
                else:
                    shots_msg.append({"role": "assistant", "content": format_answer_number_fn(shot_answer)})
        else:
            assert num_shots == 0

        if flip_problem:
            problem = problem[::-1]
        problem_msg = [{"role": "user", "content": get_query_from_problem(problem)}]
        messages = preamble_msg+shots_msg+problem_msg
        return locals()

    token_count = sum([num_tokens_from_messages(get_problem_query_info_from_index(i)['messages']) for i in range(problem_range[0], problem_range[1])])
    min_cost = token_count * PRICES[model]['prompt_tokens']
    max_cost = min_cost + PRICES[model]['completion_tokens']*max_tokens*(problem_range[1]-problem_range[0])
    print("Esimated min cost: ${:.4f} and max cost: ${:.4f} of experiment...".format(cost_multiplier*min_cost, cost_multiplier*max_cost))

    if pause_after_cost_estimate:
        print("Press c to continue")
        pdb.set_trace()

    for i in range(problem_range[0], problem_range[1]):
        for cond in flip_conds_to_use:
            problem_info = get_problem_query_info_from_index(i, **cond)
            response, success = api_call_fn(model, 
                                            problem_info['messages'], 
                                            temperature=temperature,
                                            max_tokens=max_tokens,
                                            n=num_model_outputs,
                                            verbose=verbose)
            if success is False:
                print("Too many failed API calls... quitting to analysis...")
                break
            add_subdf(i, problem_info['problem'], problem_info['all_shots'], cond, response)

            if ckpt_every:
                if i > 0 and i % ckpt_every == 0:
                    df_so_far = pd.concat(dfs, ignore_index=True)
                    df_so_far.to_csv(ckpt_path+'{}_{}_ckpt.csv'.format(start_time_str, name))

    final_df = pd.concat(dfs, ignore_index=True)
    if ckpt_every:
        final_df.to_csv(ckpt_path+'{}_{}_ckpt.csv'.format(start_time_str, name))
    return final_df


if __name__ == '__main__':

    model = 'gpt-3.5-turbo-0301'

    api_call_fn = make_safe_api_call()

    sample_problem_fn = make_digit_controlled_sample_problem(jax.random.PRNGKey(0), 7, 10)

    format_problem_number_fn = make_format_number('space_l2r')

    format_answer_number_fn = make_format_number('comma_r2l')

    format_problem_fn = make_format_addition_problem()

    sample_shots_fn = make_digit_same_sample_shots(jax.random.PRNGKey(1))

    df = run_addition_experiment(
            'test', 
            model=model,
            api_call_fn=api_call_fn,
            sample_problem_fn=sample_problem_fn,
            format_problem_number_fn=format_problem_number_fn,
            format_answer_number_fn=format_answer_number_fn,
            format_problem_fn=format_problem_fn,
            sample_shots_for_problem_fn=sample_shots_fn,
            problem_range=[0,9],
            flip_def=[True, False, False, False],
            num_shots=5,
            system_prompt='You are a helpful assistant.',
            prompt=None,
            ckpt_every=5,
            temperature=0, 
            max_tokens=10, 
            num_model_outputs=2, 
            verbose=True,
            pause_after_cost_estimate=True)

    costs = dict()
    for k in PRICES[model]:
        costs[k] = df[df['response_repeat'] ==0][k].sum()*PRICES[model][k]
    print("Prompt cost: ${:.4f}, total cost: ${:.4f}".format(costs['prompt_tokens'], costs['prompt_tokens']+costs['completion_tokens']))
    
