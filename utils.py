import tiktoken
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from matplotlib import colormaps
import re
from typing import Optional
import pdb
from functools import partial
import json

import main # oops feels weird

# Conventions:
# response = raw response string
# true_answer = true integer answer
# answer = parsed integer from model response

WIDE_DIGITS = {'0': '０',
                '1': '１',
                '2': '２',
                '3': '３',
                '4': '４',
                '5': '５',
                '6': '６',
                '7': '７',
                '8': '８',
                '9': '９'}


def subselect_df_from_dict(df, d):
    mask = None
    for k in d:
        if isinstance(d[k], list):
            val = df[k].isin(d[k])
        else:
            val = df[k] == d[k]
        if mask is None:
            mask = val
        else:
            mask = mask & val
    return df[mask]


def tokenize(s: str, tokenizer: str = 'cl100k_base'):
    '''
    Performs a bunch of tokenization-related functions and returns dictionary output
    '''
    retval = dict()
    t = tiktoken.get_encoding(tokenizer)
    # Note that token indices are the ranks
    retval['ranks'] = t.encode(s)
    retval['tokens'] = [t.decode([e]) for e in retval['ranks']]
    retval['count'] = len(retval['ranks'])
    return retval


def color_ansi(text, text_color=None, bg_color=None):
    '''
    Colors text for displaying in terminal using ANSI code.

    Args:
        text - text to be displayed
        text_color - color to use on text, RGB len 3 array
        bg_color - background to display text on, RGB len 3 array
    '''
    pieces = []
    if text_color is not None:
        if np.all(text_color <= 1):
            text_color = (text_color * 255)
        text_color = text_color.astype(int)
        pieces.append(f"\033[38;2;{text_color[0]};{text_color[1]};{text_color[2]}m")
    if bg_color is not None:
        if np.all(bg_color <= 1):
            bg_color = (bg_color * 255)
        bg_color = bg_color.astype(int)
        pieces.append(f"\033[48;2;{bg_color[0]};{bg_color[1]};{bg_color[2]}m")
    pieces.append(text)
    if (text_color is not None) or (bg_color is not None):
        pieces.append("\033[0m")
    return "".join(pieces)


def color_html(text, text_color=None, bg_color=None):
    '''
    Colors text for displaying with HTML (typically in notebooks/pandas tables)

    Args:
        text - text to be displayed
        text_color - color to use on text, RGB len 3 array
        bg_color - background to display text on, RGB len 3 array
    '''
    style = ['style="']
    colored = False
    if text_color is not None:
        colored=True
        if np.all(text_color <= 1):
            text_color = (text_color * 255)
        text_color = text_color.astype(int)
        style.append(f"color:rgb{tuple(text_color)[:3]};")
    if bg_color is not None:
        colored=True
        if np.all(bg_color <= 1):
            bg_color = (bg_color * 255)
        bg_color = bg_color.astype(int)
        style.append(f"background-color:rgb{tuple(bg_color)[:3]};")
    
    if colored:
        return f'<span {"".join(style)}">{text}</span>'
    else:
        return text


def split_string(s, length=3, direction='l2r'):
    '''
    Util for splitting string into pieces
    '''
    if direction == 'l2r':
        return [s[i:i+length] for i in range(0, len(s), length)]
    else:
        return [sub[::-1] for sub in split_string(s[::-1], length=length, direction='l2r')][::-1]


def calculate_carries(num1, num2):
    '''
    Takes two numbers as input and outputs an array with len=max(len(num1),len(num2))
    which indicates whether or not those two digits had a carry out
    '''
    retval = [0]*(max(int(np.log10(num1)), int(np.log10(num2)))+1)
    carry = 0
    for i in range(len(retval), 0, -1):
        v = (num1 % 10) + (num2 % 10) + carry
        if v >= 10:
            carry = 1
        else:
            carry = 0
        retval[i-1] = carry
        num1 = num1//10
        num2 = num2//10
    return retval


def calculate_errors(answer, true_answer, justify_answer='right'):
    '''
    Returns a len(answer) array of error magnitudes

    Note if true_answer is longer than answer due to a leading or trailing
    digit deletion in answer, this function will seem to indicate no errors.
    Length checking should be handled by the parent function
    '''
    if justify_answer == 'right':
        retval = [0]*(int(np.log10(answer))+1)
        for i in range(len(retval), 0, -1):
            retval[i-1] = abs((answer % 10) - (true_answer % 10))
            answer = answer//10
            true_answer = true_answer//10
        return retval
    else:
        return calculate_errors(int(str(answer)[::-1]), 
                                int(str(true_answer)[::-1]), 
                                justify_answer='right')[::-1]


def pretty_print(num1, num2, answer, 
                    color_fn=color_ansi,
                    tokenize_dir='l2r',
                    justify_answer='right',
                    n_tok_colors=3,
                    bg_cmap='Dark2',
                    rank_cmap='Greys',
                    rank_cmap_pow=1,
                    color_err_tok_by_corr_rank=True):
    '''
    A util function for nicely printint out an addition problem with errors
    '''

    strings = [str(num1), str(num2), str(num1+num2), str(answer)]

    bg_color_inds = []
    tokens = dict()
    token_ind = 0
    for s in strings:
        n_toks = (len(s)-1)//3+1
        arr = np.stack([np.arange(token_ind, token_ind+n_toks, dtype=int)]*3,axis=1).reshape(-1)
        if tokenize_dir == 'l2r':
            arr = arr[:len(s)]
        else:
            arr = arr[-len(s):]
        bg_color_inds.append(arr)

        for i, sub in enumerate(split_string(s, length=3, direction=tokenize_dir)):
            tokens[token_ind+i] = sub

        token_ind += n_toks
        token_ind = n_tok_colors*(1+(token_ind-1)//n_tok_colors)

    bg_colors = colormaps[bg_cmap](np.arange(n_tok_colors))

    def text_color_from_token(t):
        raw = tokenize(t)['ranks'][0]/28384 # rarest # token is ind 28384
        return np.array(colormaps[rank_cmap](raw**rank_cmap_pow)[:3])

    max_len = max([len(s) for s in strings])

    carries = calculate_carries(num1, num2)
    if len(carries) == max_len:
        assert carries[0] == 0
        carries = carries[1:]
    carries = carries + [0]

    print('   '*(max_len-len(carries)) + ''.join([' '+str(c)+' ' for c in carries]))

    def print_line(si, s):
        # print(' '.join([' ']*(max_len-len(s)) + [color_fn(c, bg_color=bg_colors[bg_color_inds[si][i] % len(bg_colors)]) for i, c in enumerate(s)]))
        print('   '*(max_len-len(s)) + 
                ''.join([color_fn(' '+c+' ', 
                                    text_color=text_color_from_token(tokens[bg_color_inds[si][i]]),
                                    bg_color=bg_colors[bg_color_inds[si][i] % len(bg_colors)]) 
                            for i, c in enumerate(s)]))

    for si, s in enumerate(strings[:2]):
        print_line(si, s)

    print('-'*max_len*3)
    print_line(2, strings[2])

    error_colors = colormaps['gist_rainbow'](np.arange(9)/9)[:, :3]
    errors = calculate_errors(answer, num1+num2, justify_answer=justify_answer)
    token_has_error = dict()
    for e, ind in zip(errors, bg_color_inds[3]):
        token_has_error.setdefault(ind, False)
        if e > 0:
            token_has_error[ind] = True

    digits = []
    offsets = [0,0.1]
    for i, c in enumerate(strings[3]):
        t_ind = bg_color_inds[3][i]
        if errors[i] > 0:
            text_c = error_colors[errors[i]-1]
        else:
            text_c = text_color_from_token(tokens[t_ind])
        if token_has_error[t_ind]:
            bg_c = np.array([1-offsets[t_ind % 2]]*3)
            if (len(strings[2]) == len(strings[3])) and color_err_tok_by_corr_rank:
                # If strings are same length, we look for the corresponding
                # answer token, and check its rank
                corr_token = tokens[t_ind - bg_color_inds[3][0] + bg_color_inds[2][0]]
                if tokenize(corr_token)['ranks'][0] < tokenize(tokens[t_ind])['ranks'][0]:
                    bg_c = np.array([offsets[t_ind % 2]]*3)
        else:
            bg_c = bg_colors[t_ind % len(bg_colors)]
        digits.append(color_fn(' '+c+' ', text_color=text_c, bg_color=bg_c))
    # if justify_response
    print('   '*(max_len-len(strings[3])) + ''.join(digits))
        
    # print_line(3, strings[3])


def escape_raw_regex_pattern(s):
    builder = []
    for c in s:
        if c in '.+*?^$()[]{}|\\':
            builder.append('\\')
        builder.append(c)
    return ''.join(builder)


def extract_formatted_number(
    s: str, 
    possible_separators=main.SEPARATOR_NAMES.values()
) -> str:
    escaped_seps = [escape_raw_regex_pattern(sep) for sep in possible_separators]
    pattern = f"([0-9](?:({'|'.join(escaped_seps)})*[0-9])*)"
    matches = re.findall(pattern, s)
    if len(matches) == 1:
        return matches[0][0]
    else:
        return 'None'


def strip_formatting(s: str) -> int:
  if s:
    pattern = r'[0-9]+'
    matches = re.findall(pattern, s)
    if matches:
      return int(''.join(matches))
    else:
      return -1
  else:
    return -1


def longest_streak_of_nonzero(list):
    current_streak = 0
    longest_streak = 0
    for number in list:
        if number != 0:
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        else:
            current_streak = 0
    return longest_streak


def post_process_row(x):
    '''
    This function takes in a row and adds some columns that are useful for post proc
    '''
    retval = dict()

    ### Basic stuff

    format_answer = main.make_format_number(x.answer_number_format, x.get('chunk_size', 3))
    retval['true_answer'] = x.num1 + x.num2
    retval['formatted_true_answer'] = format_answer(retval['true_answer'])
    retval['len1'] = len(str(x.num1))
    retval['len2'] = len(str(x.num2))
    retval['len_true_ans'] = len(str(retval['true_answer']))
    retval['len_match_both'] = ((retval['len1'] == retval['len_true_ans']) and (retval['len2'] == retval['len_true_ans']))
    retval['len_match'] = ((retval['len1'] == retval['len_true_ans']) or (retval['len2'] == retval['len_true_ans']))

    ### Carry stats
    carries = calculate_carries(x.num1, x.num2)
    retval['num_carries'] = np.sum(carries)
    retval['max_carry_chain'] = longest_streak_of_nonzero(carries)

    ### Deal with repeats, set raw = answer part of the response

    # First, convert repeat_style -> repeat_problem_number_format
    repeat = False
    if 'repeat_style' in x:
        if not pd.isna(x.repeat_style):
            assert 'repeat_problem_number_format' not in x
            if x.repeat_style == 'same':
                x.repeat_problem_number_format = x.problem_number_format
            else:
                x.repeat_problem_number_format = x.repeat_style

    raw = None
    retval['repeat_num1'] = -1
    retval['repeat_num2'] = -1

    if 'repeat_problem_number_format' in x:
        if (not pd.isna(x.repeat_problem_number_format)) or (x.repeat_problem_number_format is not None):
            repeated_str = None
            if '=' in x.response:
                repeated_str, raw = x.response.split('=')
            else:
                if '+' in x.response:
                    repeated_str = x.response
                else:
                    raw = x.response

            if repeated_str is not None:
                repeated_pieces = repeated_str.split('+')

                retval['repeat_valid'] = (len(repeated_pieces) == 2)

                if len(repeated_pieces) == 2:
                    retval['repeat_detailed_info'] = [dict(), dict()]
                    for i, p in enumerate(repeated_pieces):
                        trim_p = p.strip()
                        to_use = retval['repeat_detailed_info'][i]
                        to_use['padded'] = (trim_p != p)
                        to_use['extracted_formatted'] = extract_formatted_number(trim_p)
                        to_use['extraneous'] = (trim_p != to_use['extracted_formatted'])
                        to_use['number'] = strip_formatting(to_use['extracted_formatted'])
                        to_use['constructed_formatted'] = main.make_format_number(x.repeat_problem_number_format)(to_use['number'])

                        # Again, we're able to divorce correct copying from correct formatting :)
                        to_use['correct_copy'] = (to_use['number'] == x['num'+str(i+1)])
                        to_use['correct_format'] = (to_use['constructed_formatted'] == to_use['extracted_formatted'])
                    
                    retval['repeat_padded'] = (retval['repeat_detailed_info'][0]['padded'] or retval['repeat_detailed_info'][1]['padded'])
                    retval['repeat_extraneous'] = (retval['repeat_detailed_info'][0]['extraneous'] or retval['repeat_detailed_info'][1]['extraneous'])
                    retval['repeat_num1'] = retval['repeat_detailed_info'][0]['number']
                    retval['repeat_num2'] = retval['repeat_detailed_info'][1]['number']
                    retval['repeat_copy_num1_match'] = retval['repeat_detailed_info'][0]['correct_copy']
                    retval['repeat_copy_num2_match'] = retval['repeat_detailed_info'][1]['correct_copy']
                    retval['repeat_copy_match'] = (retval['repeat_copy_num1_match'] and retval['repeat_copy_num2_match'])
                    retval['repeat_format_match'] = (retval['repeat_detailed_info'][0]['correct_format'] and retval['repeat_detailed_info'][1]['correct_format'])
            else:
                retval['repeat_valid'] = False

    if raw is None:
        raw = x.response

    ### Answer related post proc

    trimmed = raw.strip()
    retval['padded_answer'] = (trimmed != raw)

    # We rename formatted_answer -> extracted_formatted_answer
    retval['extracted_formatted_answer'] = extract_formatted_number(trimmed)
    retval['valid_answer'] = (retval['extracted_formatted_answer'] != 'None')
    retval['answer'] = strip_formatting(retval['extracted_formatted_answer'])

    # When the problem is correct, this should always match formatted_true_answer
    retval['constructed_formatted_answer'] = format_answer(retval['answer'])

    # Extraneous response tells us if there was anything else in the "final output"
    # besides the answer. Note this *doesn't* include repeats/those are already separated
    retval['extraneous_response'] = (trimmed != retval['extracted_formatted_answer'])
    # Proper formatted answer divorces correct answer from correct formatting
    retval['format_match'] = (retval['constructed_formatted_answer'] == retval['extracted_formatted_answer'])
    
    # The usual
    retval['answer_match'] = (retval['answer'] == retval['true_answer'])
    retval['formatted_answer_match'] = (retval['format_match'] and retval['answer_match'])

    return retval


def post_process_df(df):
    return pd.concat([df, df.apply(post_process_row, axis=1, result_type='expand')], axis=1)


def token_size_mask(int, tokenize_dir='l2r'):
    '''Returns vector of same length as int, where entries denote token size of corresponding digit.'''
    # TODO: could be rewritten to actually call the tokenizer. Basically assumes cl_100k_base right now.
    l = len(str(int))
    # Default token size is 3.
    token_sizes = np.array([3]*l)
    # ...except for last token.
    trailing_token_len = l % 3
    if trailing_token_len > 0:
        if tokenize_dir == 'l2r':
            token_sizes[-trailing_token_len:] = trailing_token_len
        else: # r2l
            token_sizes[:trailing_token_len] = trailing_token_len
    return token_sizes


def digit_within_token_mask(token_sizes):
    '''Converts output of token_size_mask to list of digit indices within tokens.'''
    l = len(token_sizes)
    i = 0
    digit_ids = []
    while i < l:
        token_size = token_sizes[i]
        digit_ids.extend(list(range(1, token_size+1)))
        i += token_size
    return digit_ids


def token_id_mask(token_sizes):
    '''Converts output of token_size_mask to list of token indices.'''
    l = len(token_sizes)
    token_id = 1
    digit_id = 0
    token_ids = []
    while digit_id < l:
        token_size = token_sizes[digit_id]
        token_ids.extend([token_id]*token_size)
        token_id += 1
        digit_id += token_size
    return np.array(token_ids)


def analyze_row_error(x, justify_answer='right', tokenize_dir='l2r'):
    '''Takes a row and adds error analysis.'''
    retval = dict()

    if x.answer != -1:  # indicates invalid answer format

        # Is the answer the correct length?
        retval['len_ans'] = len(str(x.answer))
        retval['correct_ans_len'] = x.len_true_ans == retval['len_ans']

        # Off-by-one analysis
        retval['abs_error_mag'] = abs(x.answer - x.true_answer)
        if retval['abs_error_mag'] > 0:
            possibly_just_off_by_one = np.log10(retval['abs_error_mag'])
            retval['only_off_by_one'] = (int(possibly_just_off_by_one) == possibly_just_off_by_one)
            retval['only_off_by_one_digit_num'] = int(retval['len_ans'] - int(possibly_just_off_by_one)) if retval['only_off_by_one'] else -1
        else:
            retval['only_off_by_one'] = False
            retval['only_off_by_one_digit_num'] = -1


        # Errors by digit
        error_sizes = calculate_errors(x.answer, x.true_answer, justify_answer=justify_answer)
        for i, e in enumerate(error_sizes):
            retval[f'digit_{i+1}_error_mag'] = e
            retval[f'digit_{i+1}_error'] = e > 0


        # Errors by digit within token
        token_sizes = token_size_mask(x.answer, tokenize_dir=tokenize_dir)
        digit_ids = digit_within_token_mask(token_sizes)
        token_ids = token_id_mask(token_sizes)
        for e, i, t in zip(error_sizes, digit_ids, token_ids):
            retval[f'digit_{i}_within_{t}token_error_mag'] = e
            retval[f'digit_{i}_within_{t}token_error'] = e > 0
        

        # Errors by token
        # Also adds frequency-related errors and logprob metrics
        cl100k = tiktoken.get_encoding('cl100k_base')
        num_tokens = max(token_ids)
        logprobs = json.loads(x.logprobs_json)
        logprob_ind = 0
        has_made_error = False
        retval['valid_logprobs_metrics'] = True
        retval['answer_logprob'] = 0.0
        retval['increasing_response_logprobs'] = True
        retval['decreasing_response_logprobs'] = True
        prev_logprob = None
        retval['increasing_response_lower_entropy'] = True
        retval['decreasing_response_lower_entropy'] = True
        prev_lower_entropy = None
        retval['increasing_response_top5_entropy'] = True
        retval['decreasing_response_top5_entropy'] = True
        prev_top5_entropy = None
        for i in range(1, num_tokens+1):
            mask = (token_ids == i)
            retval[f'token_{i}_error'] = np.any(np.array(error_sizes)[mask])
            has_made_error = has_made_error or retval[f'token_{i}_error']

            true_answer_str = ''.join([str(x.true_answer)[i] for i in np.where(mask)[0] if i < len(str(x.true_answer))])
            answer_str = ''.join([str(x.answer)[i] for i in np.where(mask)[0]])

            # Only add frequency error if there was an error in this token
            # And the answer given by the model is the correct length (otherwise,
            # not clear how to align)
            if retval['correct_ans_len']:
                true_ans_tok = cl100k.encode(true_answer_str)
                assert len(true_ans_tok) == 1
                true_ans_tok = true_ans_tok[0]

                ans_tok = cl100k.encode(answer_str)
                assert len(ans_tok) == 1
                ans_tok = ans_tok[0]

                # If error in this token, see if the chosen token is more frequent
                # Baseline would be 50%
                if retval[f'token_{i}_error']:
                    retval[f'token_{i}_error_more_frequent'] = (ans_tok < true_ans_tok)

                # See if the model response matches the "best" option assuming ranking by
                # frequency and only allowing off-by-one errors
                # This should be ~1/3 on correct answers
                true_answer_int = int(true_answer_str)
                off_by_one_possible_str = [str((true_answer_int-1) % 1000).zfill(len(true_answer_str)),
                                            str(true_answer_int % 1000).zfill(len(true_answer_str)),
                                            str((true_answer_int+1) % 1000).zfill(len(true_answer_str))]
                off_by_one_possible_tok = []
                for a in off_by_one_possible_str:
                    tok = cl100k.encode(a)
                    assert len(tok) == 1
                    off_by_one_possible_tok.append(tok[0])

                off_by_one_sort_inds = np.argsort(off_by_one_possible_tok)
                retval[f'token_{i}_match_most_frequent_off_by_one_last_digit'] = (off_by_one_possible_str[off_by_one_sort_inds[0]] == answer_str)

                for r in range(len(off_by_one_sort_inds)):
                    if answer_str == off_by_one_possible_str[off_by_one_sort_inds[r]]:
                        retval[f'token_{i}_picked_rank_off_by_one_last_digit'] = r
                        break

                # See if the model responses matches the "best" option assuming ranking by
                # frequence and only allowing errors in the first digit of a token
                # This should be ~1/10 on correct answers
                off_first_digit_possible_str = [(str(a) + true_answer_str[1:]) for a in range(10)]
                off_first_digit_possible_tok = []
                for a in off_first_digit_possible_str:
                    tok = cl100k.encode(a)
                    assert len(tok) == 1
                    off_first_digit_possible_tok.append(tok[0])

                off_first_digit_sort_inds = np.argsort(off_first_digit_possible_tok)
                retval[f'token_{i}_match_most_frequent_off_first_digit'] = (off_first_digit_possible_str[off_first_digit_sort_inds[0]] == answer_str)

                for r in range(len(off_first_digit_sort_inds)):
                    if answer_str == off_first_digit_possible_str[off_first_digit_sort_inds[r]]:
                        retval[f'token_{i}_picked_rank_off_first_digit'] = r
                        break

            # Need to do this because of separators
            while logprob_ind < len(logprobs):
                if logprobs[logprob_ind]['token'] != answer_str:
                    logprob_ind += 1
                else:
                    break
            if logprob_ind < len(logprobs):
                retval[f'token_{i}_logprob'] = logprobs[logprob_ind]['logprob']
                retval['answer_logprob'] += retval[f'token_{i}_logprob']
                if prev_logprob is None:
                    prev_logprob = retval[f'token_{i}_logprob']
                retval['increasing_response_logprobs'] = retval['increasing_response_logprobs'] and (retval[f'token_{i}_logprob'] >= prev_logprob)
                retval['decreasing_response_logprobs'] = retval['decreasing_response_logprobs'] and (retval[f'token_{i}_logprob'] <= prev_logprob)
                prev_logprob = retval[f'token_{i}_logprob']

                retval[f'token_{i}_logprob_top_minus_second'] = logprobs[logprob_ind]['top_5_logprobs'][0]['logprob'] - logprobs[logprob_ind]['top_5_logprobs'][1]['logprob']

                all_tok_logprobs = []
                for cand_ind, candidate in enumerate(logprobs[logprob_ind]['top_5_logprobs']):
                    if candidate['token'] == true_answer_str:
                        retval[f'token_{i}_true_answer_logprob'] = candidate['logprob']
                        retval[f'token_{i}_true_answer_rank'] = cand_ind + 1
                    all_tok_logprobs.append(candidate['logprob'])
                all_tok_logprobs = np.array(all_tok_logprobs)

                total_logprob = logsumexp(all_tok_logprobs)
                retval[f'token_{i}_lower_entropy'] = -(np.sum(np.exp(all_tok_logprobs) * all_tok_logprobs) + (1-np.exp(total_logprob))*all_tok_logprobs[-1])

                if prev_lower_entropy is None:
                    prev_lower_entropy = retval[f'token_{i}_lower_entropy']
                retval['increasing_response_lower_entropy'] = retval['increasing_response_lower_entropy'] and (retval[f'token_{i}_lower_entropy'] >= prev_lower_entropy)
                retval['decreasing_response_lower_entropy'] = retval['decreasing_response_lower_entropy'] and (retval[f'token_{i}_lower_entropy'] <= prev_lower_entropy)
                prev_lower_entropy = retval[f'token_{i}_lower_entropy']

                all_tok_logprobs -= total_logprob
                retval[f'token_{i}_top5_entropy'] = -np.sum(np.exp(all_tok_logprobs)*all_tok_logprobs)

                if prev_top5_entropy is None:
                    prev_top5_entropy = retval[f'token_{i}_top5_entropy']
                retval['increasing_response_top5_entropy'] = retval['increasing_response_top5_entropy'] and (retval[f'token_{i}_top5_entropy'] >= prev_top5_entropy)
                retval['decreasing_response_top5_entropy'] = retval['decreasing_response_top5_entropy'] and (retval[f'token_{i}_top5_entropy'] <= prev_top5_entropy)
                prev_top5_entropy = retval[f'token_{i}_top5_entropy']
            else:
                retval['valid_logprobs_metrics'] = False


        # Errors by carries
        carries_out = calculate_carries(x.num1, x.num2) # len = max(num1, num2)
        carries_in = np.array(carries_out + [0]) # no carries into last digit
        carries_out = np.array([0] + carries_out) # pad to same length as carry_in
        l = min(retval['len_ans'], len(carries_in))
        clipped_error_sizes = np.array(error_sizes[-l:])
        retval['num_carry_in_errors'] = np.sum(clipped_error_sizes[carries_in[-l:] == 1] > 0)
        retval['num_carry_out_errors'] = np.sum(clipped_error_sizes[carries_out[-l:] == 1] > 0)
        retval['carry_in_error'] = retval['num_carry_in_errors'] > 0
        retval['carry_out_error'] = retval['num_carry_out_errors'] > 0

        # Check for dropped input digit error
        # TODO: log position of dropped input digit error
        retval['dropped_digit_error'] = False
        if x.answer != x.true_answer:
            # loop over num1 digits
            for i in range(x.len1):
                s = str(x.num1)
                new_num1 = int(s[:i] + s[i+1:])
                if x.answer == new_num1 + x.num2:
                    retval['dropped_digit_error'] = True
            # loop over num2 digits
            for i in range(x.len2):
                s = str(x.num2)
                new_num2 = int(s[:i] + s[i+1:])
                if x.answer == x.num1 + new_num2:
                    retval['dropped_digit_error'] = True

        # Check for left-alignment addition (i.e. implicit zero-padding of shorter number)
        l1 = len(str(x.num1))
        l2 = len(str(x.num2))
        if l1 == l2:
            retval['left_alignment_error'] = False
        else:
            if l1 > l2:
                # pad shorter num2 and add to num1
                padded_num2 = x.num2 * 10**(l1-l2)
                padded_ans = x.num1 + padded_num2
            else:  # l2 > l1
                # pad shorter num1 and add to num2
                padded_num1 = x.num1 * 10**(l2-l1)
                padded_ans = x.num2 + padded_num1
            retval['left_alignment_wrong_answer'] = padded_ans
            retval['left_alignment_error'] = x.answer == padded_ans


        # Check for chunk by chunk addition
        # ...except there are no chunks for nosep, so skip those
        if 'nosep' not in x.problem_number_format:
            format_problem_number = main.make_format_number(x.problem_number_format, x.get('chunk_size', 3))
            num1_chunks = re.findall(r'\d+', format_problem_number(x.num1))
            num2_chunks = re.findall(r'\d+', format_problem_number(x.num2))
            l1 = len(num1_chunks)
            l2 = len(num2_chunks)
            if l1 == l2:
                # zip chunks, convert string elements to numbers and add them, convert back to string and join, then back to number
                chunked_ans = int(''.join([str(np.sum([int(num) for num in pair])) for pair in zip(num1_chunks, num2_chunks)]))
                chunked_error = x.answer == chunked_ans
                retval['left_chunked_error'] = chunked_error
                retval['right_chunked_error'] = chunked_error
            elif l1 < l2:
                # pad shorter num1_chunks from left to check for right-aligned chunked error
                left_padded_num1_chunks = ['0']*(l2-l1) + num1_chunks
                right_chunked_ans = int(''.join([str(np.sum([int(num) for num in pair])) for pair in zip(left_padded_num1_chunks, num2_chunks)]))
                retval['right_chunked_error'] = x.answer == right_chunked_ans
                # pad shorter num1_chunks from right to check for left-aligned chunked error
                right_padded_num1_chunks =  num1_chunks + ['0']*(l2-l1)
                left_chunked_ans = int(''.join([str(np.sum([int(num) for num in pair])) for pair in zip(right_padded_num1_chunks, num2_chunks)]))
                retval['left_chunked_error'] = x.answer == left_chunked_ans
            else: # l1 > l2
                # pad shorter num2_chunks from left to check for right-aligned chunked error
                left_padded_num2_chunks = ['0']*(l1-l2) + num2_chunks
                right_chunked_ans = int(''.join([str(np.sum([int(num) for num in pair])) for pair in zip(num1_chunks, left_padded_num2_chunks)]))
                retval['right_chunked_error'] = x.answer == right_chunked_ans
                # pad shorter num2_chunks from right to check for left-aligned chunked error
                right_padded_num2_chunks =  num2_chunks + ['0']*(l1-l2)
                left_chunked_ans = int(''.join([str(np.sum([int(num) for num in pair])) for pair in zip(num1_chunks, right_padded_num2_chunks)]))
                retval['left_chunked_error'] = x.answer == left_chunked_ans

    return retval


def add_error_analysis(df, justify_answer='right', tokenize_dir='l2r'):
    return pd.concat([df, df.apply(partial(analyze_row_error, justify_answer=justify_answer, tokenize_dir=tokenize_dir), axis=1, result_type='expand')], axis=1)


def bootstrap(np_arr, sample_size=80, num_samples=10):
    perms = np.argsort(np.random.random((num_samples, np_arr.shape[0])), axis=1)
    vals = np.mean(np_arr[perms[:, :sample_size]], axis=1)
    return np.mean(vals), np.std(vals)


if __name__ == '__main__':
    # Note that the below doesn't work on macOS terminal, but seems to work
    # in jupyter notebooks (and thus I assume, VSCode)
    print(color_ansi('hi', np.array([31, 119, 180]), np.array([255, 127,  14])))

    df = pd.read_csv('results/20230524_gpt_comma_effects_repeat_style_triplets.csv', index_col=0)
    df2 = post_process_df(df)
    pdb.set_trace()
