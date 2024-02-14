# Used to create:
# 20240125_run_gpt_comma_effects_repeat_style_triplets_new_models.csv
# 20240125_run_gpt1106_comma_effects_repeat_style_triplets_new_models.csv

from main import *

# openai should get imported from main.py
client = openai.OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

PROBLEM_KEY = 0
SHOTS_KEY = 1

api_call_fn = make_safe_openai_api_call(client)
# api_call_fn = dummy_api_call_fn

sample_problem_fn = make_digit_controlled_sample_problem(jax.random.PRNGKey(PROBLEM_KEY), 7, 10)

sample_shots_fn = make_digit_same_sample_shots(jax.random.PRNGKey(SHOTS_KEY))

# sweeps
models = ['gpt-3.5-turbo-1106', 'gpt-4-1106-preview']#['gpt-3.5-turbo-0613', 'gpt-4-0613']
num_shots = [1, 2, 4, 8]
system_prompts = ['You are a helpful assistant.']
number_format_triplets = [ # query, repeat, answer
	('nosep', 'comma_r2l', 'nosep'), # comma_r2l bottleneck
	('comma_r2l', 'nosep', 'comma_r2l'), # nosep bottleneck
	('nosep', 'comma_r2l', 'comma_r2l'), # translate nosep to comma_r2l
	('comma_r2l', 'nosep', 'nosep'), # translate comma_r2l to nosep
	('comma_r2l', 'comma_r2l', 'comma_r2l'), # comma_r2l w repeat
	('nosep', 'nosep', 'nosep'), # nosep w repeat
    ('comma_r2l', None, 'comma_r2l'), # comma_r2l w/o repeat
	('nosep', None, 'nosep'), # nosep w/o repeat
    ('nosep', None, 'comma_r2l'),
    ('comma_r2l', None, 'nosep')
    ]
exp_name = 'run_gpt1106_comma_effects_repeat_style_triplets_new_models'

# 2 * 4 * 10 * 90 = 7200

# fixed
problem_range = [0, 90]
flip_def = [True, False, False, False]
spaces = 0

dfs = []
run_index = 0
for model in models:

    for system_prompt in system_prompts:

        for qf, rf, af in number_format_triplets:

            for k in num_shots:

                format_problem_number_fn = make_format_number(qf)
                format_problem_fn = make_format_addition_problem(num_spaces=spaces, num_trailing_spaces=spaces)

                if rf is not None:
                    format_problem_number_repeat_fn = make_format_number(rf)
                    format_problem_repeat_fn = format_problem_fn 
                else:
                    format_problem_number_repeat_fn = None
                    format_problem_repeat_fn = None

                format_answer_number_fn = make_format_number(af) 

                df = run_addition_experiment(
                    f'{exp_name}_{run_index}',
                    model=model,
                    api_call_fn=api_call_fn,
                    sample_problem_fn=sample_problem_fn,
                    format_problem_number_fn=format_problem_number_fn,
                    format_answer_number_fn=format_answer_number_fn,
                    format_problem_fn=format_problem_fn,
                    format_problem_number_repeat_fn=format_problem_number_repeat_fn,
                    format_problem_repeat_fn=format_problem_repeat_fn,
                    sample_shots_for_problem_fn=sample_shots_fn,
                    problem_range=problem_range,
                    flip_def=flip_def,
                    num_shots=k,
                    system_prompt=system_prompt,
                    prompt=None,
                    ckpt_every=1e9,
                    temperature=0, 
                    # max_tokens=10 if r is None else 20,
                    max_tokens=20,
                    num_model_outputs=1, 
                    verbose=True,
                    pause_after_cost_estimate=False)

                df['model'] = model
                df['system_prompt'] = system_prompt
                df['problem_number_format'] = qf
                df['repeat_problem_number_format'] = rf
                df['answer_number_format'] = af
                df['num_spaces'] = spaces
                df['num_trailing_spaces'] = spaces
                df['num_shots'] = k
                df['problem_rng_key'] = PROBLEM_KEY
                df['shots_rng_key'] = SHOTS_KEY

                dfs.append(df)
                run_index += 1


df = pd.concat(dfs, ignore_index=True)
df.to_csv('results/{}_{}.csv'.format(datetime.now().strftime("%Y%m%d"), exp_name))
costs = dict()
for k in PRICES[model]:
  costs[k] = df[df['response_repeat'] ==0][k].sum()*PRICES[model][k]
print("Prompt cost: ${:.4f}, total cost: ${:.4f}".format(costs['prompt_tokens'], costs['prompt_tokens']+costs['completion_tokens']))
