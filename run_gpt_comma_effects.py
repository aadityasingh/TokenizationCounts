# Used to create:
# 20230407_gpt35_comma_effects.csv
# 20230502_gpt4_comma_effects.csv
# This experiment was run before the API update to the
# client formalism

from main import *

PROBLEM_KEY = 0
SHOTS_KEY = 1

api_call_fn = make_safe_api_call()
# api_call_fn = dummy_api_call_fn

sample_problem_fn = make_digit_controlled_sample_problem(jax.random.PRNGKey(PROBLEM_KEY), 7, 10)

sample_shots_fn = make_digit_same_sample_shots(jax.random.PRNGKey(SHOTS_KEY))

# sweeps
models = ['gpt-3.5-turbo-0301']
formats = ['nosep', 'comma_r2l', 'comma_l2r', 'space_r2l', 'space_l2r']
num_shots = [1, 2, 4, 8]
system_prompts = [
	'You are a helpful assistant.',
	'You are MathGPT, an expert at solving math problems. When given a math problem, you respond only by repeating the problem statement and appending the answer. You do not say any other words.',
	]
repeat_styles = [None]
exp_name = 'gpt_comma_effects'

# fixed
problem_range = [0, 90]
flip_def = [True, False, False, False]

dfs = []
run_index = 0
for model in models:

	for system_prompt in system_prompts:

		for f in formats:

			if f=='nosep':
				spaces_to_sweep = [0, 1, 2]
			else:
				spaces_to_sweep = [0]

			for spaces in spaces_to_sweep:

				for k in num_shots:

					format_problem_number_fn = make_format_number(f)

					format_answer_number_fn = make_format_number(f)

					format_problem_fn = make_format_addition_problem(num_spaces=spaces, num_trailing_spaces=spaces)

					for r in repeat_styles:

						if r is None:
							# don't repeat
							format_problem_number_repeat_fn = None
							format_problem_repeat_fn = None
						elif r == 'same':
							# repeat in same style as query
							format_problem_number_repeat_fn = format_problem_number_fn
							format_problem_repeat_fn = format_problem_fn
						else:
							# repeat in a fixed style (e.g. comma_r2l)
							format_problem_number_repeat_fn = make_format_number(r)
							format_problem_repeat_fn = make_format_addition_problem()
							# in this case, we need to overwrite format_answer_number_fn
							format_answer_number_fn = make_format_number(r)

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
						df['problem_number_format'] = f
						df['answer_number_format'] = f
						df['repeat_style'] = r
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
