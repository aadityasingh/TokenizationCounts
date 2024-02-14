# Used to create:
# 20240125_gpt35_0301_answer_length_effects.csv
# 20240125_gpt4_turbo_answer_length_effects_new_models.csv
# 20240125_gpt_answer_length_effects_new_models.csv
# 20240130_gpt35_1106_turbo_answer_length_effects_new_models.csv

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

lengths = [(7,7,7), 
						(7,7,8), 
						(8,7,8), 
						(7,8,8),
						(8,7,9),
						(7,8,9),
						(8,8,8),
						(8,8,9),
						(9,7,9),
						(7,9,9),
						(9,8,9),
						(8,9,9),
						(9,9,9)]

sample_problem_fn = make_all_digit_length_controlled_sample_problem(jax.random.PRNGKey(PROBLEM_KEY), lengths)

sample_shots_fn = make_all_digit_length_same_sample_shots(jax.random.PRNGKey(SHOTS_KEY))

# sweeps
# Configure model and exp_name as desired
# We split these up across runs to parallelize
models = ['gpt-3.5-turbo-1106']#'gpt-3.5-turbo-0301','gpt-3.5-turbo-0613', 'gpt-4-0613']
formats = ['nosep', 'comma_r2l']
num_shots = [8]
system_prompts = ['You are a helpful assistant.']
repeat_styles = [None]
exp_name = 'gpt35_1106_answer_length_effects_new_models'

# 13 * 2 * 2 * 100 = 6800

# fixed
problem_range = [0, len(lengths)*100]

# Note we accidentally ran with flipped problem and flipped shots
# WHICH IS FINE for this experiment
# since it basically just reorders our list of lengths
flip_def = [False, False, False, True]

dfs = []
run_index = 0
for model in models:

	for system_prompt in system_prompts:

		for f in formats:

			for k in num_shots:

				format_problem_number_fn = make_format_number(f)

				format_answer_number_fn = make_format_number(f)

				format_problem_fn = make_format_addition_problem()

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
					df['num_spaces'] = 0
					df['num_trailing_spaces'] = 0
					df['num_shots'] = k
					df['problem_rng_key'] = PROBLEM_KEY
					df['shots_rng_key'] = SHOTS_KEY

					dfs.append(df)
					run_index += 1


df = pd.concat(dfs, ignore_index=True)
df.to_csv('results/{}_{}.csv'.format(datetime.now().strftime("%Y%m%d"), exp_name))
costs = dict()
# Reminder that response_repeat refers to temperature sampling,
# nothing to do with repeating the problem
for k in PRICES[model]:
  costs[k] = df[df['response_repeat'] ==0][k].sum()*PRICES[model][k]
print("Prompt cost: ${:.4f}, total cost: ${:.4f}".format(costs['prompt_tokens'], costs['prompt_tokens']+costs['completion_tokens']))
