# Largely written by GPT-4, some modifications posthoc by me.
# Prompt used:
#
# I want to create a tokenization game in Python. The code should load in the cl100k_base tokenizer from tiktoken. Through a command-line interface, the user will first choose a variable, `num_choices`. Once this is selected, the game will begin.
#
# The game will be endless, with an infinite number of rounds, until the user provides an exit signal. On each round of the game, a random sample of `num_choices` distinct tokens will be chosen. These will be presented to the user surrounded by quotation marks as options. For example, for `num_choices = 3`:
#
# '''
# a) "banana"
# b) "  "
# c) "543"
# most common? _
# '''
#
# where the _ indicates awaiting keyboard input from the user. The user will then guess one of the letters (in this case, a,b,c), in which case the program will evaluate if the human correctly guessed which of the tokens is the most common (this corresponds to having the lowest value in the `_mergeable ranks` dictionary, which maps tokens to indices). The program will then output "Correct!" or "Wrong!", followed by the tokens with their actual indices, followed by the user's average accuracy so far, expressed as a percentage with two decimal points. Then, the program will start the next round, and so on.
#
# Please write code to do this.

import random
import tiktoken

# Load the cl100k_base tokenizer
tokenizer = tiktoken.get_encoding('cl100k_base')

# Get the _mergeable_ranks dictionary
mergeable_ranks = tokenizer._mergeable_ranks

# Function to get a random sample of tokens
def get_random_tokens(num_choices):
    return random.sample(list(mergeable_ranks.keys()), num_choices)

# Function to get the most common token from a list of tokens
def get_most_common_token(tokens):
    return min(tokens, key=lambda token: mergeable_ranks[token])

# Main game loop
def play_game(num_choices):
    correct_count = 0
    total_count = 0

    while True:
        tokens = get_random_tokens(num_choices)
        most_common_token = get_most_common_token(tokens)

        print("Choose the most common token:")
        for i, token in enumerate(tokens):
            print(f"{chr(ord('a') + i)}) {repr(token.decode('utf-8'))}")

        user_input = input("most common? ")

        if user_input.lower() == "exit":
            break

        user_choice = tokens[ord(user_input.lower()) - ord('a')]
        total_count += 1

        if user_choice == most_common_token:
            print("Correct!")
            correct_count += 1
        else:
            print("Wrong!")

        print("Tokens with their actual indices:")
        for token in tokens:
            print(f"{repr(token.decode('utf-8'))}: {mergeable_ranks[token]}")

        accuracy = (correct_count / total_count) * 100
        print(f"Your average accuracy so far: {accuracy:.2f}%")
        print('-'*20)

if __name__ == "__main__":
    num_choices = int(input("Enter the number of choices: "))
    play_game(num_choices)