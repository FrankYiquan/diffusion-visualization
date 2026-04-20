import random

def load_random_prompts(num_prompts, seed):
    random.seed(seed)

    # Hardcode a small subset of prompts (safe + simple)
    prompts = [
        "A black apple and a green backpack.",
        "A small blue book sitting on a large red book.",
        "A couple of glasses are sitting on a table.",
        "A side view of an owl sitting in a field.",
        "A cross-section view of a brain.",
        "A long curved fruit which grows in clusters and has soft pulpy flesh and yellow skin when ripe.",
        "A domesticated carnivorous mammal that typically has a long snout, an acute sense of smell, nonretractable claws, and a barking, howling, or whining voice.",
        "A machine resembling a human being and able to replicate certain human movements and functions automatically.",
        "A large thick-skinned semiaquatic African mammal, with massive jaws and large tusks.",
        "Paying for a quarter-sized pizza with a pizza-sized quarter."
    ]

    return random.sample(prompts, num_prompts)