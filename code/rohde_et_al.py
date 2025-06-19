import json
from util import fill_mask

with open("stims.json") as f:
    data = json.load(f)


def run_all():
    for trial in data:
        action = trial["action"]
        instruments = [trial["predictable-instrument"], trial["unpredictable-instrument"]]

        surprising = f"Alice is a surprising person who never does anything the way you'd expect. For example, yesterday I saw her {action} with a [MASK]"
        boring = f"Alice is a boring person who always does things exactly the way you'd expect. For example, yesterday I saw her {action} with a [MASK]"
        
        surprising_probs = fill_mask(surprising, targets = instruments)
        boring_probs = fill_mask(boring, targets = instruments)


run_all()

