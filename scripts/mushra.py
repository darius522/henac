import json
from collections import defaultdict
import math

# Load JSON from file
with open('/N/slate/daripete/jstsp-dac/scripts/mushra_scores.json', 'r') as f:
    data = json.load(f)

# Containers
system_scores = defaultdict(list)
discarded_keys = []
kept_keys = []

for participant_key, participant_json in data.items():
    participant_data = json.loads(participant_json)
    trials = participant_data.get('trials', [])

    # Count how many times 'reference' was rated < 90
    reference_low_count = 0
    for trial in trials:
        for response in trial.get('responses', []):
            if response['stimulus'] == 'reference' and float(response['score']) < 90:
                reference_low_count += 1

    if reference_low_count >= 2:
        discarded_keys.append(participant_data['participant']['response'][0]+f', missed {reference_low_count} times')
        continue  # Skip this participant's data

    kept_keys.append(participant_data['participant']['response'][0])

    # Accumulate scores
    for trial in trials:
        for response in trial.get('responses', []):
            stimulus = response['stimulus']
            score = float(response['score'])
            system_scores[stimulus].append(score)

# Compute mean and standard deviation per system
system_stats = {}
for system, scores in system_scores.items():
    n = len(scores)
    mean = sum(scores) / n
    variance = sum((x - mean) ** 2 for x in scores) / n
    std_dev = math.sqrt(variance)
    system_stats[system] = (mean, std_dev)

# Compute mean and standard deviation per system
system_stats = {}
for system, scores in system_scores.items():
    n = len(scores)
    mean = sum(scores) / n
    variance = sum((x - mean) ** 2 for x in scores) / n
    std_dev = math.sqrt(variance)
    system_stats[system] = (mean, std_dev)

# Print results
print("System-wise average and standard deviation:")
for system, (avg, std) in sorted(system_stats.items()):
    print(f"{system}: mean = {avg:.2f}, std = {std:.2f}")

# Print participant filtering report
print("\nKept participant keys (reference rated <90 less than twice):")
for key in kept_keys:
    print(f"- {key}")

print("\nDiscarded participant keys (reference rated <90 at least twice):")
for key in discarded_keys:
    print(f"- {key}")