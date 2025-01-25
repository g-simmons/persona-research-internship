import json

# Load the data from the JSONL file
data = []
with open('toy_data.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Threshold for local minima
threshold = 10.0

# Find data points close to local minima
local_minima = []
for entry in data:
    if entry['value'] < threshold:
        local_minima.append(entry)


# Save the local minima data in a new JSONL file
with open('local_minima.jsonl', 'w') as output_file:
    for entry in local_minima:
        json.dump(entry, output_file)
        output_file.write('\n')
