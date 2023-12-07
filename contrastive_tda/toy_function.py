import numpy as np
import json

# Defining a function
def formula(x,y):
    function = (((x**2)+y-11)**2+(x+(y**2)-7)**2)
    return function

# Generates values based on given x,y 
def generate_data(func, num_samples,seed):
    np.random.seed(seed)
    temp = []

    for i in range(num_samples):
        x, y = np.random.uniform(-5, 5), np.random.uniform(-5, 5)
        value = func(x, y)
        temp.append({'x': x, 'y': y, 'value': value})

    return temp

data = generate_data(formula, 1000, 1)

# Save the generated data in a JSONL file
with open('toy_data.jsonl', 'w') as f:
    for line in data:
        json.dump(line, f)
        f.write('\n')
