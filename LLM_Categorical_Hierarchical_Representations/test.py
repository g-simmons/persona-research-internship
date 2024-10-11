#!/usr/bin/env python3

import ontology_class
import os

test = ontology_class.Onto("owl/ado.owl")

all_noun_synsets = test.all_synsets()

print([i.name() for i in all_noun_synsets])
print(len(all_noun_synsets))

for filename in os.listdir("owl"):
    try:
        test = ontology_class.Onto(f"owl/{filename}")
        all_noun_synsets = test.all_synsets()
        if len(all_noun_synsets) > 1000:
            with open(f"textfiles/{filename.split(".")[0]}.txt", "w") as f:
                for i in all_noun_synsets:
                    if i.name() != None:
                        f.write(i.name() + "\n")
    except Exception as e:
        print(Exception)