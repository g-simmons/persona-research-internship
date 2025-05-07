#!/usr/bin/env python3

import os
import torch

# import ontology_scores


# steps = [f"step{i}" for i in range(1000, 145000, 2000)]
# param_model = "1.4B"

# for step in steps:
#     mats = ontology_scores.get_mats(params = param_model, step = step, multi=True, model_name="pythia")
#     with open("test.txt", "a") as f:
#         f.write("\n\n\n")
#         f.write(str(step))
#         f.write("\n")
#     for mat in mats:
#         with open("test.txt", "a") as f:
#             f.write(str(mat.shape))
#             f.write("\n")

bs = []
wtvr = "/mnt/bigstorage/raymond/heatmaps-pythia/2.8B"
for path in os.listdir(wtvr):
    heatmap = torch.load(wtvr + "/" + path)
    bs.append(f"{path}, {heatmap.shape}")

bs.sort(key=lambda x: int(x.split("-")[1].split("p")[1]))

for i in range(len(bs)//4):
    print(sorted(bs[i*4:i*4+4]))