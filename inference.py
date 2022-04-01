import torch 
from puppersim.encoder import TransformerEncoder
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

trajectory_type = "heavy_alternating"
pretrained_encoder = "encoder.pt"
window_size = 20

csv_file = trajectory_type + ".csv"
png_file = trajectory_type + "_" + pretrained_encoder[:-3] + "_window_" + str(window_size)  + ".png"


f = open(csv_file, "r")
all_lines = f.readlines()
f.close()
#print(all_lines[0][1:-2].split(", "))
states = [np.array([float(val) for val in line[1:-2].split(", ")]) for line in all_lines]

print(f"Number of states from trajectory {trajectory_type}: {len(states)}")

model = TransformerEncoder(len(states[0]))
model.load_state_dict(torch.load(pretrained_encoder))

predictions = []
for idx in tqdm(range(len(states) - window_size)):
    pred = model(states[idx:idx+window_size])
    pred = pred.squeeze(0).detach().numpy()[0]
    predictions.append(pred)

plt.plot(np.linspace(0, len(predictions), len(predictions)), predictions)
plt.savefig(png_file)



