import argparse
import json
import os

import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt


def path_exists_extension(ext):
    def valid(path: str):
        if os.path.exists(path) and path.endswith(f".{ext}"):
            return path
        raise argparse.ArgumentTypeError(f"{path} doesn't exist or not {ext}")

    return valid


parser = argparse.ArgumentParser()
parser.add_argument("model", type=path_exists_extension("pth"), help="File containing the model weights")
parser.add_argument("-d", "--dict", type=path_exists_extension("dict"), help="Line of data to test", required=True)
args = parser.parse_args()

with open(args.dict) as file:
    norm_dict = json.load(file)

datestart, dateend = map(lambda x: pd.to_datetime(x, unit="s"), norm_dict["time"])

inputs = []
dummies = []
si = {}
dates = {}

for c, norm in norm_dict['columns'].items():
    if c == norm_dict['output']:
        continue
    print(f"Enter value for {c}:")
    if norm['type'] == '1':
        for i, key in enumerate(norm['data']):
            print(f"{i}\t{key}")
        print(f"{len(norm['data'])}\tNone of the following")
        sel = int(input())
        for i in range(len(norm['data'])):
            dummies.append(1 if (i == sel) else 0)
    elif norm['type'] == '2':
        date = pd.Timestamp(input("[yyyy-mm-dd] "))
        dates[c] = date
        inputs.append((date - datestart).days / (dateend - datestart).days)
    elif norm['type'] == '3':
        l, h = norm['data']
        x = float(input(f"[{l}..{h}] "))
        inputs.append((x - l) / (h - l))
    elif norm['type'] == '4':
        inputs.append(1.0 if bool(input("[True/False] ")) else 0)
    if norm['type'] != '1':
        si[c] = len(si)

last = len(si)
last_len = 0
for c, norm in norm_dict['columns'].items():
    if norm['type'] == '1':
        si[c] = last + last_len
        last = si[c]
        last_len = len(norm['data'])

inputs += dummies
inputs = list(map(float, inputs))

# print(f"Enter output value: ({norm_dict['output']}")
# output = float(input(
#     f"[{norm_dict['columns'][norm_dict['output']]['data'][0]}"
#     f"..{norm_dict['columns'][norm_dict['output']]['data'][1]}] "))

input_width = len(inputs)


class MLPRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_width, 80, dtype=float),
            nn.ReLU(),
            nn.Linear(80, 1, dtype=float)
        )

    def forward(self, x):
        return self.linear_stack(x)


model = MLPRegressor()
model.load_state_dict(torch.load(args.model))

pred = model(torch.tensor(inputs))
output_norm = norm_dict['columns'][norm_dict['output']]
pred_value = output_norm[0] + pred * (output_norm[1] - output_norm[0])
print(f"Predicted output: {pred_value} ({norm_dict['output']})\n")

for i, c in norm_dict['columns'].enumerate:
    if norm_dict['columns'][c]['type']:
        print(f"{i} - {c}")
print()

while (i := int(input("Enter column to get recommendations or '-1' to quit"))) != -1:
    c = list(norm_dict['columns'].keys())[i]
    norm = norm_dict['columns'][c]

    if norm['type'] == 1:
        test = [
            [x if (not si[c] <= k <= si[c] + len(norm['data'])) else (1 if k == j else 0) for k, x in enumerate(inputs)]
            for j in range(len(norm['data']))]
        pred = model(torch.tensor(inputs))
        plt.bar(height=pred, tick_label=norm['data'])

    if norm['type'] == 2:
        val = dates[c]


        def normalise(dt):
            return (dt - datestart).days / (dateend - datestart).days


        variance = [val - pd.Timedelta(months=3), val - pd.Timedelta(months=1), val - pd.Timedelta(weeks=1),
                    val,
                    val + pd.Timedelta(weeks=1), val + pd.Timedelta(months=1), val + pd.Timedelta(months=3)]

        test = [
            [x if (not k == si[c]) else (normalise(variance[j])) for k, x in enumerate(inputs)]
            for j in range(len(variance))]

        pred = model(torch.tensor(inputs))
        plt.plot([dt.strftime("%d/%m/%Y") for dt in variance], pred)

    if norm['type'] == 3:
        l, h = norm['data']
        variance = range(l, h, (h - l) / 100)

        test = [
            [x if (not k == si[c]) else ((variance[j] - l) / (h - l)) for k, x in enumerate(inputs)]
            for j in range(len(variance))]

        pred = model(torch.tensor(inputs))
        plt.plot(variance, pred)
    if norm['type'] == 4:
        test = [
            [x if (not k == si[c]) else j for k, x in enumerate(inputs)]
            for j in range(2)]

        pred = model(torch.tensor(inputs))
        plt.bar(heights=pred, tick_label=["False", "True"])

    plt.xlabel(c)
    plt.ylabel("Yield")
    plt.show()
