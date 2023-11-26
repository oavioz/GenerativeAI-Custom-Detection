import argparse
import json
import os
import sys

import numpy
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
parser.add_argument("-d", "--dict", type=path_exists_extension("dict"), help="Normalisation dict", required=True)
args = parser.parse_args()

with open(args.dict) as file:
    norm_dict = json.load(file)

if 'version' not in norm_dict.keys() or norm_dict['version'] != 2:
    print("The normalisation dictionary version is unsupported. "
          "You can create a new one using make_model.py with --no-model")
    sys.exit(8)

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
        inputs.append(float(input("[0..100]% ")) / 100)
    elif norm['type'] == '4':
        l, h = norm['data']
        x = float(input(f"[{l} - {h}] "))
        inputs.append((x - l) / (h - l))
    elif norm['type'] == '5':
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
model.eval()

pred = model(torch.tensor(inputs, dtype=torch.float64))
output_norm = norm_dict['columns'][norm_dict['output']]


def denorm_output(output):
    tensor = output_norm['data'][0] + output * (output_norm['data'][1] - output_norm['data'][0])
    return tensor.squeeze().detach().numpy()


print(f"Predicted output: {denorm_output(pred)} ({norm_dict['output']})\n")

for i, c in enumerate(norm_dict['columns'].keys()):
    if norm_dict['columns'][c]['type']:
        print(f"{i} - {c}")
print()

while (i := int(input("Enter column to get recommendations or '-1' to quit"))) != -1:
    c = list(norm_dict['columns'].keys())[i]
    norm = norm_dict['columns'][c]

    if norm['type'] == '1':
        test = [
            [x if (not si[c] <= k <= (si[c] + len(norm['data']))) else (1 if (k - si[c]) == j else 0) for k, x in enumerate(inputs)]
            for j in range(len(norm['data']))]
        pred = model(torch.tensor(test, dtype=torch.float64))
        plt.bar(norm['data'], denorm_output(pred))

    if norm['type'] == '2':
        val = dates[c]


        def normalise(dt):
            return (dt - datestart).days / (dateend - datestart).days


        variance = [val - pd.Timedelta(days=90), val - pd.Timedelta(days=30), val - pd.Timedelta(weeks=1),
                    val,
                    val + pd.Timedelta(weeks=1), val + pd.Timedelta(days=30), val + pd.Timedelta(days=90)]

        test = [
            [x if (not k == si[c]) else (normalise(variance[j])) for k, x in enumerate(inputs)]
            for j in range(len(variance))]

        pred = model(torch.tensor(test, dtype=torch.float64))
        plt.plot([dt.strftime("%d/%m/%Y") for dt in variance], denorm_output(pred))

    if norm['type'] == '3':
        variance = [i / 20 for i in range(21)]

        test = [
            [x if (not k == si[c]) else variance[j] for k, x in enumerate(inputs)]
            for j in numpy.arange(len(variance))]

        pred = model(torch.tensor(test, dtype=torch.float64))
        plt.plot(variance, denorm_output(pred))

    if norm['type'] == '4':
        l, h = norm['data']
        variance = numpy.arange(l, h, (h - l) / 100)

        test = [
            [x if (not k == si[c]) else ((variance[j] - l) / (h - l)) for k, x in enumerate(inputs)]
            for j in numpy.arange(len(variance))]

        pred = model(torch.tensor(test, dtype=torch.float64))
        plt.plot(variance, denorm_output(pred))
    if norm['type'] == '5':
        test = [
            [x if (not k == si[c]) else j for k, x in enumerate(inputs)]
            for j in range(2)]

        pred = model(torch.tensor(test, dtype=torch.float64))
        plt.bar(["False", "True"], denorm_output(pred))

    plt.xlabel(c)
    plt.ylabel("Yield")
    plt.show()
