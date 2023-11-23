import argparse
import json
import os
from datetime import datetime

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, random_split, DataLoader
from torcheval.metrics import R2Score


def float_in_range(a: float, b: float):
    def f(s: str):
        x = float(s)
        if a <= x <= b:
            return x
        raise argparse.ArgumentTypeError(f'{x} not in range [{a}, {b}]')

    return f


def existing_csv_file(path: str):
    if os.path.exists(path) and path.endswith(".csv"):
        return path
    raise argparse.ArgumentTypeError(f"{path} doesn't exist or not csv")


parser = argparse.ArgumentParser()
parser.add_argument("table", type=existing_csv_file, help="CSV file containing the data")
gr_training = parser.add_mutually_exclusive_group()
gr_training.add_argument("-i", "--epochs", type=int, help="Train the model for a final number of epochs")
gr_training.add_argument("--r2", type=float_in_range(0, 1), help="Train the model until r2 more than argument")
gr_training.add_argument("--mse", type=float_in_range(0, 1), help="Train the model until MSE lower than argument")
gr_time = parser.add_argument_group()
gr_time.add_argument("--epochstart", type=str, help="First possible date (in data) [yyyy-mm-dd]",
                     default="2000-01-01"),
gr_time.add_argument("--epochend", type=str, help="Last possible date (in data) [yyyy-mm-dd]",
                     default="2040-01-01"),
parser.add_argument("--dict", help="Normalisation table filename", default="normalisation.dict")
parser.add_argument("-r", "--ratio", type=float_in_range(0, 1), help="Part of data to use for testing/validation",
                    default=0.25)
args = parser.parse_args()
args.epochstart = pd.Timestamp(args.epochstart)
args.epochend = pd.Timestamp(args.epochend)

data = pd.read_csv(args.table, encoding="latin-1")

norm_dict = {}

print("""Normalisation types:
1 - Categories
2 - Timestamp
3 - Scale
4 - Boolean (True/False)
5 - Drop column""")

for c in data.columns:
    print(f"Select normalisation for column {c}:")
    norm = input()

    while not ('1' <= norm <= '5'):
        print("Invalid input")
        norm = input()


    def serialise(norm, column):
        if norm == '1':
            return list(column.dropna().unique())
        if norm == '2':
            return None
        if norm == '4':
            return None
        if norm == '3':
            amplitude = column.max() - column.min()
            return float(column.min() - amplitude / 10), float(column.max() + amplitude * 10)


    if norm != '5':
        norm_dict[c] = {'type': norm, 'data': serialise(norm, data[c])}

    if norm == '1':
        data = pd.get_dummies(data, columns=[c], dtype=float)
    elif norm == '2':
        data[c] = pd.to_datetime(data[c], format='mixed')
        data[c] = (data[c] - pd.Timestamp(args.epochstart)).dt.days / (args.epochend - args.epochstart).days
    elif norm == '3':
        amplitude = data[c].max() - data[c].min()
        data[c] = (data[c] - (data[c].min() - amplitude / 10)) / (amplitude * 1.2)
    elif norm == '4':
        data[c] = data[c].astype(float)
    elif norm == '5':
        data = data.drop(columns=[c])

y_column = input("Enter output column name: ")
while y_column not in data.columns:
    print("Doesn't exist")
    y_column = input("Enter output column name: ")

norm_dict = {'time': (args.epochstart.timestamp(), args.epochend.timestamp()), 'columns': norm_dict, 'output': y_column}
with open(args.dict, mode='w') as file:
    json.dump(norm_dict, file)
    print(f"Saved normalisation dictionary in {file.name}")

data = data.fillna(float(0))

X = data.drop(columns=[y_column])
y = data[y_column]

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device selected: {device}")


# Create custom dataset object
class MyDataset(Dataset):
    def __init__(self, x_df, y_df):
        self.X = torch.tensor(x_df.to_numpy(dtype=float)).to(device)
        self.y = torch.tensor(y_df.to_numpy(dtype=float)).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Create dataset and split into training and test
train_dataset, test_dataset = random_split(MyDataset(X, y),
                                           [1 - args.ratio, args.ratio])  # 75% training, 25% test

# Wrap datasets in Dataloaders
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

input_width = len(X.columns)


# Define model
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


model = MLPRegressor().to(device)

loss_fn = torch.nn.MSELoss().to(device)
optimiser = torch.optim.Adam(model.parameters())
r2_metric = R2Score().to(device)


# Define training function
def train_epoch(verbose: bool = False):
    loss, r2 = [None] * 2
    for batch_index, sample in enumerate(train_loader):
        ins, out = sample
        out = out.unsqueeze(1)
        optimiser.zero_grad()

        predict = model(ins)
        loss = loss_fn(predict, out)
        loss.backward()

        r2_metric.update(predict, out)
        r2 = r2_metric.compute()

        optimiser.step()

        if verbose and batch_index % 10 == 0:
            print(f"Batch {batch_index}\tLoss: {loss}\tR2: {r2}")

    return loss, r2


# Train the model
best_mse = 1
best_R2 = 0
epochs = 0
model.train(True)

while ((args.epochs and epochs < args.epochs) or
       (args.mse and best_mse > args.mse) or
       (args.r2 and best_R2 < args.r2)):
    epochs += 1
    print(f"\nEpoch: {epochs}\n")

    mse, R2 = train_epoch(epochs == 1)
    print(f"Epoch {epochs} done\tMSE: {mse}\tR2: {R2}")

    best_mse = min(best_mse, mse)
    best_R2 = max(best_R2, R2)

print("Finished training\n")
# Evaluate the model
model.eval()

sum_mse = 0
sum_r2 = 0

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        inputs, expected = batch
        # inputs = inputs.to(device)
        expected = expected.unsqueeze(1)
        actual = model(inputs)

        mse = loss_fn(actual, expected)
        r2_metric.update(actual, expected)
        R2 = r2_metric.compute()

        print(f"Validating batch {i}\tMSE: {mse}\tR2: {R2}")

        sum_mse += mse
        sum_r2 += R2

print(f"Testing done:\tAverage MSE: {sum_mse / len(test_loader)}\tAverage R2: {sum_r2 / len(test_loader)}")

# Save the model
filename = f'model-{datetime.now().strftime("%Y_%m_%d-%H_%M")}.pth'
torch.save(model.state_dict(), filename)
print(f"Saved model weights at {filename}")
