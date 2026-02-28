import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

<<<<<<< HEAD

SEQ_LEN = 30       
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001
=======
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
>>>>>>> e5716e7 (Added vectorbt integration and daily prediction model)

import vectorbt as vbt



df = pd.read_csv("data/output.csv")



df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)


daily = df.resample('1D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})

daily.dropna(inplace=True)


daily['return'] = daily['close'].pct_change()
daily['ma7'] = daily['close'].rolling(7).mean()
daily['ma14'] = daily['close'].rolling(14).mean()
daily['volatility'] = daily['return'].rolling(7).std()

daily.dropna(inplace=True)


daily['target'] = (daily['close'].shift(-1) > daily['close']).astype(int)
daily.dropna(inplace=True)


features = ['return', 'ma7', 'ma14', 'volatility']
X = daily[features]
y = daily['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False, test_size=0.2
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


daily['prediction'] = model.predict(X)


entries = daily['prediction'] == 1
exits = daily['prediction'] == 0


portfolio = vbt.Portfolio.from_signals(
    close=daily['close'],
    entries=entries,
    exits=exits,
    init_cash=10000,
    fees=0.001,
    slippage=0.001,
    freq='1D'
)


print("Model Accuracy:", accuracy_score(y_test, model.predict(X_test)))
print(portfolio.stats())


daily.to_csv("predictions.csv")

<<<<<<< HEAD
df = convert_to_daily(df)

print("Converted to daily candles:", len(df))


df = create_features(df)

df["day_of_week"] = df["timestamp"].dt.dayofweek

df = df.dropna()


features = df.drop(columns=["target", "timestamp"])
features = features.select_dtypes(include=[np.number])

features.replace([np.inf, -np.inf], np.nan, inplace=True)
features = features.ffill().bfill()
features = features.clip(-10, 10)

target = df["target"]

print("Feature shape:", features.shape)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

X = X_scaled.astype(np.float32)
y = target.values.astype(np.float32)

split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_dataset = CryptoDataset(X_train, y_train, SEQ_LEN)
test_dataset = CryptoDataset(X_test, y_test, SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print("Training samples:", len(train_dataset))


model = TemporalCNN(num_features=X.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()


print("Training model...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        preds = model(X_batch)
        loss = loss_fn(preds, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {avg_loss:.6f}")



model.eval()
preds = []
actual = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        out = model(X_batch).cpu().numpy()
        preds.extend(out)
        actual.extend(y_batch.numpy())

preds = np.array(preds)
actual = np.array(actual)

direction_accuracy = np.mean((preds > 0) == (actual > 0))
print("Accuracy:", direction_accuracy)

MODEL_PATH = os.path.join(BASE_DIR, "daily_temporal_cnn.pth")
torch.save(model.state_dict(), MODEL_PATH)

print("Model saved:", MODEL_PATH)
=======
print("Saved predictions to predictions.csv")
>>>>>>> e5716e7 (Added vectorbt integration and daily prediction model)
