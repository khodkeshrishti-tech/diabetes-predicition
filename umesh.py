  # Split data 80% train 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("=== DATA SPLIT ===")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\n=== MODEL TRAINED SUCCESSFULLY ===")