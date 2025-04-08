import numpy as np
import RandomForest

X = np.array([
    # Normal transactions
    [25.50, 24.3, 1.2],     # Small amount, reasonable time gap, close to home
    [32.40, 12.7, 2.1],     # Small amount, reasonable time gap, close to home
    [15.20, 48.2, 0.5],     # Small amount, large time gap, very close to home
    [42.80, 36.5, 3.2],     # Medium amount, reasonable time gap, close to home
    [67.30, 72.1, 5.6],     # Medium amount, large time gap, reasonable distance
    [110.50, 50.3, 4.8],    # Larger amount, reasonable time gap, reasonable distance
    [95.20, 48.6, 6.2],     # Larger amount, reasonable time gap, reasonable distance
    [150.40, 168.2, 8.5],   # Larger amount, very large time gap, further distance
    [212.70, 36.8, 7.3],    # Larger amount, reasonable time gap, reasonable distance
    [45.60, 12.4, 2.8],     # Medium amount, reasonable time gap, close to home
    [38.20, 5.6, 1.9],      # Medium amount, small time gap, close to home
    [125.90, 24.5, 3.7],    # Larger amount, reasonable time gap, close to home
    
    # Fraudulent transactions
    [999.99, 0.8, 156.2],   # Very large amount, very small time gap, very far from home
    [875.40, 1.2, 245.7],   # Very large amount, very small time gap, very far from home
    [1250.30, 0.5, 542.3],  # Very large amount, very small time gap, extremely far from home
    [450.80, 0.3, 325.9],   # Large amount, very small time gap, very far from home
    [685.20, 0.6, 125.4],   # Large amount, very small time gap, very far from home
    [522.70, 2.4, 98.6],    # Large amount, small time gap, far from home
    [756.90, 1.7, 187.3],   # Large amount, small time gap, far from home
    [325.60, 0.4, 256.9]    # Medium-large amount, very small time gap, very far from home
])

y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

model = RandomForest.RandomForest(n_estimators=10)
model.fit(X, y)

predict = model.predict(np.array([
    [125.90, 24.5, 3.7],
    [325.60, 0.4, 256.9],
]))
print("Prediction:")
print(predict)
