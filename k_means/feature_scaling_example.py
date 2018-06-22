from sklearn.preprocessing import MinMaxScaler
import numpy as np


weights = np.array([[115.0],
                    [140.0],
                    [175.0]])
scaler = MinMaxScaler()
rescaled_weights = scaler.fit_transform(weights)

print(rescaled_weights)
