import numpy as np
import pandas as pd

fs = 1000  # Flutter sends 100Hz (every 10th sample)
t  = np.arange(0, 60, 1/fs)
# fake a contracting muscle: noise centered at 0, amplitude ~100 ADC RMS
rng = np.random.default_rng(42)
emg = rng.normal(0, 1000, len(t))
df  = pd.DataFrame({'time': t, 'emgData': emg})
df.to_csv('flutter_format_test.csv', index=False)
print("Saved flutter_format_test.csv")