import numpy as np
#& 'C:\Users\Userhome\Documents\code\py9 + easter\py29\Scripts\Activate.ps1' 
X = np.load('gesture_landmarks.npy')
y = np.load('gesture_labels.npy')

keep = y != 'name of the gesture'
np.save('gesture_landmarks.npy', X[keep])
np.save('gesture_labels.npy', y[keep])
print(f"Removed. Remaining: {dict(zip(*np.unique(y[keep], return_counts=True)))}")