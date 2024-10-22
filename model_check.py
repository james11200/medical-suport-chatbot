import h5py
from keras.models import load_model

# Load the Keras model from the HDF5 file
file_path = 'chatbot_model_lstm.h5'
model = load_model(file_path)

# Open the HDF5 file
h5_file = h5py.File(file_path, 'r')
h5_keys = h5_file.keys()

# List all the keys (datasets) in the file
print("Keys in HDF5 file:", list(h5_file.keys()))
for i in h5_keys:
    print(i)
print(h5_file)
# Close the HDF5 file after use
h5_file.close()
