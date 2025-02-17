import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the model without compilation
model = load_model("poetry_model.h5", compile=False)

# Save the model again with a fixed name
model.save("poetry_model_fixed.h5")

print("✅ Model successfully re-saved as 'poetry_model_fixed.h5'")
