import pandas as pd
import json
from tensorflow.keras.preprocessing.text import Tokenizer

# Path to the CSV file containing the captions
csv_file_path = r"D:\\Internship\\1\\train\\radiologytraindata.csv"

# Load the CSV file
data = pd.read_csv(csv_file_path)

# Extract the captions
texts = data['caption'].tolist()

# Create and fit the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Save the tokenizer
tokenizer_json = tokenizer.to_json()
with open(r'D:\Internship\1\saved_models\tokenizer.json', 'w') as f:

    f.write(tokenizer_json)

print("Tokenizer created and saved successfully.")
