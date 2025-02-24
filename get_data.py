from datasets import load_dataset

# Load only the first 10,000 samples of the training split.
train_data = load_dataset('codeparrot/codeparrot-clean-train', split='train[:2000]')

# Save the subset in JSON lines format.
train_data.to_json("codeparrot_data.json", lines=True)
