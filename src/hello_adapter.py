import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdapterType

model = AutoModelForSequenceClassification.from_pretrained("../models/sst-2/batch_32_64/checkpoint-2000")
model.load_adapter("../adapters/sst-2/batch_32_64/sst-2/")

#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#tokens = tokenizer.tokenize("AdapterHub is awesome!")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("AdapterHub is awesome!")
input_tensor = torch.tensor([
    tokenizer.convert_tokens_to_ids(tokens)
])
outputs = model(
    input_tensor,
    adapter_names=['sst-2']
)


def predict(sentence):
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))
    input_tensor = torch.tensor([token_ids])

    # predict output tensor
    outputs = model(input_tensor, adapter_names=['sst-2'])

    # retrieve the predicted class label
    return 'positive' if 1 == torch.argmax(outputs[0]).item() else 'negative'


print("Those who find ugly meanings in beautiful things are corrupt without being charming.",
      end=' -sentiment-> ')
print(predict("Those who find ugly meanings in beautiful things are corrupt without being charming."))
print("There are slow and repetitive parts, but it has just enough spice to keep it interesting.", end=' -sentiment-> ')
print(predict("There are slow and repetitive parts, but it has just enough spice to keep it interesting."))
