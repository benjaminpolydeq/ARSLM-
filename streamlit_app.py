from transformers import BertTokenizer
import random

class ARSLM:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def predict(self, sequences):
        results = []
        for seq in sequences:
            random_score = round(random.uniform(0, 1), 4)
            results.append({
                "sequence": seq,
                "prediction": random_score
            })
        return results
