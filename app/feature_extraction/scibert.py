import numpy as np

from .base import FeatureExtractorBase


class Scibert(FeatureExtractorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(tokenizer="TF-IDF", *args, **kwargs)

    def _scibert_helper(self, data):
        # Mean Pooling - Take attention mask into account for correct averaging
        import torch
        from transformers import AutoModel, AutoTokenizer

        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[
                0
            ]  # First element of model_output contains all token embeddings
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

        # Load AutoModel from huggingface model repository
        tokenizer = AutoTokenizer.from_pretrained("gsarti/scibert-nli")
        model = AutoModel.from_pretrained("gsarti/scibert-nli")

        # Tokenize sentences
        data = list(data)
        encoded_input = tokenizer(
            data, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling. In this case, mean pooling
        return np.array(mean_pooling(model_output, encoded_input["attention_mask"]))

    def vectorize_init(self):
        features_vectorized = np.concatenate(
            (
                self._scibert_helper(self.data["title"]),
                self._scibert_helper(self.data["abstract"]),
            ),
            axis=1,
        )
        test_set_vectorized = features_vectorized.copy()
        return features_vectorized, test_set_vectorized
