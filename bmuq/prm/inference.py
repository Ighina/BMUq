from safetensors.torch import load_file as safe_load_file
from torch import nn
import torch
from typing import List
from transformers import BertConfig
import json
import sys

from bmuq.prm.featurizer import PRMFeaturizer
from bmuq.prm.train import BertForTokenClassificationWithEmbeddings


class InferenceBertForTokenClassificationWithEmbeddings(nn.Module):
    """
    BERT model for token-level classification that accepts pre-computed embeddings.

    This model is designed to work with sentence embeddings from sentence-transformers
    and skips the embedding layer of BERT.
    """

    def __init__(self, pretrained_model: str, featurizer_model: str):
        super().__init__()
        config = BertConfig.from_pretrained(pretrained_model)
        state_dict = safe_load_file(
            f"{pretrained_model}/model.safetensors", device="cpu"
        )
        self.model = BertForTokenClassificationWithEmbeddings(config)

        self.model.load_state_dict(state_dict, strict=False)

        self.featiurizer = PRMFeaturizer(
            featurizer_type="embeddings", model_name_or_path=featurizer_model
        )

    def forward(
        self,
        inputs_text: List[str],
    ) -> List[List[float]]:
        """
        Forward pass using pre-computed embeddings.

        Args:
            inputs_text: The steps to analyze: in inference accepts only 2-D lists (i.e. one
            reasoning chain at the time)

        Returns:
            Dictionary containing loss (if labels provided) and logits
        """
        with torch.no_grad():
            inputs_embeds = self.featiurizer(inputs_text)

            if inputs_embeds.dim() == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)

            # Pass through BERT encoder
            logits = self.model(
                inputs_embeds=inputs_embeds,
            )["logits"]

            probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()

        return probs


class UnsupervisedCoherencePRM(nn.Module):
    """
    BERT model for token-level classification that accepts pre-computed embeddings.

    This model is designed to work with sentence embeddings from sentence-transformers
    and skips the embedding layer of BERT.
    """

    def __init__(self, pretrained_model: str, featurizer_model: str):
        super().__init__()
        config = BertConfig.from_pretrained(pretrained_model)
        state_dict = safe_load_file(
            f"{pretrained_model}/model.safetensors", device="cpu"
        )
        self.model = BertForTokenClassificationWithEmbeddings(config)

        self.model.load_state_dict(state_dict, strict=False)

        self.featiurizer = PRMFeaturizer(
            featurizer_type="embeddings", model_name_or_path=featurizer_model
        )

    def forward(
        self,
        inputs_text: List[str],
    ) -> List[List[float]]:
        """
        Forward pass using pre-computed embeddings.

        Args:
            inputs_text: The steps to analyze: in inference accepts only 2-D lists (i.e. one
            reasoning chain at the time)

        Returns:
            Dictionary containing loss (if labels provided) and logits
        """
        with torch.no_grad():
            inputs_embeds = self.featiurizer(inputs_text)

            if inputs_embeds.dim() == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)

            # Pass through BERT encoder
            logits = self.model(
                inputs_embeds=inputs_embeds,
            )["logits"]

            probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()

        return probs


if __name__ == "__main__":
    pretrained_model = sys.argv[1]
    featurizer_model = sys.argv[2]
    dataset = sys.argv[3]

    model = InferenceBertForTokenClassificationWithEmbeddings(
        pretrained_model=pretrained_model,
        featurizer_model=featurizer_model,
    )

    model.to("mps")

    if dataset == "reasoneval-incorrect":
        with open("bmuq/prm/data/reasoneval/mr-math_invalid_errors.json", "r") as f:
            data = []
            for line in f.readlines():
                data.append(json.loads(line))
    elif dataset == "reasoneval-redundant":
        with open("bmuq/prm/data/reasoneval/mr-math_redundant_errors.json", "r") as f:
            data = []
            for line in f.readlines():
                data.append(json.loads(line))
    elif dataset == "reasoneval-gsm8k":
        with open("data/reasoneval/mr-gsm8k.json", "r") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    outputs = []
    for item in data:
        input_text = [item["question"]] + [
            it[0] for it in item["model_output_step_format"]
        ]
        print(input_text)
        output = model(
            inputs_text=input_text,
        )
        outputs.append(output)

    with open("prm_inference.jsonl", "w") as f:
        for output in outputs:
            f.write(json.dumps(f"{output}\n"))
