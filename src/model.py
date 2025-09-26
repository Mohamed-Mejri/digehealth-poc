import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput

from src.base.base_model import HubertForAudioFrameClassification


class DigeHealthModel(nn.Module):

    def __init__(self, pretrained_model_name_or_path: str, num_labels: int):
        super(DigeHealthModel, self).__init__()
        base_model = HubertForAudioFrameClassification.from_pretrained(
            pretrained_model_name_or_path
        )
        self.backbone = base_model.hubert
        in_features = self.backbone.encoder.layers[
            -1
        ].final_layer_norm.normalized_shape[0]
        self.classifier = nn.Linear(in_features, num_labels)
        self.num_labels = num_labels
        self.config = base_model.config

        num_layers = (
            self.config.num_hidden_layers + 1
        )  # transformer layers + input embeddings
        if self.config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

    def freeze_feature_extractor(self):
        self.backbone.feature_extrator._freeze_parameters()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | TokenClassifierOutput:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_hidden_states = (
            True if self.config.use_weighted_layer_sum else output_hidden_states
        )

        outputs = self.backbone(
            input_values,
            output_attentions=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[1]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = F.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
