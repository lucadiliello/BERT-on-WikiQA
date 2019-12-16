import torch
from transformers import BertPreTrainedModel, BertModel
from torch import nn


class BertForWikiQA(BertPreTrainedModel):
    r"""
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"
        input_ids = tokenizer.encode(input_text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        print(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))
        # a nice puppet
    """
    def __init__(self, config):
        super(BertForWikiQA, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, 
                token_type_ids=None, position_ids=None, 
                head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        logits = self.qa_outputs(outputs[0][:, 0, :]).squeeze()
        predicted_labels = nn.functional.softmax(logits, dim=-1)

        if labels is not None:
            loss = self.loss_fn(predicted_labels, labels)
            return loss, predicted_labels
        else:
            return predicted_labels