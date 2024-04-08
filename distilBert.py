from transformers import AutoTokenizer, DefaultDataCollator, AutoModelForQuestionAnswering, TrainingArguments, Trainer
import wandb


def initialize_tokenizer_model_collator():
    """
    Initialize new AutoTokenizer AutoModel Data collator
    :return:
    data_collator
    tokenizer： from AutoTokenizer
    model: from AutoModelForQuestionAnswering
    """

    data_collator = DefaultDataCollator()
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")

    return data_collator, tokenizer, model


def preprocess_function_squad(examples, tokenizer):
    """
    It is a preprocessing example from the hugging hub
    :param examples:
    :param tokenizer:
    :return:
    """
    questions = [q.strip() for q in examples["question"]]  # Strip the question
    inputs = tokenizer(  # tokenize input
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",  # if len(questions+context) > max_input, only context will be truncated to fit
        return_offsets_mapping=True,  # offset mapping in the tokenizers output, map token position to the character
        # position in the original text
        padding="max_length",  # ensure all tokenized input are padded to the same length (max_length)
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def training(output_dir: str, model: AutoModelForQuestionAnswering, train_dataset, test_dataset, tokenizer: AutoTokenizer, data_collator: DefaultDataCollator):
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_gpu_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()


if __name__ == "__main__":


    print('test')







