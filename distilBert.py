from transformers import (AutoTokenizer, DefaultDataCollator, AutoModelForQuestionAnswering,
                          TrainingArguments, Trainer, pipeline)
from datasets import load_dataset


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


def training(output_dir: str, model: AutoModelForQuestionAnswering, train_dataset, test_dataset,
             tokenizer: AutoTokenizer, data_collator: DefaultDataCollator):
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_gpu_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,  # no connection to Hugging HUb
        report_to=None  # it require the set up of the wandb, will do it probably
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
    trainer.save_model('./model')
    tokenizer.save_pretrained('./model')


def prepared_squad(tokenizer):
    """
    It downloads and prepare the SQuAD(Stanford Question Answering Dataset) for training
    :return:
    tokenized_squad: tokenized SQuAD
    """
    squad = load_dataset("squad", split="train[:5000]")
    squad = squad.train_test_split(test_size=0.2)

    def preprocess_function_squad(examples):
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

    tokenized_squad = squad.map(preprocess_function_squad, batched=True, remove_columns=squad["train"].column_names)
    return tokenized_squad, tokenizer


def initialize_model_with_squad():
    data_collator, tokenizer, model = initialize_tokenizer_model_collator()
    tokenized_squad, tokenizer = prepared_squad(tokenizer)
    training(output_dir='qa_sample1', model=model, train_dataset=tokenized_squad['train'],
             test_dataset=tokenized_squad['test'], tokenizer=tokenizer, data_collator=data_collator)


def question_answer(model_path, question, context):
    question_answerer = pipeline("question-answering", model=model_path, tokenizer=model_path)
    return question_answerer(question=question, context=context)


if __name__ == "__main__":
    question = 'How many programming languages does BLOOM support?'
    context = ("BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 "
               "programming languages.")
    print(question_answer(model_path='./model', question=question,context=context))
    question = ('How much higher are the post-test odds of a high RDI compared to the pre-test odds following '
                'a positive test?')
    context = ('Based on a moderate classification threshold from the boosting algorithm, the estimated post-test odds '
               'of a high RDI were 2.20 times higher than the pre-test odds given a positive test, while the '
               'corresponding post-test odds were decreased by 52% given a negative test (sensitivity and specificity '
               'of 0.66 and 0.70, respectively).')
    print(question_answer(model_path='./model', question=question, context=context))

