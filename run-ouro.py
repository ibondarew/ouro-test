import os
import time

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class TimeLimitCallback(TrainerCallback):
    """
    Колбэк для безопасного завершения обучения по таймеру.
    Синхронизирован между всеми GPU для предотвращения зависаний (deadlocks).
    """

    def __init__(self, max_time_seconds):
        self.max_time = max_time_seconds
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is None:
            return

        elapsed = time.time() - self.start_time
        local_stop = elapsed >= self.max_time

        # Синхронизируем флаг остановки между всеми активными GPU
        if dist.is_initialized():
            stop_tensor = torch.tensor([1.0 if local_stop else 0.0], device=args.device)
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            should_stop = stop_tensor.item() > 0.5
        else:
            should_stop = local_stop

        if should_stop:
            control.should_training_stop = True
            # Печатаем лог только из главного процесса (GPU 0)
            if args.process_index == 0:
                print(
                    f"\n[TimeLimitCallback] Достигнут лимит времени ({self.max_time} сек). Завершаем обучение..."
                )


def main():
    # Читаем время работы из переменной окружения (по умолчанию 3600 секунд / 1 час)
    max_time_seconds = int(os.getenv("TRAINING_DURATION_SECONDS", "3600"))

    model_name = "ByteDance/Ouro-2.6B-Thinking"

    # 1. Загрузка токенизатора и модели (trust_remote_code=True обязателен для Ouro)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # bfloat16 — нативный и самый быстрый формат для H100
    )

    # 2. ПОДГОТОВКА ДАТАСЕТА (с упаковкой в блоки по 2048 токенов)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    def tokenize_function(examples):
        # Просто переводим текст в токены без обрезки на данном этапе
        return tokenizer(examples["text"], add_special_tokens=False)

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    def group_texts(examples):
        block_size = 2048
        # Склеиваем все списки токенов в один сплошной поток (избавляет от пустых строк)
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # Округляем до длины, кратной block_size
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        # Нарезаем на плотные куски ровно по block_size токенов
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    # Финальный упакованный датасет без пустых элементов
    packed_dataset = tokenized_dataset.map(group_texts, batched=True, batch_size=1000)

    # 3. Конфигурация обучения
    training_args = TrainingArguments(
        output_dir="./ouro_output",
        overwrite_output_dir=True,
        per_device_train_batch_size=8,  # Безопасный батч для старта
        gradient_accumulation_steps=2,  # Аккумуляция для компенсации меньшего числа GPU (3 вместо 8)
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=500,
        bf16=True,  # Обязательно для H100
        fp16=False,
        max_steps=1000000,  # Ограничением управляет TimeLimitCallback
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        report_to="none",
        gradient_checkpointing=True,  # Жёсткая защита от OOM (высвобождает ~50-60% памяти)
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # 4. Инициализация Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=packed_dataset,  # Используем подготовленный плотный датасет
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[TimeLimitCallback(max_time_seconds)],
    )

    # Запуск процесса
    trainer.train()

    # Сохраняем веса после остановки (только на главном GPU)
    if trainer.is_world_process_zero():
        print("Сохранение финальной модели...")
        trainer.save_model("./ouro_final_model")
        tokenizer.save_pretrained("./ouro_final_model")


if __name__ == "__main__":
    main()
