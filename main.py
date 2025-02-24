import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# nltk.download('punkt')


# -------------------------------
# Загрузка и подготовка реального датасета
# -------------------------------
def load_translation_dataset(num_samples=500):
    """
    Загружает датасет WMT16 для перевода с английского на немецкий.
    Из датасета выбирается случайное подмножество из num_samples примеров.
    Формируется пара: source - "translate English to German: {en_text}"
                     target - соответствующий немецкий перевод.
    """
    dataset = load_dataset("wmt16", "de-en", split="train")
    # Перемешиваем датасет и берем подмножество
    dataset = dataset.shuffle(seed=42).select(range(num_samples))
    texts = []
    targets = []
    for example in dataset:
        en_text = example["translation"]["en"].strip()
        de_text = example["translation"]["de"].strip()
        # Формируем вход в формате T5 для перевода
        texts.append(f"translate English to German: {en_text}")
        targets.append(de_text)
    return texts, targets


# -------------------------------
# Класс Dataset для перевода
# -------------------------------
class TranslationDataset(Dataset):
    def __init__(self, tokenizer, texts, targets, max_length=64):
        self.tokenizer = tokenizer
        self.texts = texts
        self.targets = targets
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        source = self.tokenizer.encode_plus(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.encode_plus(
            self.targets[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": source["input_ids"].squeeze(0),
            "attention_mask": source["attention_mask"].squeeze(0),
            "labels": target["input_ids"].squeeze(0),
            "target_text": self.targets[idx]  # для оценки качества
        }


# -------------------------------
# Класс для Prompt Tuning модели
# -------------------------------
class PromptTuningT5(nn.Module):
    def __init__(self, model_name, prompt_length):
        """
        model_name: название предобученной модели T5 (например, "t5-small")
        prompt_length: количество soft-токенов, добавляемых к каждому входу
        """
        super(PromptTuningT5, self).__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        # Замораживаем все параметры T5
        for param in self.t5.parameters():
            param.requires_grad = False

        self.hidden_size = self.t5.config.d_model
        self.prompt_length = prompt_length

        # Инициализируем параметры soft prompt: размер [prompt_length, hidden_size]
        self.soft_prompt = nn.Parameter(torch.randn(prompt_length, self.hidden_size))

    def forward(self, input_ids, attention_mask, labels=None):
        # Получаем эмбеддинги входных токенов
        inputs_embeds = self.t5.encoder.embed_tokens(input_ids)
        batch_size = input_ids.size(0)
        soft_prompt_expanded = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        # Конкатенируем soft prompt и эмбеддинги входного текста
        inputs_embeds = torch.cat([soft_prompt_expanded, inputs_embeds], dim=1)
        # Обновляем attention_mask, добавляя единицы для soft prompt
        prompt_attention = torch.ones(batch_size, self.prompt_length, device=attention_mask.device,
                                      dtype=attention_mask.dtype)
        attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)
        # Передаём изменённые эмбеддинги и маску в T5
        outputs = self.t5(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs


# -------------------------------
# Функция тренировки модели
# -------------------------------
def train_model(model, dataloader, optimizer, device, num_epochs=3, model_type="Prompt Tuning"):
    model.to(device)
    model.train()
    total_steps = len(dataloader) * num_epochs
    print(f"\n[{model_type}] Начало тренировки на {total_steps} шагах.")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        print(f"\n[{model_type}] Эпоха {epoch + 1}/{num_epochs} началась.")
        for step, batch in enumerate(dataloader, start=1):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if step % 10 == 0 or step == len(dataloader):
                print(f"[{model_type}] Эпоха {epoch + 1}, шаг {step}/{len(dataloader)}: потеря = {loss.item():.4f}")
        avg_loss = epoch_loss / len(dataloader)
        print(f"[{model_type}] Эпоха {epoch + 1} завершена, Средняя потеря: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"\n[{model_type}] Тренировка завершена за {elapsed:.2f} секунд.")
    return elapsed


# -------------------------------
# Функция оценки качества модели с использованием BLEU score
# -------------------------------
def evaluate_model(model, dataloader, tokenizer, device, model_type="Prompt Tuning"):
    model.eval()
    bleu_scores = []
    smoothing = SmoothingFunction().method1
    print(f"\n[{model_type}] Начало оценки качества на тестовом датасете.")
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            generated_ids = model.t5.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=64,
            )
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            targets = batch["target_text"]
            for pred, target in zip(preds, targets):
                score = sentence_bleu([target.split()], pred.split(), smoothing_function=smoothing)
                bleu_scores.append(score)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"[{model_type}] Средний BLEU score: {avg_bleu:.4f}")
    return avg_bleu


# -------------------------------
# Пример классического fine-tuning для сравнения
# -------------------------------
def fine_tuning_example(model_name, dataloader, optimizer, device, num_epochs=3):
    model_ft = T5ForConditionalGeneration.from_pretrained(model_name)
    model_ft.to(device)
    model_ft.train()
    start_time = time.time()
    print("\n[Fine-Tuning] Начало тренировки fine-tuning модели.")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        print(f"\n[Fine-Tuning] Эпоха {epoch + 1}/{num_epochs} началась.")
        for step, batch in enumerate(dataloader, start=1):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model_ft(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if step % 10 == 0 or step == len(dataloader):
                print(f"[Fine-Tuning] Эпоха {epoch + 1}, шаг {step}/{len(dataloader)}: потеря = {loss.item():.4f}")
        avg_loss = epoch_loss / len(dataloader)
        print(f"[Fine-Tuning] Эпоха {epoch + 1} завершена, Средняя потеря: {avg_loss:.4f}")
    elapsed = time.time() - start_time
    print(f"\n[Fine-Tuning] Тренировка завершена за {elapsed:.2f} секунд.")
    return model_ft, elapsed


# -------------------------------
# Главная функция эксперимента
# -------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "t5-small"  # Можно заменить на более крупную модель для экспериментов
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    print("Загрузка реального датасета перевода (WMT16) ...")
    texts, targets = load_translation_dataset(num_samples=1000)
    dataset = TranslationDataset(tokenizer, texts, targets, max_length=64)

    # Разбиваем датасет на train и test (например, 80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    dataloader_train = DataLoader(train_dataset, batch_size=8, shuffle=True)
    dataloader_test = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # -------------------------------
    # Prompt Tuning
    # -------------------------------
    prompt_length = 5  # Количество soft-токенов
    prompt_tuning_model = PromptTuningT5(model_name, prompt_length)
    optimizer_pt = optim.Adam([prompt_tuning_model.soft_prompt], lr=1e-3)

    print("\n=== Запуск Prompt Tuning ===")
    pt_train_time = train_model(prompt_tuning_model, dataloader_train, optimizer_pt, device, num_epochs=3,
                                model_type="Prompt Tuning")
    if device.type == "cuda":
        mem_pt = torch.cuda.memory_allocated(device)
        print(f"[Prompt Tuning] Использованная GPU память: {mem_pt / 1e6:.2f} MB")

    pt_bleu = evaluate_model(prompt_tuning_model, dataloader_test, tokenizer, device, model_type="Prompt Tuning")

    # -------------------------------
    # Fine-Tuning
    # -------------------------------
    print("\n=== Запуск Fine-Tuning ===")
    # Создаём новый экземпляр модели для fine-tuning
    ft_model = T5ForConditionalGeneration.from_pretrained(model_name)
    ft_model.to(device)
    optimizer_ft = optim.Adam(ft_model.parameters(), lr=1e-4)
    model_ft, ft_train_time = fine_tuning_example(model_name, dataloader_train, optimizer_ft, device, num_epochs=3)
    if device.type == "cuda":
        mem_ft = torch.cuda.memory_allocated(device)
        print(f"[Fine-Tuning] Использованная GPU память: {mem_ft / 1e6:.2f} MB")

    # Оценка fine-tuned модели
    ft_model.eval()
    bleu_scores_ft = []
    smoothing = nltk.translate.bleu_score.SmoothingFunction().method1
    print("\n[Fine-Tuning] Начало оценки качества fine-tuned модели.")
    with torch.no_grad():
        for batch in dataloader_test:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            generated_ids = ft_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=64,
            )
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            targets_batch = batch["target_text"]
            for pred, target in zip(preds, targets_batch):
                score = sentence_bleu([target.split()], pred.split(), smoothing_function=smoothing)
                bleu_scores_ft.append(score)
    ft_bleu = sum(bleu_scores_ft) / len(bleu_scores_ft)
    print(f"[Fine-Tuning] Средний BLEU score: {ft_bleu:.4f}")

    # -------------------------------
    # Итоговый отчет
    # -------------------------------
    print("\n=== Итоговый отчет ===")
    print(f"Device type: {device.type}")
    print(f"Prompt Tuning: время обучения = {pt_train_time:.2f} сек, BLEU = {pt_bleu:.4f}")
    print(f"Fine-Tuning:   время обучения = {ft_train_time:.2f} сек, BLEU = {ft_bleu:.4f}")
    if device.type == "cuda":
        print(f"Prompt Tuning: GPU память = {mem_pt / 1e6:.2f} MB")
        print(f"Fine-Tuning:   GPU память = {mem_ft / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
