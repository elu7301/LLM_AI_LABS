import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from torch.utils.data import DataLoader

# Определяем устройство: GPU, если доступно, иначе CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(model_name="gpt2", num_prompt_tokens=5):
    """
    Загружает предобученную модель GPT-2 и токенизатор.
    Создает обучаемые prompt-токены и перемещает модель на выбранное устройство.
    Замораживает все параметры модели, чтобы обновлялись только prompt-токены.
    """
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Создаем обучаемые prompt-токены: размер (num_prompt_tokens, hidden_size)
    prompt_tuning_tokens = torch.nn.Parameter(torch.randn(num_prompt_tokens, model.config.n_embd, device=device))

    # Замораживаем все параметры модели, чтобы обновлялись только prompt-токены
    for param in model.parameters():
        param.requires_grad = False

    return model, tokenizer, prompt_tuning_tokens


def prepare_sample(source_text, target_text, tokenizer, prompt_tuning_tokens, model):
    """
    Подготавливает один обучающий пример.

    Формируется входная последовательность из трех частей:
      1. Обучаемые prompt-токены.
      2. Эмбеддинги исходного текста (source_text).
      3. Эмбеддинги целевого текста (target_text).

    Формируются метки (labels), где для prompt и source ставится значение -100 (игнорирование),
    а для target — реальные id, по которым считается кросс-энтропия.
    """
    # Токенизация исходного и целевого текстов (без добавления спец. токенов)
    source = tokenizer(source_text, add_special_tokens=False, return_tensors="pt")
    target = tokenizer(target_text, add_special_tokens=False, return_tensors="pt")

    source_ids = source.input_ids.to(device)  # [1, source_length]
    target_ids = target.input_ids.to(device)  # [1, target_length]

    # Получаем эмбеддинги исходного и целевого текста
    with torch.no_grad():
        source_embeds = model.transformer.wte(source_ids)  # [1, source_length, hidden_size]
        target_embeds = model.transformer.wte(target_ids)  # [1, target_length, hidden_size]

    # Приводим обучаемые prompt-токены к размеру батча
    prompt_embeds = prompt_tuning_tokens.unsqueeze(0)  # [1, num_prompt_tokens, hidden_size]

    # Объединяем эмбеддинги: prompt + source + target
    input_embeds = torch.cat([prompt_embeds, source_embeds, target_embeds], dim=1)

    # Формируем метки: для prompt и source игнорируем (-100), для target — реальные id
    batch_size = 1
    num_prompt = prompt_tuning_tokens.size(0)
    source_len = source_ids.size(1)
    target_len = target_ids.size(1)
    total_length = num_prompt + source_len + target_len

    labels = torch.full((batch_size, total_length), -100, dtype=torch.long, device=device)
    labels[:, num_prompt + source_len:] = target_ids  # Вычисление loss только по target-токенам

    return input_embeds, labels


def train_epoch(model, tokenizer, prompt_tuning_tokens, dataloader, optimizer):
    """
    Осуществляет одну эпоху обучения по даталоадеру.

    Для каждого примера:
      - Извлекаются исходный и целевой тексты из поля "translation".
      - Подготавливается вход (с prompt-токенами) и метки.
      - Вычисляется loss и обновляются только prompt-токены.
    """
    model.train()  # Режим обучения (даже если модель заморожена)
    total_loss = 0.0

    for batch in dataloader:
        # Извлекаем переводы из батча.
        # В батче поле "translation" представляет собой словарь с ключами "en" и "fr"
        translation_en = batch["translation"]["en"][0]
        translation_fr = batch["translation"]["fr"][0]

        source_text = "Translate this to French: " + translation_en
        target_text = translation_fr

        optimizer.zero_grad()
        input_embeds, labels = prepare_sample(source_text, target_text, tokenizer, prompt_tuning_tokens, model)

        # Прямой проход через модель с вычислением loss
        outputs = model(inputs_embeds=input_embeds, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def generate_translation(model, tokenizer, prompt_tuning_tokens, source_text, max_gen_length=20):
    """
    Генерирует перевод для заданного исходного текста.

    Объединяет обучаемые prompt-токены с эмбеддингами исходного текста,
    затем использует метод generate для получения продолжения (перевода).
    """
    model.eval()
    # Токенизация исходного текста
    source = tokenizer(source_text, add_special_tokens=False, return_tensors="pt")
    source_ids = source.input_ids.to(device)

    with torch.no_grad():
        source_embeds = model.transformer.wte(source_ids)

    prompt_embeds = prompt_tuning_tokens.unsqueeze(0)
    input_embeds = torch.cat([prompt_embeds, source_embeds], dim=1)

    total_input_length = input_embeds.size(1)

    # Генерация продолжения
    output = model.generate(
        inputs_embeds=input_embeds,
        max_length=total_input_length + max_gen_length,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id  # чтобы избежать предупреждений
    )

    # Отбрасываем часть prompt и source, оставляем только сгенерированный перевод
    generated_tokens = output[0][total_input_length:]
    translation = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return translation


def main():
    # Загружаем большой датасет перевода с английского на французский
    dataset = load_dataset("wmt14", "fr-en", split="train")

    # Для демонстрации ограничим датасет, например, первыми 1000 примерами
    dataset = dataset.select(range(1000))

    # Создаем DataLoader с batch_size=1 (для упрощения обработки)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Загружаем модель, токенизатор и обучаемые prompt-токены
    model, tokenizer, prompt_tuning_tokens = load_model_and_tokenizer(model_name="gpt2", num_prompt_tokens=5)

    # Определяем оптимизатор для обновления prompt-токенов
    optimizer = torch.optim.Adam([prompt_tuning_tokens], lr=0.001)

    # Обучение на нескольких эпохах (для демонстрации – 3 эпохи)
    epochs = 3
    for epoch in range(epochs):
        avg_loss = train_epoch(model, tokenizer, prompt_tuning_tokens, dataloader, optimizer)
        print(f"Epoch {epoch + 1}/{epochs} — Average Loss: {avg_loss:.4f}")

    # Тестовая генерация перевода для нового примера
    test_source = "Translate this to French: I love machine learning"
    translation = generate_translation(model, tokenizer, prompt_tuning_tokens, test_source, max_gen_length=20)
    print("\nTest Input:", test_source)
    print("Generated Translation:", translation)


if __name__ == "__main__":
    main()
