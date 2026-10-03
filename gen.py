"""gen.py — интерактивная генерация с обученной моделью."""
import torch
from main import Config, MiniTransformer, CharTokenizer

cfg = Config()
tokenizer = CharTokenizer(cfg.alphabet)
cfg.vocab_size = tokenizer.vocab_size

model = MiniTransformer(cfg).to(cfg.device)
model.load_state_dict(torch.load("wiki_rtx4050_gpt.pt", map_location=cfg.device))
model.eval()

print("Модель загружена. Пустой ввод — выход.")
while True:
    prompt = input("\nПромпт: ").strip()
    if not prompt:
        break
    ids = tokenizer.encode(prompt)
    if not ids:
        print("Промпт пуст после фильтрации алфавита.")
        continue
    x = torch.tensor([ids], dtype=torch.long, device=cfg.device)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = model.generate(x, max_new_tokens=400,
                             temperature=cfg.temperature, top_k=cfg.top_k)
    print(tokenizer.decode(out[0].tolist()))