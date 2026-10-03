"""Мини-GPT на PyTorch: обучение char-level модели на русской Википедии.

Оптимизировано под RTX 4050 (6GB VRAM):
- mixed precision (float16) через autocast + GradScaler
- streaming-датасет, не грузим Википедию в память целиком

Примечание: torch.compile не используем — он пока не работает на Python 3.14+.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm


# ==========================================
# Конфигурация
# ==========================================
class Config:
    # Гиперпараметры модели
    vocab_size: int = None      # заполняется после построения алфавита
    d_model: int = 512          # размер внутренних векторов
    n_layers: int = 8           # глубина модели
    block_size: int = 192       # длина контекста (сколько символов модель «помнит»)
    dropout: float = 0.1        # регуляризация

    # Обучение
    batch_size: int = 48
    learning_rate: float = 6e-4
    warmup_steps: int = 100     # постепенный разгон LR в начале
    min_lr_frac: float = 0.1    # LR в конце = learning_rate * min_lr_frac
    grad_clip: float = 1.0      # клиппинг градиентов против всплесков loss
    max_steps: int = 100_000
    eval_every: int = 300       # как часто печатать loss и пример генерации
    gen_length: int = 200       # длина примера генерации

    # Генерация (сэмплирование)
    temperature: float = 0.8    # <1.0 — смелее и разнообразнее, >1.0 — осторожнее
    top_k: int = 40             # рассматриваем только 40 самых вероятных символов

    seed: int = 42
    checkpoint_path: str = "wiki_rtx4050_gpt.pt"
    device: str = "cuda"

    alphabet: str = (
        " abcdefghijklmnopqrstuvwxyz"
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        "0123456789.,!?-:;()\"\'\n"
    )


# ==========================================
# 1. Архитектура
# ==========================================
class CausalSelfAttention(nn.Module):
    """Одноголовое причинное внимание: токен видит только предыдущие."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # (B, T, C) @ (B, C, T) -> (B, T, T)
        attn = torch.bmm(q, k.transpose(1, 2)) / (C ** 0.5)

        # маска: запрещаем смотреть «в будущее»
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        attn = attn.masked_fill(~mask, float("-inf"))

        attn = self.dropout(F.softmax(attn, dim=-1))
        return self.out_proj(torch.bmm(attn, v))


class Block(nn.Module):
    """Трансформерный блок: attention + FFN с residual connections и LayerNorm."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniTransformer(nn.Module):
    """Минимальный GPT: эмбеддинги -> блоки -> языковая голова."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            Block(cfg.d_model, cfg.dropout) for _ in range(cfg.n_layers)
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"Длина {T} > block_size={self.cfg.block_size}"

        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None
                 ) -> torch.Tensor:
        """Сэмплированная генерация: idx (B, T) -> (B, T + max_new_tokens).

        temperature < 1.0 — разнообразнее, > 1.0 — консервативнее.
        top_k ограничивает выбор k самых вероятных символов.
        """
        self.eval()
        for _ in range(max_new_tokens):
            cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                threshold = torch.topk(logits, k).values[:, [-1]]
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=-1)
        self.train()
        return idx


# ==========================================
# 2. Токенизатор (символьный уровень)
# ==========================================
class CharTokenizer:
    def __init__(self, alphabet: str):
        vocab = sorted(set(alphabet))
        self.vocab_size = len(vocab)
        self.char_to_id = {ch: i for i, ch in enumerate(vocab)}
        self.id_to_char = {i: ch for i, ch in enumerate(vocab)}

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[c] for c in text.lower() if c in self.char_to_id]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char[i] for i in ids)


# ==========================================
# 3. Обучение
# ==========================================
def get_lr(step: int, cfg: Config) -> float:
    """LR с warmup в начале и косинусным затуханием к концу обучения."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (
        1.0 + math.cos(math.pi * min(progress, 1.0))
    )
    return cfg.learning_rate * coeff


def train(cfg: Config):
    torch.manual_seed(cfg.seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Видеокарта NVIDIA (CUDA) не найдена! Проверьте драйверы."
        )
    print(f"Обучение стартует на: {torch.cuda.get_device_name(0)}")

    tokenizer = CharTokenizer(cfg.alphabet)
    cfg.vocab_size = tokenizer.vocab_size
    print(f"Размер словаря: {cfg.vocab_size}")

    print("Подключение к стримингу Википедии...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ru",
                           split="train", streaming=True)

    model = MiniTransformer(cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров модели: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    # Защищает от зануления маленьких градиентов в float16
    scaler = torch.amp.GradScaler("cuda")

    print("Начало обучения.")
    model.train()

    token_buffer: list[int] = []
    step = 0
    sample_prompt = tokenizer.encode("история россии — это ")

    for article in tqdm(dataset):
        token_buffer.extend(tokenizer.encode(article["text"]))

        while len(token_buffer) >= cfg.batch_size * cfg.block_size + 1:
            chunk = token_buffer[: cfg.batch_size * cfg.block_size + 1]
            token_buffer = token_buffer[cfg.batch_size * cfg.block_size:]

            xb = torch.tensor(chunk[:-1], dtype=torch.long) \
                   .view(cfg.batch_size, cfg.block_size).to(cfg.device, non_blocking=True)
            yb = torch.tensor(chunk[1:], dtype=torch.long) \
                   .view(cfg.batch_size, cfg.block_size).to(cfg.device, non_blocking=True)

            lr = get_lr(step, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, loss = model(xb, yb)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # клиппинг через скалер: относительно текущего масштаба
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            step += 1

            if step % cfg.eval_every == 0:
                print(f"\nШаг {step} | Loss: {loss.item():.4f} | LR: {lr:.2e}")
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    generated = model.generate(
                        torch.tensor([sample_prompt],
                                     dtype=torch.long, device=cfg.device),
                        max_new_tokens=cfg.gen_length,
                        temperature=cfg.temperature,
                        top_k=cfg.top_k,
                    )
                print(f"ИИ выдал: {tokenizer.decode(generated[0].tolist())}")

            if step >= cfg.max_steps:
                print("\nОбучение завершено!")
                torch.save(model.state_dict(), cfg.checkpoint_path)
                print(f"Веса сохранены в {cfg.checkpoint_path}")
                return


if __name__ == "__main__":
    train(Config())