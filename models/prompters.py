import torch
import torch.nn as nn
import numpy as np


import torch
from torch import nn
from modified_clip import clip

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from modified_clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


_tokenizer = _Tokenizer()

#############

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
#############3




class NullPrompter(nn.Module):
    def __init__(self):
        super(NullPrompter, self).__init__()
        pass

    def forward(self, x):
        return x

class NonePrompter(nn.Module):
    def __init__(self):
        super(NonePrompter, self).__init__()
        pass

    def forward(self):
        return None

class PadPrompter(nn.Module):
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size

        self.base_size = image_size - pad_size*2
        self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.base_size).cuda()
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])

        return x + prompt

class TokenPrompter(nn.Module):
    def __init__(self, prompt_len) -> None:
        super(TokenPrompter, self).__init__()

        self.prompt = nn.Parameter(torch.randn([1, prompt_len, 768]))
    
    def forward(self):
        return self.prompt


class TokenPrompter_w_pos(nn.Module):
    def __init__(self, prompt_len) -> None:
        super(TokenPrompter_w_pos, self).__init__()

        self.prompt = nn.Parameter(torch.randn([1, prompt_len, 768]))
        self.pos_embedding = nn.Parameter(torch.randn(1, prompt_len, 1))

    def forward(self):
        return self.prompt + self.pos_embedding


class TokenPrompter_w_pos_TransformerGEN(nn.Module):
    def __init__(self, prompt_len) -> None:
        super(TokenPrompter_w_pos_TransformerGEN, self).__init__()

        self.prompt = nn.Parameter(torch.randn([1, prompt_len, 768]))

        self.dropout = nn.Dropout(0)
        self.transformer = Transformer(768, 3, 4, 768, 768)

        self.pos_embedding = nn.Parameter(torch.randn(1, prompt_len, 1))

    def forward(self):
        return self.transformer(self.prompt + self.pos_embedding)

class FixedPatchPrompter(nn.Module):
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, :self.psize, :self.psize] = self.patch

        return x + prompt


class RandomPatchPrompter(nn.Module):
    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        x_ = np.random.choice(self.isize - self.psize)
        y_ = np.random.choice(self.isize - self.psize)

        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, x_:x_ + self.psize, y_:y_ + self.psize] = self.patch

        return x + prompt


"""
cfg.TRAINER.COOP.CSC = False
cfg.TRAINER.COOP.CTX_INIT = ""
cfg.TRAINER.COOP.N_CTX = args.ctx
fg.INPUT.SIZE = (224, 224)
cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = 'end'
"""

class PromptLearner(nn.Module):
    # def __init__(self, cfg, classnames, clip_model):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = args.ctx
        ctx_init = args.ctx_init
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(clip_model, torch.nn.DataParallel):
            dtype = clip_model.module.dtype
            ctx_dim = clip_model.module.ln_final.weight.shape[0]
            clip_imsize = clip_model.module.visual.input_resolution
        else:
            dtype = clip_model.dtype
            ctx_dim = clip_model.ln_final.weight.shape[0]
            clip_imsize = clip_model.visual.input_resolution
        input_size = (224,224)
        cfg_imsize = input_size[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))                            # n_ctx --- Context number (words)
            prompt = clip.tokenize(ctx_init).to(device)                 # prompt --- 77-dim vector to represent the whole sentence [1,77] only valid for [CLS] [1 to 1+n_ctx] [SEP]
            with torch.no_grad():
                if isinstance(clip_model, torch.nn.DataParallel):
                    embedding = clip_model.module.token_embedding(prompt).type(dtype)   # embedding --- [1, 77, 512]
                else:
                    embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors).to(device)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]          # Here the tokenized result is number
        # prompts = [prompt_prefix + " " + name + "." for name in classnames]
        prompts = [prompt_prefix + " " + name for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p).to(device) for p in prompts])
        with torch.no_grad():
            if isinstance(clip_model, torch.nn.DataParallel):
                embedding = clip_model.module.token_embedding(tokenized_prompts).type(dtype)
            else:
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS -- Start of sequence
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        # The above two token embeddings should be fixed

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = args.position

    def forward(self):
        ctx = self.ctx          # The parameters for optimization
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":      # prompts [1000, 77, 512] (prompt embedding)
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts



def padding(args):
    return PadPrompter(args)


def fixed_patch(args):
    return FixedPatchPrompter(args)


def random_patch(args):
    return RandomPatchPrompter(args)

def null_patch(args):
    return NullPrompter()
