from __future__ import annotations

import contextlib
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import Config, PROFILES, TOWERS
from .contracts import Call, ExecutionPlan


class ByteTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    vocab_size = 258

    def encode(self, text, **kwargs):
        return [b+2 for b in text.encode("utf-8")]

    def decode(self, ids, **kwargs):
        return bytes(i-2 for i in ids if 2 <= i < 258).decode("utf-8", errors="replace")


class Block(nn.Module):
    """Pre-normalized rotary causal attention; no learned context-size table."""
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width*3)
        self.proj = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, width*4), nn.GELU(), nn.Linear(width*4, width))

    def forward(self, x):
        b, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        d = width//self.heads
        def heads(t):
            return t.view(b, length, self.heads, d).transpose(1, 2)
        q, k, v = map(heads, (q, k, v))
        angles = torch.outer(torch.arange(length, device=x.device).float(),
                             1/(1000000**(torch.arange(0, d, 2, device=x.device).float()/d)))
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        def rotate(t):
            a, z = t[..., 0::2], t[..., 1::2]
            return torch.stack((a*cos-z*sin, a*sin+z*cos), -1).flatten(-2)
        attended = F.scaled_dot_product_attention(rotate(q), rotate(k), v, is_causal=True)
        x = x+self.proj(attended.transpose(1, 2).reshape(b, length, width))
        return x+self.ff(self.norm2(x))


class Tower(nn.Module):
    def __init__(self, vocab, profile, context):
        super().__init__()
        layers, width, heads = PROFILES[profile]
        self.width, self.context = width, context
        self.embedding = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.gradient_checkpointing = True
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None: nn.init.zeros_(module.bias)

    def hidden(self, ids, message=None, receiver=None):
        if ids.shape[1] > self.context:
            raise ValueError("tower context overflow; never truncate training targets")
        x = self.embedding(ids)
        if message is not None and receiver is None:
            x = x+message.mean(1, keepdim=True)
        receive_at = {len(self.blocks)//3, 2*len(self.blocks)//3}
        for index, block in enumerate(self.blocks):
            x = checkpoint(block, x, use_reentrant=False) if self.training and self.gradient_checkpointing else block(x)
            if message is not None and receiver is not None and index in receive_at:
                x = receiver(x, message)
        return self.norm(x)

    def logits(self, hidden):
        return F.linear(hidden, self.embedding.weight)

    def forward(self, ids):
        return self.logits(self.hidden(ids))


class Bridge(nn.Module):
    def __init__(self, source_width, target_width, tokens):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, tokens, target_width)*0.02)
        self.project = nn.Linear(source_width, target_width)
        self.attn = nn.MultiheadAttention(target_width, 4, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(target_width*2, target_width), nn.GELU(), nn.Linear(target_width, 1))
        nn.init.constant_(self.gate[-1].bias, -2)

    def forward(self, source, shuffle=False):
        memory = self.project(source)
        q = self.queries.expand(source.shape[0], -1, -1)
        msg, _ = self.attn(q, memory, memory, need_weights=False)
        context = memory.mean(1, keepdim=True).expand_as(msg)
        msg = msg*torch.sigmoid(self.gate(torch.cat((msg, context), -1)))
        return msg.roll(1, 0) if shuffle else msg


class Receiver(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.gate = nn.Linear(width*2, 1)
        self.output = nn.Linear(width, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden, message):
        context, _ = self.attn(hidden, message, message, need_weights=False)
        gate = torch.sigmoid(self.gate(torch.cat((hidden, context), -1)))
        return hidden+gate*self.output(context)


class Coordinator(nn.Module):
    def __init__(self, config, initialize=True):
        super().__init__()
        self.is_tiny = config.coordinator == "tiny"
        if self.is_tiny:
            self.base = Tower(258, "tiny", config.context)
            self.width = self.base.width
        else:
            from transformers import AutoModelForCausalLM, AutoConfig
            if initialize:
                self.base = AutoModelForCausalLM.from_pretrained(config.coordinator,
                    revision=config.revision, torch_dtype=torch.bfloat16)
                config.hf_config = self.base.config.to_dict()
            else:
                hf = dict(config.hf_config or {})
                kind = hf.pop("model_type", None)
                if not kind:
                    raise ValueError("bundle lacks coordinator architecture")
                self.base = AutoModelForCausalLM.from_config(AutoConfig.for_model(kind, **hf))
            self.width = self.base.config.hidden_size
            self.base.gradient_checkpointing_enable()
        self.base.requires_grad_(False)
        self.adapter = nn.Sequential(nn.Linear(self.width, 16, bias=False), nn.Linear(16, self.width, bias=False))
        nn.init.zeros_(self.adapter[-1].weight)
        self.receivers = nn.ModuleList(Receiver(self.width) for _ in range(3))

    def base_hidden(self, ids):
        self.base.eval()
        if self.is_tiny:
            return self.base.hidden(ids)
        return self.base.model(input_ids=ids, use_cache=False).last_hidden_state

    def stable_features(self, ids):
        with torch.no_grad():
            return self.base_hidden(ids).mean(1)

    def hidden(self, ids, messages=(), adapters=True):
        hidden = self.base_hidden(ids)
        for receiver, message in zip(self.receivers, messages[-3:]):
            hidden = receiver(hidden, message)
        return hidden+self.adapter(hidden) if adapters else hidden

    def logits(self, hidden):
        return self.base.logits(hidden) if self.is_tiny else self.base.lm_head(hidden)


class Dispatcher(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(width, 256), nn.GELU())
        self.wake_gates = nn.Linear(256, 12)
        self.round_head = nn.Linear(256, 48)
        self.dependency_head = nn.Linear(256, 144)
        self.halt_gate = nn.Linear(256, 1)

    def forward(self, features):
        h = self.encoder(features)
        return self.wake_gates(h), self.round_head(h).view(-1, 12, 4), self.halt_gate(h)

    def dependencies(self, features):
        return self.dependency_head(self.encoder(features)).view(-1, 12, 12)


class CFTN(nn.Module):
    def __init__(self, config: Config, tokenizer=None, initialize=True):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer or ByteTokenizer()
        self.coordinator = Coordinator(config, initialize)
        vocab = self.tokenizer.vocab_size if isinstance(self.tokenizer, ByteTokenizer) else len(self.tokenizer)
        self.towers = nn.ModuleDict({name: Tower(258 if name == "string" else vocab,
            "small" if name == "string" and config.profile != "tiny" else config.profile,
            config.long_context if name == "long_context" else config.context) for name in TOWERS})
        for name,spec in config.specialist_specs.items():
            if spec['kind']!='legacy_math_v12' or name!='math':raise ValueError('unsupported native specialist')
            from .local_specialist import LocalMathTower
            self.towers[name]=LocalMathTower(spec['spec'])
        cw = self.coordinator.width
        self.bridges = nn.ModuleDict({name: nn.ModuleDict({
            "request": Bridge(cw, tower.width, config.message_tokens),
            "return": Bridge(tower.width, cw, config.message_tokens),
            "receiver": Receiver(tower.width)}) for name, tower in self.towers.items()})
        self.dispatcher = Dispatcher(cw)
        self.last_execution_trace=[]

    def tokenizer_for(self, tower):
        if tower in self.config.specialist_specs:
            from .local_specialist import MathTokenizer
            return MathTokenizer()
        return ByteTokenizer() if tower == "string" else self.tokenizer

    def ids(self, text, tower=None):
        tokenizer = self.tokenizer_for(tower) if tower else self.tokenizer
        values = tokenizer.prefix(text) if hasattr(tokenizer,'prefix') else tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([values or [tokenizer.eos_token_id]], device=next(self.parameters()).device)

    def route(self, prompt):
        with torch.no_grad():
            wake, rounds, halt = self.dispatcher(self.coordinator.stable_features(self.ids(prompt)))
            probs = wake.sigmoid()[0]
            deps = self.dispatcher.dependencies(self.coordinator.stable_features(self.ids(prompt))).sigmoid()[0]
            selected = [i for i in range(12) if TOWERS[i] in self.config.active and probs[i] >= self.config.threshold]
            calls = [Call(TOWERS[i], int(rounds[0, i].argmax()), prompt,
                          tuple(TOWERS[j] for j in selected if j != i and deps[i,j] >= self.config.threshold)) for i in selected]
            confidence = min([float(probs[TOWERS.index(c.tower)]) for c in calls], default=1.0)
        return ExecutionPlan(calls, confidence).validate(self.config.active, threshold=self.config.threshold)

    def communicate(self, prompt, plan, disabled=(), shuffle=False):
        # Request and return messages never see teacher-forced target tokens.
        plan.validate(TOWERS)
        self.last_execution_trace=[]
        prompt_ids = self.ids(prompt)
        workspace = self.coordinator.hidden(prompt_ids, adapters=False)
        messages = []
        for round_id in range(4):
            returned = []
            for call in plan.calls:
                if call.round != round_id or call.tower in disabled:
                    continue
                name = call.tower
                self.last_execution_trace.append({'tower':name,'round':round_id,'request':call.request,
                    'depends_on':list(call.depends_on)})
                bridge, tower = self.bridges[name], self.towers[name]
                request = bridge['request'](workspace)
                native = tower.hidden(self.ids(call.request, name), message=request, receiver=bridge['receiver'])
                returned.append(bridge['return'](native, shuffle=shuffle))
            if returned:
                message = torch.cat(returned, 1)
                messages.append(message)
                workspace = self.coordinator.receivers[min(round_id, 2)](workspace, message)
        return messages

    @torch.no_grad()
    def generate(self, prompt, tower=None, max_tokens=128, plan=None, disabled=(), shuffle=False):
        self.eval()
        self.last_execution_trace=[]
        if tower in self.config.specialist_specs:
            self.last_execution_trace=[{'tower':tower,'round':0,'request':prompt.rstrip('\n'),'depends_on':[]}]
            return self.towers[tower].generate(prompt.rstrip('\n'),max_tokens)[0]
        tokenizer = self.tokenizer_for(tower) if tower else self.tokenizer
        ids = self.ids(prompt, tower)
        prefix = ids.shape[1]
        messages = self.communicate(prompt, plan, disabled, shuffle) if plan else []
        context = self.towers[tower].context if tower else self.config.context
        for _ in range(max_tokens):
            if ids.shape[1] >= context:
                break
            logits = self.towers[tower](ids) if tower else self.coordinator.logits(self.coordinator.hidden(ids, messages))
            token = logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat((ids, token), 1)
            if int(token) == tokenizer.eos_token_id:
                break
        return tokenizer.decode(ids[0, prefix:].tolist(), skip_special_tokens=True)
