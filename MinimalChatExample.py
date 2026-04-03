"""

---

# InstructVLA 部署异常与修复复盘报告

## 1. 异常现象与核心痛点
* **初始现象**：在调用模型进行对话生成时，终端抛出致命错误：`RuntimeError: The size of tensor a (576) must match the size of tensor b (288) at non-singleton dimension 3`。
* **痛点**：该报错发生在最底层的 PyTorch 算子中，且仅在生成阶段（`generate`）触发，常规的外层参数调整无法触及。

## 2. 根因剖析（三连暴击）
这个 Bug 是由硬件限制、HuggingFace 框架缺陷与第三方 PEFT 插件漏洞共同引发的“连环车祸”：

1. **硬件降级触发了地雷（V100 vs A100）**：
   因为服务器使用的是 V100（Volta 架构），不支持最新的 Flash Attention 2。系统被迫降级使用了最原始的 `eager_attention`。在原始注意力机制中，系统必须显式生成一个 `[288, 288]` 的因果掩码（Causal Mask）去进行矩阵相加，这为后续的维度碰撞埋下了伏笔。
2. **X-LoRA 插件的“幽灵缓存泄漏”**：
   项目中使用的动态路由插件 X-LoRA 存在底层设计缺陷。它在正式生成文本前，为了计算路由权重，会在后台偷偷跑一次前向传播。这次传播遗留了 288 个 Token 的 KV Cache，并且 X-LoRA 强行将这批“幽灵缓存”打包传递给了主模型的生成进程。
3. **Transformers 的切片失效**：
   主模型同时接收到了 288 长度的新图像特征和 288 长度的幽灵缓存，导致序列长度倍增至 576。此时，系统试图将 `[288, 288]` 的掩码硬塞给 `[288, 576]` 的注意力权重矩阵，物理碰撞发生，程序崩溃。

## 3. 攻坚过程（历经三次迭代）

* **第一轮防御（外层规避 - 失败）**：
  * **操作**：在 `generate` 函数中传入 `use_cache=False`。
  * **结果**：被击穿。因为 X-LoRA 的劫持级别极高，它无视了用户的外层指令，依然把缓存塞进了底层传参（`kwargs`）中。

* **第二轮防御（参数准备层拦截 - 失败）**：
  * **操作**：重写了语言模型的 `prepare_inputs_for_generation`，试图在组装参数时剔除缓存。
  * **结果**：再次被绕过。日志表明，X-LoRA 的代码逻辑甚至跳过了 HuggingFace 的标准参数准备阶段，直接将带有毒性缓存的参数砸向了模型前向传播。

* **第三轮防御（算子级物理断点 - 成功）**：
  * **操作**：实施终极 Monkey Patch，直接下沉到底层 `Qwen2Attention.forward` 注意力算子。
  * **逻辑**：我们在算子内部安插了“安检口”。每当算子接收到张量时，先检查长度。如果是推理的第一步（Prompt 输入阶段，长度 > 1），就无情地将传入的 `past_key_value` 内存指针抹除为 `None`。
  * **结果**：完美阻断。不论上层框架如何混乱、如何强塞参数，最终的物理计算入口被我们锁死了，576 维度的错误张量被物理消灭。

## 4. 后续建议与隐患排查
既然现在链路已经打通，如果你发现模型总是输出感叹号或乱码，请排查以下两点
1. **测试数据**：确保你传入的 `./text_image.png` 是一张包含实际内容的真实图片，而不是纯白/纯黑的占位图。
2. **精度敏感**：目前为了适配 V100 强制使用了 `float16`。如果模型原先是用 `bfloat16` 训练的，在 `float16` 下极易发生数值溢出（导致疯狂重复同一个标点符号）。如果更换图片后依然输出感叹号，建议尝试在加载模型时加入 `.half()` 并在生成参数中增加 `repetition_penalty=1.2` 等惩罚项。"""

import torch
import numpy as np
from PIL import Image

# ================= 终极物理断点：Qwen2 底层注意力算子拦截 =================
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

original_qwen2_forward = Qwen2Attention.forward

def safe_qwen2_attn_forward(self, hidden_states, *args, **kwargs):
    # 提取当前处理的 Token 序列长度
    q_len = hidden_states.shape[1]
    
    # 核心制裁：处于首步预填充阶段（Prompt长度 > 1）时，强行切断任何传入的 KV Cache
    if q_len > 1:
        # 1. 拦截关键字传参中的 cache
        if 'past_key_value' in kwargs:
            kwargs['past_key_value'] = None
        
        # 2. 拦截位置传参中的 cache (Qwen2Attention forward参数顺序: attention_mask, position_ids, past_key_value)
        new_args = list(args)
        if len(new_args) >= 3 and new_args[2] is not None:
            new_args[2] = None
        args = tuple(new_args)

    return original_qwen2_forward(self, hidden_states, *args, **kwargs)

# 挂载算子级拦截器
Qwen2Attention.forward = safe_qwen2_attn_forward
# ======================================================================

from vla.instructvla_eagle_dual_sys_v2_meta_query_v2 import load, load_vla

model_path = 'outputs/release_ckpts/instructvla_finetune_v2_xlora_freeze_head_instruction/checkpoints/step-013500-epoch-01-loss=0.1093.pt'

# 维持 float16 适应 V100 架构
model = load_vla(model_path, stage="stage2").eval().to(torch.float16).cuda()

messages = [
    {"content": "You are a helpful assistant."},  # system
    {
        "role": "user",
        "content": "Can you describe the main idea of this image?",
        "image": [{'np_array': np.asarray(Image.open("./text_image.png"))}]
    }
]

# Preprocess input
inputs = model.processor.prepare_input(dict(prompt=messages))
autocast_dtype = torch.float16

with torch.autocast("cuda", dtype=autocast_dtype, enabled=True):
    output = model.vlm.generate(
        input_ids=inputs['input_ids'].cuda(),
        attention_mask=inputs['attention_mask'].cuda(),
        pixel_values=inputs['pixel_values'].cuda(),
        max_new_tokens=200,
        output_hidden_states=False,
    )

response = model.processor.tokenizer.decode(output[0])
print(response)

