"""InstructVLA 部署侧推理工具。

职责:
1. 加载 InstructVLA checkpoint 并切换到评测模式
2. 统一图像中心裁剪/缩放逻辑
3. 提供动作 chunk 缓存与自适应 ensemble
"""

import json
import os
import time
from collections import deque
from vla.instructvla_eagle_dual_sys_v2_meta_query_v2_libero_wrist import load_vla
from typing import Callable, Dict, List, Optional, Type, Union, Tuple, Any, Sequence
import numpy as np
import tensorflow as tf
import torch
from PIL import Image

class AdaptiveEnsembler:
    def __init__(self, pred_action_horizon, adaptive_ensemble_alpha=0.0):
        # pred_action_horizon: 保留最近多少步动作用于加权融合。
        self.pred_action_horizon = pred_action_horizon
        self.action_history = deque(maxlen=self.pred_action_horizon)
        # alpha 越大，越偏向与当前动作方向一致的历史预测。
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha

    def reset(self):
        # 每个新任务/新 episode 开始前清空历史。
        self.action_history.clear()

    def ensemble_action(self, cur_action):
        # 追加当前动作到历史缓存。
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            # 单步动作: 直接堆叠历史向量。
            curr_act_preds = np.stack(self.action_history)
        else:
            # 动作块场景: 取不同时间偏移对应的动作进行对齐融合。
            curr_act_preds = np.stack(
                [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history)]
            )

        # 计算“当前动作”和历史动作的余弦相似度。
        ref = curr_act_preds[num_actions-1, :]
        previous_pred = curr_act_preds
        dot_product = np.sum(previous_pred * ref, axis=1)  
        norm_previous_pred = np.linalg.norm(previous_pred, axis=1)  
        norm_ref = np.linalg.norm(ref)  
        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)

        # 将相似度映射为权重并归一化。
        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum()
  
        # 按权重求和，得到当前时刻最终动作。
        cur_action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return cur_action

def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # 兼容单图与批量图输入，统一到 4D。
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # 根据面积比例计算裁剪后的高宽比例。
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # 构造中心裁剪框 [y1, x1, y2, x2]。
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # 先中心裁剪，再缩放回 224x224。
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # 若输入是单张图，恢复为 3D。
    if expanded_dims:
        image = image[0]

    return image

def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Center crop an image to match training data distribution.

    Args:
        image: Input image (PIL or numpy array)

    Returns:
        Image.Image: Cropped PIL Image
    """
    batch_size = 1
    crop_scale = 0.9

    # 统一转为 TensorFlow Tensor，便于复用上面的 crop_and_resize。
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))

    orig_dtype = image.dtype

    # 转到 [0,1] 浮点域做几何处理。
    image = tf.image.convert_image_dtype(image, tf.float32)

    # 执行中心裁剪与缩放。
    image = crop_and_resize(image, crop_scale, batch_size)

    # 转回原始 dtype，避免后续类型不匹配。
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # 最终输出统一为 PIL RGB。
    return Image.fromarray(image.numpy()).convert("RGB")

class InstructVLAServer:
    def __init__(
        self,
        cfg, 
    ) -> None:
        # cfg 通常来自评测脚本的 dataclass 配置。
        self.cfg = cfg
        self.action_ensemble = cfg.action_ensemble
        self.adaptive_ensemble_alpha = cfg.adaptive_ensemble_alpha
        self.action_ensemble_horizon = cfg.action_ensemble_horizon
        self.horizon = cfg.horizon

        self.task_description = None

        # 是否开启动作级集成（对时序抖动更鲁棒）。
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        
        # 通过统一入口加载 VLA（含 norm_stats / stage 等信息）。
        self.vla = load_vla(
            cfg.pretrained_checkpoint,
            load_for_training=False, 
            future_action_window_size=cfg.future_action_window_size,
            past_action_window_size=cfg.horizon,
            action_dim=cfg.action_dim,
        )
        # if self.cfg.use_bf16:
        #     # 仅把语言视觉主干切到 bf16，兼顾显存与吞吐。
        #     self.vla.vlm = self.vla.vlm.to(torch.bfloat16)
        if self.cfg.use_bf16:
            self.vla.vlm = self.vla.vlm.to(torch.bfloat16)
        else:
            self.vla.vlm = self.vla.vlm.to(torch.float16)

    


        self.vla = self.vla.to("cuda").eval()
        # global_step 用于按 use_length 复用动作块。
        self.global_step = 0
        self.last_action_chunk = None



    def reset(self, task_description: str) -> None:
        # 任务切换时重置状态，避免跨任务污染。
        self.task_description = task_description
        if self.action_ensemble:
            self.action_ensembler.reset()

        self.global_step = 0
        self.last_action_chunk = None

    def crop_and_resize(self, image, crop_scale, batch_size):
        """
        Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
        to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
        distribution shift at test time.

        Args:
            image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
                values between [0,1].
            crop_scale: The area of the center crop with respect to the original image.
            batch_size: Batch size.
        """
        # 兼容单图与批量图输入，统一到 4D。
        assert image.shape.ndims == 3 or image.shape.ndims == 4
        expanded_dims = False
        if image.shape.ndims == 3:
            image = tf.expand_dims(image, axis=0)
            expanded_dims = True

        # 根据面积比例计算裁剪后的高宽比例。
        new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
        new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

        # 构造中心裁剪框 [y1, x1, y2, x2]。
        height_offsets = (1 - new_heights) / 2
        width_offsets = (1 - new_widths) / 2
        bounding_boxes = tf.stack(
            [
                height_offsets,
                width_offsets,
                height_offsets + new_heights,
                width_offsets + new_widths,
            ],
            axis=1,
        )

        # 先中心裁剪，再缩放回 224x224。
        image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

        # 若输入是单张图，恢复为 3D。
        if expanded_dims:
            image = image[0]

        return image


    def get_vla_action(self, vla, cfg, base_vla_name, obs, task_label, unnorm_key, center_crop=True):
        """执行一次 InstructVLA 动作推理。

        调用流程:
        1) 取主视角 + wrist 图像
        2) 按训练分布做中心裁剪（可选）
        3) 调用 vla.predict_action 输出动作块
        4) 根据 use_length 选择当前步动作，或做 ensemble
        """
        # 拼接多视角输入，主图像放在首位。
        all_images = [obs["full_image"]]
        all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

        # 若训练用了 image_aug，评测通常也应开启 center_crop 以减小分布偏移。
        processed_images = []
        for image in all_images:
            pil_image = Image.fromarray(image).convert("RGB")

            if center_crop:
                pil_image = center_crop_image(pil_image)
            
            processed_images.append(pil_image)
            

        if task_label is not None:
            if task_label.lower() != self.task_description:
                # 指令变化时重置状态机。
                self.reset(task_label.lower())

        prompt_mode = getattr(cfg, "prompt_mode", "default")
        if self.cfg.use_length == -1 or self.global_step % self.cfg.use_length == 0:
            # 到达动作块刷新时机，重新跑一次完整推理。

            action, normalized_actions, cognition_features_current = vla.predict_action(image=processed_images, 
                                                                            instruction=self.task_description,
                                                                            unnorm_key=unnorm_key,
                                                                            do_sample=False,
                                                                            prompt_mode=prompt_mode,
                                                                            )
            self.last_action_chunk = action
        
        if self.cfg.use_length > 0:
            # 固定窗口复用动作块中的第 t 个动作。
            action = self.last_action_chunk[self.global_step % self.cfg.use_length]
        elif self.cfg.use_length == -1: # do ensemble
            # -1 代表直接对历史动作做自适应融合。
            action = self.action_ensembler.ensemble_action(action)

        # 步计数递增，供下一步选择动作块索引。
        self.global_step+=1

        return action


