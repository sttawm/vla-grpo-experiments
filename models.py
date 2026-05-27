"""
VLM₂  — Qwen2.5-VL-7B-Instruct (planner, online, sees last N frames)
VLM₃  — OpenVLA-OFT (frozen controller, reward source)

Both are loaded once and reused across all calls.
"""

import re
import numpy as np
import torch
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────────

VLM2_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
VLM3_MODEL_ID = "openvla/openvla-7b"
UNNORM_KEY    = "bridge_orig"
DEVICE        = "cuda"
DTYPE         = torch.bfloat16

N_HISTORY     = 4   # number of past frames Qwen sees

# ── VLM₂: Qwen planner ────────────────────────────────────────────────────────

_qwen_model     = None
_qwen_processor = None


def load_vlm2():
    global _qwen_model, _qwen_processor
    if _qwen_model is not None:
        return
    from transformers import AutoProcessor
    # Qwen2.5-VL uses Qwen2_5_VLForConditionalGeneration (transformers >= 4.52)
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as QwenCls
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as QwenCls
    print(f"Loading VLM₂: {VLM2_MODEL_ID} with {QwenCls.__name__}")
    _qwen_processor = AutoProcessor.from_pretrained(VLM2_MODEL_ID)
    _qwen_model = QwenCls.from_pretrained(
        VLM2_MODEL_ID,
        torch_dtype=DTYPE,
        device_map=DEVICE,
    )
    _qwen_model.eval()
    print("VLM₂ ready.")


def _build_qwen_messages(goal: str, frames: list[np.ndarray], n_steps: int = 2) -> list[dict]:
    """
    Build the Qwen chat message with interleaved images (last N frames) and text.
    frames: list of numpy RGB arrays, ordered oldest → newest.
    n_steps: how many immediate actions to request (1, 2, or 4).
    """
    image_content = [
        {"type": "image", "image": Image.fromarray(f)} for f in frames
    ]
    if n_steps == 1:
        action_prompt = (
            f"What is the next immediate action for the robot arm? "
            f"Be more specific and lower-level than the goal — describe the "
            f"exact physical movement.\n\n"
            f"1. [action]"
        )
    else:
        steps_word = {2: "two", 4: "four"}.get(n_steps, str(n_steps))
        numbered = "\n".join(f"{i+1}. [action]" for i in range(n_steps))
        action_prompt = (
            f"What are the next {steps_word} immediate actions for the robot arm? "
            f"Be more specific and lower-level than the goal — describe the "
            f"exact physical movement.\n\n"
            f"{numbered}"
        )
    text_content = {
        "type": "text",
        "text": (
            f"You are controlling a robot arm. The high-level goal is: {goal}\n\n"
            f"The {len(frames)} image(s) above show the robot's recent state "
            f"(oldest → newest).\n\n"
            f"{action_prompt}"
        ),
    }
    return [{"role": "user", "content": image_content + [text_content]}]


def plan_vlm2(
    goal: str,
    frame_history: list[np.ndarray],
    n_steps: int = 2,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> tuple[str, str]:
    """
    Generate a plan from Qwen and return (full_plan_text, step_1_label).
    frame_history: list of up to N_HISTORY frames (numpy uint8 RGB).
    n_steps: number of steps to request from Qwen (only step 1 is used as label).
    """
    load_vlm2()
    frames   = list(frame_history[-N_HISTORY:])
    messages = _build_qwen_messages(goal, frames, n_steps=n_steps)

    text_input   = _qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs = [Image.fromarray(f) for f in frames]

    inputs = _qwen_processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)

    gen_kwargs = dict(max_new_tokens=60, do_sample=do_sample)
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        out = _qwen_model.generate(**inputs, **gen_kwargs)

    generated = out[0][inputs["input_ids"].shape[1]:]
    full_text  = _qwen_processor.decode(generated, skip_special_tokens=True).strip()
    step1      = _parse_step1(full_text)
    return full_text, step1


def _parse_step1(plan_text: str) -> str:
    """Extract the first step from a numbered plan."""
    m = re.search(r"1[.)]\s*(.+?)(?:\n|$)", plan_text)
    if m:
        return m.group(1).strip()
    # Fallback: first non-empty line
    for line in plan_text.splitlines():
        line = line.strip()
        if line:
            return line
    return plan_text.strip()


# ── VLM₃: OpenVLA-OFT frozen controller ───────────────────────────────────────

_openvla_model     = None
_openvla_processor = None


def load_vlm3():
    global _openvla_model, _openvla_processor
    if _openvla_model is not None:
        return

    # transformers>=4.50 added smart_apply which calls module._initialize_weights
    # on every child. OpenVLA's custom VisionTransformer doesn't define it.
    # Patch smart_apply to silently skip modules that lack _initialize_weights.
    from transformers.modeling_utils import PreTrainedModel
    # OpenVLA's VisionTransformer doesn't define _initialize_weights.
    # Patch initialize_weights (called during from_pretrained for missing keys)
    # to swallow the AttributeError rather than crashing.
    _orig_init_weights = PreTrainedModel.initialize_weights
    def _safe_init_weights(self, *args, **kwargs):
        try:
            return _orig_init_weights(self, *args, **kwargs)
        except AttributeError:
            pass
    PreTrainedModel.initialize_weights = _safe_init_weights

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print(f"Loading VLM₃: {VLM3_MODEL_ID}")
    _openvla_processor = AutoProcessor.from_pretrained(
        VLM3_MODEL_ID, trust_remote_code=True
    )
    _openvla_model = AutoModelForVision2Seq.from_pretrained(
        VLM3_MODEL_ID,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(DEVICE)
    _openvla_model.eval()
    # Freeze all parameters — never updated
    for p in _openvla_model.parameters():
        p.requires_grad_(False)
    print(f"VLM₃ ready. Action dim: {_openvla_model.get_action_dim(UNNORM_KEY)} DoF")


def predict_vlm3(goal: str, frame: np.ndarray, label: str = None) -> np.ndarray:
    """
    Return a 7-DoF action vector from OpenVLA.

    goal:  high-level task description (always provided; matches Bridge training dist.)
    label: mid-level action from Qwen (optional).
           - None  → VLA-only prompt:  "...take to {goal}?"
           - set   → combined prompt:  "...take to {label} in order to {goal}?"
    """
    load_vlm3()
    if label is not None:
        prompt = f"In: What action should the robot take to {label} in order to {goal}?\nOut:"
    else:
        prompt = f"In: What action should the robot take to {goal}?\nOut:"
    image  = Image.fromarray(frame)
    inputs = _openvla_processor(prompt, image).to(DEVICE, dtype=DTYPE)
    with torch.no_grad():
        action = _openvla_model.predict_action(
            **inputs, unnorm_key=UNNORM_KEY, do_sample=False
        )
    return np.array(action, dtype=np.float32)


# ── Reward ─────────────────────────────────────────────────────────────────────

def _action_std() -> np.ndarray:
    """Per-dimension std from Bridge dataset (used to normalize L2)."""
    load_vlm3()
    try:
        stats = _openvla_model.norm_stats["bridge_orig"]["action"]
        std = np.array(stats["std"], dtype=np.float32)
        # Guard against zero std (e.g. constant gripper dim in some splits)
        std = np.where(std < 1e-6, 1.0, std)
        return std
    except (KeyError, AttributeError):
        # Fallback: empirical Bridge bridge_orig stds (xyz ~0.02m, rpy ~0.08rad, gripper ~0.5)
        return np.array([0.02, 0.02, 0.02, 0.08, 0.08, 0.08, 0.5], dtype=np.float32)


def compute_reward(
    goal: str,
    frame: np.ndarray,
    gt_action: np.ndarray,
    label: str = None,
    normalized: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Returns (reward, pred_action, per_dim_l2).

    reward       — negative normalized L2 (each dim divided by its Bridge std)
    pred_action  — raw 7-DoF prediction from VLM₃  (shape: 7,)
    per_dim_l2   — |pred - gt| per dimension        (shape: 7,)

    label=None   → VLA-only mode (goal passed directly to OpenVLA)
    label=str    → VLA+mid-level mode (label + goal passed to OpenVLA)
    normalized=False falls back to raw (unnormalized) L2 for backward compat.
    """
    pred = predict_vlm3(goal, frame, label)
    per_dim = np.abs(pred - gt_action).astype(np.float32)
    if normalized:
        std = _action_std()
        reward = -float(np.linalg.norm(per_dim / std))
    else:
        reward = -float(np.linalg.norm(per_dim))
    return reward, pred, per_dim
