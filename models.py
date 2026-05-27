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
VLM3_MODEL_ID = "openvla/openvla-7b-oft"
UNNORM_KEY    = "bridge_orig"
DEVICE        = "cuda"
DTYPE         = torch.bfloat16

N_HISTORY     = 4   # number of past frames Qwen sees
N_PLAN_STEPS  = 4   # Qwen outputs a plan of this many steps

# ── VLM₂: Qwen planner ────────────────────────────────────────────────────────

_qwen_model     = None
_qwen_processor = None


def load_vlm2():
    global _qwen_model, _qwen_processor
    if _qwen_model is not None:
        return
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    print(f"Loading VLM₂: {VLM2_MODEL_ID}")
    _qwen_processor = AutoProcessor.from_pretrained(VLM2_MODEL_ID)
    _qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        VLM2_MODEL_ID,
        torch_dtype=DTYPE,
        device_map=DEVICE,
    )
    _qwen_model.eval()
    print("VLM₂ ready.")


def _build_qwen_messages(goal: str, frames: list[np.ndarray]) -> list[dict]:
    """
    Build the Qwen chat message with interleaved images (last N frames) and text.
    frames: list of numpy RGB arrays, ordered oldest → newest.
    """
    image_content = [
        {"type": "image", "image": Image.fromarray(f)} for f in frames
    ]
    text_content = {
        "type": "text",
        "text": (
            f"You are controlling a robot arm.\n"
            f"Goal: {goal}\n\n"
            f"The {len(frames)} image(s) above show the robot's recent state "
            f"(oldest → newest).\n\n"
            f"Output a {N_PLAN_STEPS}-step plan for what the robot should do next. "
            f"Be concise (5–10 words per step). Format exactly:\n"
            f"1. [action]\n2. [action]\n3. [action]\n4. [action]"
        ),
    }
    return [{"role": "user", "content": image_content + [text_content]}]


def plan_vlm2(goal: str, frame_history: list[np.ndarray],
              do_sample: bool = False, temperature: float = 1.0) -> tuple[str, str]:
    """
    Generate a 4-step plan from Qwen and return (full_plan_text, step_1_label).
    frame_history: list of up to N_HISTORY frames (numpy uint8 RGB).
    """
    load_vlm2()
    frames = frame_history[-N_HISTORY:]
    messages = _build_qwen_messages(goal, frames)

    text_input = _qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs = [Image.fromarray(f) for f in frames]

    inputs = _qwen_processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)

    gen_kwargs = dict(max_new_tokens=200, do_sample=do_sample)
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        out = _qwen_model.generate(**inputs, **gen_kwargs)

    generated = out[0][inputs["input_ids"].shape[1]:]
    full_text = _qwen_processor.decode(generated, skip_special_tokens=True).strip()
    step1 = _parse_step1(full_text)
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


def predict_vlm3(label: str, frame: np.ndarray) -> np.ndarray:
    """
    Given a mid-level label and the current frame, return a 7-DoF action vector.
    Uses greedy decoding (do_sample=False) for a deterministic reward signal.
    """
    load_vlm3()
    prompt = f"In: What action should the robot take to {label}?\nOut:"
    image  = Image.fromarray(frame)
    inputs = _openvla_processor(prompt, image).to(DEVICE, dtype=DTYPE)
    with torch.no_grad():
        action = _openvla_model.predict_action(
            **inputs, unnorm_key=UNNORM_KEY, do_sample=False
        )
    return np.array(action, dtype=np.float32)


# ── Reward ─────────────────────────────────────────────────────────────────────

def compute_reward(label: str, frame: np.ndarray, gt_action: np.ndarray) -> float:
    """Negative L₂ distance between VLM₃'s prediction and the GT action."""
    pred = predict_vlm3(label, frame)
    return -float(np.linalg.norm(pred - gt_action))
