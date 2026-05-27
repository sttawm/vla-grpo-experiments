"""
Five prompt strategies for eliciting step-1 robot commands from Qwen.

Design goals for GRPO:
  1. High lexical variance across K=8 samples (reward spread requires diverse outputs)
  2. Bridge-aligned style — short imperative commands OpenVLA was trained on
  3. Object- and direction-specific (gives OpenVLA something to act on)

Each variant is a function: (goal: str, frames: list[np.ndarray]) -> list[dict]
returning a Qwen chat message list.
"""

from PIL import Image
import numpy as np

N_PLAN_STEPS = 4

# Action verbs present in Bridge training captions (used to score alignment)
BRIDGE_VERBS = {
    "pick", "place", "put", "grasp", "grab", "move", "push", "pull",
    "lift", "lower", "slide", "rotate", "turn", "open", "close",
    "reach", "press", "drop", "insert", "remove", "carry", "bring",
}


def _image_content(frames):
    return [{"type": "image", "image": Image.fromarray(f)} for f in frames]


# ── A: current baseline ───────────────────────────────────────────────────────
# Concise + strict format. Suppresses variance — step 1s cluster tightly.
def messages_A(goal: str, frames: list) -> list[dict]:
    content = _image_content(frames) + [{
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
    }]
    return [{"role": "user", "content": content}]


# ── B: Bridge-verb focused ────────────────────────────────────────────────────
# Anchors step 1 to the verb vocabulary OpenVLA understands.
# No "concise" constraint — lets the model vary object + spatial detail.
def messages_B(goal: str, frames: list) -> list[dict]:
    content = _image_content(frames) + [{
        "type": "text",
        "text": (
            f"You are controlling a robot arm.\n"
            f"Goal: {goal}\n\n"
            f"The {len(frames)} image(s) show the scene (oldest → newest).\n\n"
            f"Write a 4-step plan. Every step must start with a robot action verb: "
            f"pick, place, grasp, move, push, pull, lift, slide, open, close, etc.\n\n"
            f"Step 1 is most important — name the exact object and where the arm "
            f"should move (direction or target location).\n\n"
            f"1. [verb + object + where]\n"
            f"2. [action]\n"
            f"3. [action]\n"
            f"4. [action]"
        ),
    }]
    return [{"role": "user", "content": content}]


# ── C: step-1 command only ────────────────────────────────────────────────────
# Drops steps 2-4 entirely — forces all generation capacity onto step 1.
# Shorter context → more temperature-driven variance.
def messages_C(goal: str, frames: list) -> list[dict]:
    content = _image_content(frames) + [{
        "type": "text",
        "text": (
            f"You are controlling a robot arm.\n"
            f"Goal: {goal}\n\n"
            f"The {len(frames)} image(s) show the current scene (oldest → newest).\n\n"
            f"Write ONE robot command (4–8 words) for what the arm should do "
            f"right now. Start with an action verb. Name the specific object. "
            f"Include direction or position.\n\n"
            f"Command:"
        ),
    }]
    return [{"role": "user", "content": content}]


# ── D: few-shot Bridge-style examples ─────────────────────────────────────────
# Shows the model the exact register OpenVLA was trained on.
# Examples are generic enough not to anchor on a specific object.
def messages_D(goal: str, frames: list) -> list[dict]:
    content = _image_content(frames) + [{
        "type": "text",
        "text": (
            f"You are controlling a robot arm.\n"
            f"Goal: {goal}\n\n"
            f"The {len(frames)} image(s) show the current scene (oldest → newest).\n\n"
            f"Write a 4-step plan. Step 1 must follow this style — short, "
            f"specific, action-verb first:\n"
            f'  "grasp the red bowl from the left"\n'
            f'  "move arm down toward the blue block"\n'
            f'  "pick up the yellow cup and lift"\n'
            f'  "slide the pot toward the right burner"\n'
            f'  "lower gripper onto the green lid"\n\n'
            f"1. [step 1 in that style]\n"
            f"2. [action]\n"
            f"3. [action]\n"
            f"4. [action]"
        ),
    }]
    return [{"role": "user", "content": content}]


# ── E: explicit spatial detail + diversity nudge ──────────────────────────────
# Asks for precise spatial language and explicitly discourages generic phrasing.
# Higher expected variance but may produce longer step-1 strings.
def messages_E(goal: str, frames: list) -> list[dict]:
    content = _image_content(frames) + [{
        "type": "text",
        "text": (
            f"You are controlling a robot arm.\n"
            f"Goal: {goal}\n\n"
            f"The {len(frames)} image(s) show the current scene (oldest → newest).\n\n"
            f"Write a 4-step plan. For step 1, be precise: include the object name, "
            f"the direction of arm movement (left/right/up/down/forward/back), and "
            f"the gripper state (open, close, or grasp). Avoid vague phrases like "
            f"'move to the object' — describe exactly what you see.\n\n"
            f"1. [precise spatial command]\n"
            f"2. [action]\n"
            f"3. [action]\n"
            f"4. [action]"
        ),
    }]
    return [{"role": "user", "content": content}]


VARIANTS = {
    "A_baseline":     messages_A,
    "B_bridge_verbs": messages_B,
    "C_step1_only":   messages_C,
    "D_fewshot":      messages_D,
    "E_spatial":      messages_E,
}
