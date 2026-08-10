import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)


def main():
    import sys

    sys.path.insert(0, PROJECT_ROOT)
    from rl.export_to_numpy import export_weights_to_numpy

    model_path = os.path.join(PROJECT_ROOT, "logs", "bc_full_model.pt")
    npz_path = os.path.join(PROJECT_ROOT, "logs", "bc_weights.npz")
    export_weights_to_numpy(model_path, npz_path)

    with open(npz_path, "rb") as f:
        b64_str = base64.b64encode(f.read()).decode("utf-8")

    with open(os.path.join(PROJECT_ROOT, "agent_torch.py")) as f:
        agent_code = f.read()

    with open(os.path.join(PROJECT_ROOT, "rl", "action_space.py")) as f:
        action_space_code = f.read()

    with open(os.path.join(PROJECT_ROOT, "rl", "numpy_inference.py")) as f:
        numpy_code = f.read()

    # Strip imports that we will redefine or don't need
    agent_code = agent_code.replace("import torch\n", "")

    local_imports_block = """# Ensure local imports work correctly for Kaggle environment
try:
    from rl.action_space import decode_market_actions, decode_unit_action
    from rl.architecture import KaggriculturePolicyFullRL
except ImportError:
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from rl.action_space import decode_market_actions, decode_unit_action
    from rl.architecture import KaggriculturePolicyFullRL
"""
    agent_code = agent_code.replace(local_imports_block, "")

    action_space_code = action_space_code.replace("import numpy as np\n", "")
    numpy_code = numpy_code.replace("import numpy as np\n", "")

    # Define new load_model block
    load_model_old = """_MODEL_PATH = os.path.join(os.path.dirname(__file__), "logs", "bc_full_model.pt")

_POLICY = None
_DEVICE = None


def _load_model():
    global _POLICY, _DEVICE
    if _POLICY is not None:
        return

    _DEVICE = torch.device("cpu")  # Inference on CPU is fine for Kaggle submissions usually
    _POLICY = KaggriculturePolicyFullRL()
    if os.path.exists(_MODEL_PATH):
        try:
            _POLICY.load_state_dict(torch.load(_MODEL_PATH, map_location=_DEVICE))
            _POLICY.eval()
        except Exception as e:
            print(f"Failed to load weights: {e}", file=sys.stderr)
    else:
        print(
            f"WARNING: No trained weights found at {_MODEL_PATH}. Using random init.",
            file=sys.stderr,
        )"""

    load_model_new = f"""
_BC_WEIGHTS_B64 = "{b64_str}"

_POLICY = None

def _load_model():
    global _POLICY
    if _POLICY is not None:
        return

    try:
        with np.load(io.BytesIO(base64.b64decode(_BC_WEIGHTS_B64))) as data:
            weights = {{k: data[k] for k in data.files}}
        _POLICY = NumpyPolicyFullRL(weights)
    except Exception as e:
        print(f"Failed to load numpy weights: {{e}}", file=sys.stderr)
"""

    agent_code = agent_code.replace(load_model_old, load_model_new)

    # Replace torch tensor conversions in agent()
    agent_code = agent_code.replace(
        "        t_spatial = torch.tensor(spatial_feat).unsqueeze(0).to(_DEVICE)\n", ""
    )
    agent_code = agent_code.replace(
        "        t_vector = torch.tensor(vector_feat).unsqueeze(0).to(_DEVICE)\n", ""
    )
    agent_code = agent_code.replace("        with torch.no_grad():\n", "")
    agent_code = agent_code.replace(
        "            f_logits, h_logits, m_preds = _POLICY(t_vector, t_spatial)  # type: ignore\n",
        "        f_logits, h_logits, m_preds = _POLICY(vector_feat, spatial_feat)\n",
    )

    agent_code = agent_code.replace(
        "        f_idx = torch.argmax(f_logits[0]).item()\n",
        "        f_idx = int(np.argmax(f_logits))\n",
    )

    agent_code = agent_code.replace(
        "        h_preds = torch.argmax(h_logits[0], dim=0).cpu().numpy()  # shape (10, 10)\n",
        "        h_preds = np.argmax(h_logits, axis=0)\n",
    )

    agent_code = agent_code.replace(
        "        m_tensor = m_preds[0].cpu().numpy()  # shape (12, 3)\n",
        "        m_tensor = m_preds\n",
    )

    # Inject action_space and numpy_inference
    final_code = agent_code.replace(
        "import numpy as np\n",
        f"import base64\nimport io\nimport numpy as np\n\n{action_space_code}\n{numpy_code}\n",
    )

    with open(os.path.join(PROJECT_ROOT, "main.py"), "w") as f:
        f.write(final_code)

    print("Successfully built main.py for Kaggle submission!")


if __name__ == "__main__":
    main()
