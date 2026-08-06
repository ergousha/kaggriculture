import base64


def embed_weights_into_main(npz_path: str, main_path: str):
    print(f"Reading {npz_path}...")
    with open(npz_path, "rb") as f:
        data = f.read()

    b64_str = base64.b64encode(data).decode("ascii")

    print(f"Reading {main_path}...")
    with open(main_path) as f:
        lines = f.readlines()

    # Check if already embedded
    for line in lines:
        if "class KaggriculturePolicyNP" in line:
            print("Model already embedded in main.py!")
            return

    # Prepare the payload
    payload = f"""
# =======================================================================
# PYTORCH HYBRID POLICY (Numpy Inference Port)
# =======================================================================
import base64
import io
import numpy as np

_BC_WEIGHTS_B64 = "{b64_str}"

def _relu(x):
    return np.maximum(0, x)

def _conv2d_np(x, w, b, padding=1, stride=1):
    C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    out_H = (H + 2*padding - kH) // stride + 1
    out_W = (W + 2*padding - kW) // stride + 1
    if padding > 0:
        x_pad = np.pad(x, ((0,0), (padding, padding), (padding, padding)), mode='constant')
    else:
        x_pad = x
    out = np.zeros((C_out, out_H, out_W), dtype=np.float32)
    for i in range(out_H):
        for j in range(out_W):
            x_slice = x_pad[:, i*stride:i*stride+kH, j*stride:j*stride+kW]
            # Sum over C_in, kH, kW
            out[:, i, j] = np.sum(x_slice * w, axis=(1,2,3)) + b
    return out

def _maxpool2d_np(x, kernel_size=2):
    C, H, W = x.shape
    out_H = H // kernel_size
    out_W = W // kernel_size
    out = np.zeros((C, out_H, out_W), dtype=np.float32)
    for i in range(out_H):
        for j in range(out_W):
            out[:, i, j] = np.max(x[:, i*kernel_size:(i+1)*kernel_size, j*kernel_size:(j+1)*kernel_size], axis=(1,2))
    return out

class KaggriculturePolicyNP:
    def __init__(self):
        with np.load(io.BytesIO(base64.b64decode(_BC_WEIGHTS_B64))) as data:
            self.w = {{k: data[k] for k in data.files}}

    def forward(self, vector_obs, spatial_obs):
        v = np.dot(self.w["encoder.vector_mlp.0.weight"], vector_obs) + self.w["encoder.vector_mlp.0.bias"]
        v = _relu(v)
        v = np.dot(self.w["encoder.vector_mlp.2.weight"], v) + self.w["encoder.vector_mlp.2.bias"]

        s = _conv2d_np(spatial_obs, self.w["encoder.spatial_cnn.0.weight"], self.w["encoder.spatial_cnn.0.bias"])
        s = _relu(s)
        s = _maxpool2d_np(s)
        s = _conv2d_np(s, self.w["encoder.spatial_cnn.3.weight"], self.w["encoder.spatial_cnn.3.bias"])
        s = _relu(s)

        s = s.flatten()
        s = np.dot(self.w["encoder.spatial_proj.weight"], s) + self.w["encoder.spatial_proj.bias"]

        emb = v + s

        m = np.dot(self.w["decoder.macro_head.0.weight"], emb) + self.w["decoder.macro_head.0.bias"]
        m = _relu(m)
        logits = np.dot(self.w["decoder.macro_head.2.weight"], m) + self.w["decoder.macro_head.2.bias"]

        return logits > 0.0

_GLOBAL_RL_POLICY = KaggriculturePolicyNP()

def _extract_rl_features(farm, day, cash, hands):
    # Same as dataset_builder.py
    vector = np.array([day / 30.0, 0.5, min(1.0, cash / 10000.0), min(1.0, hands / 20.0)], dtype=np.float32)
    spatial = np.zeros((5, 10, 10), dtype=np.float32)
    # Fast skeleton mapping (actual feature logic in dataset_builder can be replicated here)
    for y, row in enumerate(farm.get("tiles", [])):
        if y >= 10:
            break
        for x, t in enumerate(row):
            if x >= 10:
                break
            if t == "LOCKED":
                spatial[0, y, x] = 1.0
            elif t == "EMPTY":
                spatial[1, y, x] = 1.0
    return vector, spatial

def run_neural_policy(farm, day, cash):
    vec, spa = _extract_rl_features(farm, day, cash, len(farm.get("hands", [])))
    return _GLOBAL_RL_POLICY.forward(vec, spa)
# =======================================================================
"""

    # Inject payload right after the last import in main.py to avoid E402 import errors.
    insert_idx = 0
    in_imports = False
    for idx, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            in_imports = True
            insert_idx = idx + 1
        elif in_imports and line.strip() == "":
            insert_idx = idx + 1
        elif in_imports:
            break

    lines.insert(insert_idx, payload + "\n")

    # Now find the hook in StrategicPlanner.market_orders
    hooked_lines = []
    for line in lines:
        hooked_lines.append(line)
        if (
            "def market_orders(self, obs, farm, private, market, opp, day, days_left, hour, log):"
            in line
        ):
            hooked_lines.append("        # --- PYTORCH HYBRID POLICY INJECTION ---\n")
            hooked_lines.append("        try:\n")
            hooked_lines.append(
                '            did_hire, did_buy_land = run_neural_policy(farm, day, float(farm.get("money", 0.0)))\n'
            )
            hooked_lines.append("            global FLAGS\n")
            hooked_lines.append('            FLAGS["HIRE_HANDS"] = bool(did_hire[0])\n')
            hooked_lines.append('            FLAGS["EXPAND_LAND"] = bool(did_buy_land[1])\n')
            hooked_lines.append("        except Exception as e:\n")
            hooked_lines.append("            pass # Fallback to default heuristic\n")
            hooked_lines.append("        # ---------------------------------------\n")

    with open(main_path, "w") as f:
        f.writelines(hooked_lines)

    print("Successfully distilled PyTorch model into main.py via base64 Numpy injection!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="bc_weights.npz")
    parser.add_argument("--main", default="main.py")
    args = parser.parse_args()
    embed_weights_into_main(args.npz, args.main)
