import numpy as np


def _relu(x):
    return np.maximum(0, x)


def _maxpool2d_np(x, size=2):
    C, H, W = x.shape
    out_H = H // size
    out_W = W // size
    return x.reshape(C, out_H, size, out_W, size).max(axis=(2, 4))


def _conv2d_np(x, w, b, padding=1, stride=1):
    C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    out_H = (H + 2 * padding - kH) // stride + 1
    out_W = (W + 2 * padding - kW) // stride + 1
    if padding > 0:
        x_pad = np.pad(x, ((0, 0), (padding, padding), (padding, padding)), mode="constant")
    else:
        x_pad = x
    out = np.zeros((C_out, out_H, out_W), dtype=np.float32)
    for i in range(out_H):
        for j in range(out_W):
            x_slice = x_pad[:, i * stride : i * stride + kH, j * stride : j * stride + kW]
            out[:, i, j] = np.sum(x_slice * w, axis=(1, 2, 3)) + b
    return out


class NumpyPolicyFullRL:
    def __init__(self, weights_dict):
        self.w = weights_dict

    def _linear(self, name, x):
        return np.dot(self.w[f"{name}.weight"], x) + self.w[f"{name}.bias"]

    def _conv2d(self, name, x, padding=1, stride=1):
        return _conv2d_np(x, self.w[f"{name}.weight"], self.w[f"{name}.bias"], padding, stride)

    def __call__(self, vector_obs, spatial_obs):
        # vector_obs: (4,)
        # spatial_obs: (5, 10, 10)

        # 1. Observation Encoder
        v = self._linear("encoder.vector_mlp.0", vector_obs)
        v = _relu(v)
        v = self._linear("encoder.vector_mlp.2", v)

        s = self._conv2d("encoder.spatial_cnn.0", spatial_obs, padding=1)
        s = _relu(s)
        s = _maxpool2d_np(s, 2)
        s = self._conv2d("encoder.spatial_cnn.3", s, padding=1)
        s = _relu(s)

        s_flat = s.flatten()
        s_proj = self._linear("encoder.spatial_proj", s_flat)

        state_emb = v + s_proj

        # 2. Farmer Head
        f = self._linear("farmer_head.net.0", state_emb)
        f = _relu(f)
        f_logits = self._linear("farmer_head.net.2", f)

        # 3. Market Head
        m = self._linear("market_head.net.0", state_emb)
        m = _relu(m)
        m_preds = self._linear("market_head.net.2", m)
        m_preds = m_preds.reshape((12, 3))

        # 4. Hands Spatial Head
        v_grid = np.broadcast_to(vector_obs[:, None, None], (4, 10, 10))
        x_hands = np.concatenate([spatial_obs, v_grid], axis=0)

        h = self._conv2d("hands_head.conv_net.0", x_hands, padding=1)
        h = _relu(h)
        h = self._conv2d("hands_head.conv_net.2", h, padding=1)
        h = _relu(h)
        h_logits = self._conv2d("hands_head.conv_net.4", h, padding=0)

        return f_logits, h_logits, m_preds
