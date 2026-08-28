import matplotlib.pyplot as plt
import torch

def save_projection_plot(harmful_means, harmless_means, direction, position_idx, out_path):
    d = direction.float().cpu()
    h = harmful_means[:, position_idx, :].float().cpu()
    s = harmless_means[:, position_idx, :].float().cpu()

    h_proj = torch.einsum("ld,d->l", h, d).numpy()
    s_proj = torch.einsum("ld,d->l", s, d).numpy()
    xs = list(range(len(h_proj)))

    plt.figure(figsize=(8, 5))
    plt.plot(xs, h_proj, marker="o", label="harmful mean projection")
    plt.plot(xs, s_proj, marker="o", label="harmless mean projection")
    plt.xlabel("Decoder layer")
    plt.ylabel("Projection onto selected direction")
    plt.title("Mean residual projection by layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
