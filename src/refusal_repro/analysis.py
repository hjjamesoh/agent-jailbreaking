import matplotlib.pyplot as plt
import torch


def cosine_similarity(a, b, eps=1e-8):
    a = a.float().cpu().flatten()
    b = b.float().cpu().flatten()
    denom = a.norm().clamp_min(eps) * b.norm().clamp_min(eps)
    return float(torch.dot(a, b) / denom)


def cosine_similarity_grid(reference_direction, candidates):
    reference = reference_direction.float().cpu()
    candidate_grid = candidates.float().cpu()
    reference = reference / reference.norm().clamp_min(1e-8)
    candidate_grid = candidate_grid / candidate_grid.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.einsum("d,lpd->lp", reference, candidate_grid)


def top_cosine_matches(reference_direction, candidates, positions, top_k=10):
    grid = cosine_similarity_grid(reference_direction, candidates)
    rows = []
    for layer in range(grid.shape[0]):
        for position_idx, position in enumerate(positions):
            rows.append({
                "layer": int(layer),
                "position": int(position),
                "cosine_similarity": float(grid[layer, position_idx]),
                "abs_cosine_similarity": abs(float(grid[layer, position_idx])),
            })
    rows.sort(key=lambda row: row["abs_cosine_similarity"], reverse=True)
    return rows[:top_k], grid


def save_cosine_heatmap(grid, positions, out_path, title="Direction cosine similarity"):
    values = grid.float().cpu().numpy()
    plt.figure(figsize=(8, 5))
    plt.imshow(values.T, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="cosine similarity")
    plt.yticks(range(len(positions)), [str(p) for p in positions])
    plt.xlabel("Decoder layer")
    plt.ylabel("Token position")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


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
