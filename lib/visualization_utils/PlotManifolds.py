import os

import torch
import numpy as np
from matplotlib import pyplot as plt
import ternary
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap

from saving_utils.get_repo_root import get_repo_root




def simplex_plot(
    ax, 
    points,
    scale=80, 
    max_prob=0.012,
    gap=0.4,
    space_around=0.0,
    cmap='inferno'
):
    points = points.reshape(-1, 3)
    x, y, z = 100 * points.mean(dim=0)
    
    gap = gap * max_prob
    coef = (max_prob - gap) / max_prob
    field_scale = scale
    bound = int(space_around * scale)
    scale += 3 * bound
    
    all_indexes = torch.arange(3)
    for p in points:
        S = p.sum()
        if abs(S - 1.) > 1e-2 or (p < 0).any() or (p > 1).any():
            min_index = p.argmin()
            not_min_index = (all_indexes != min_index)
            min_coord_val = 0 - abs(1 - S) - 5e-2
            p[min_index] = min_coord_val
            if p[not_min_index].sum() < 0:
                p[not_min_index][0], p[not_min_index][1] = p[not_min_index][1], p[not_min_index][0]
            p[not_min_index] = p[not_min_index] / p[not_min_index].sum()
            p[not_min_index] -= min_coord_val / 2
            # assert abs(p.sum() - 1) < 1e-5
    
    # point in simplex == rpoint in [0, field_scale]
    rpoints = (points.clone() * field_scale).round().to(int) 
    heatmap = {}
    for i in range(scale+1): # [0, ..., bound, ..., field_scale+bound, ..., field_scale+3*bound]
        for j in range(scale+1):
            freq = (rpoints[:,:2] == torch.tensor([i-bound,j-bound])).all(dim=1).sum().item()
            if freq > 0:
                heatmap[(i, j)] = (gap + coef * freq / points.shape[0])**3
            else:
                heatmap[(i, j)] = 0
            
    figure, tax = ternary.figure(ax=ax, scale=scale)
    tax.boundary(linewidth=2.0)
    tax.heatmap(
        heatmap, colorbar=False, 
        vmin=0, vmax=0.007**3, 
        cmap=cmap
    )
    #tax.bottom_axis_label(f"({x:4.1f}, {y:4.1f}, {z:4.1f})", fontsize=12)
    
    bound = bound-1
    points = [ 
        [scale-2*bound,bound,bound],
        [bound,scale-2*bound,scale],
        [bound,bound,scale-2*bound],
        [scale-2*bound,bound,bound]
    ]
    tax.plot(points, linewidth=2, color='white')
    tax.clear_matplotlib_ticks()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    pass
simplex_plot.size = (7,7)




def matrix2ellipse(matrixes):
    Rinv = torch.linalg.inv(torch.linalg.cholesky(matrixes, upper=True))
    t = torch.linspace(-np.pi, np.pi, 2000)
    xx = torch.cos(t)
    yy = torch.sin(t)
    circle = torch.stack((xx, yy))
    ellipses = Rinv @ circle
    return ellipses




def pdm2x2_plot(
    ax, 
    samples,
    color_gap=0.004,
    color_max=0.020,
    red_ellipses=None,
    cmap='bone'
):
    samples = samples.reshape(-1, 2, 2)
    # check is positive definite
    notneg_eigvals = (torch.linalg.eigvals(samples).to(torch.float32) > 0)
    is_pos_def = (notneg_eigvals.sum(dim=1) == samples.shape[-1])
    percent_of_pos_def = is_pos_def.to(int).sum() / samples.shape[0]
    if percent_of_pos_def < 0.9:
        print(f'WARNING: {100 * (1 - percent_of_pos_def):4.2f}% not positive definite')
    samples = samples[is_pos_def]

    # ellipses 
    ellipses = matrix2ellipse(samples)
    if not (red_ellipses is None):
        red_ellipses = matrix2ellipse(red_ellipses)

    xlim = (-1.2, 1.2)
    ylim = (-1.2, 1.2)
    scale = 1000

    eps = xlim[1] / 100
    heatmap = np.zeros((scale, scale), dtype=float)
    for ellipse in ellipses.numpy():
        mask = (
            (xlim[0] + eps < ellipse[0]) * (ellipse[0] < xlim[1] - eps) *
            (ylim[0] + eps < ellipse[1]) * (ellipse[1] < ylim[1] - eps)
        )
        if mask.sum() == 0:
            continue
        ex, ey = ellipse[0][mask], ellipse[1][mask]
        x_coords = ((ex - xlim[0]) / (xlim[1] - xlim[0]) * scale).round().astype(int)
        y_coords = ((ey - ylim[0]) / (ylim[1] - ylim[0]) * scale).round().astype(int)
        heatmap[y_coords, x_coords] += 1
    
    not_zero = (heatmap != 0)
    heatmap[not_zero] += color_gap * samples.shape[0]  
    heatmap *= -1

    if not (red_ellipses is None):
        for ellipse in red_ellipses.numpy():
            mask = (
              (xlim[0] + eps < ellipse[0]) * (ellipse[0] < xlim[1] - eps) *
              (ylim[0] + eps < ellipse[1]) * (ellipse[1] < ylim[1] - eps)
            )
            if mask.sum() == 0:
                continue
            ex, ey = ellipse[0][mask], ellipse[1][mask]
            x_coords = ((ex - xlim[0]) / (xlim[1] - xlim[0]) * scale).round().astype(int)
            y_coords = ((ey - ylim[0]) / (ylim[1] - ylim[0]) * scale).round().astype(int)
            heatmap[y_coords, x_coords] = float('nan')

    cmap = plt.cm.__dict__[cmap]
    cmap.set_bad((1, 0, 0.4, 1))

    plt.imshow(heatmap, cmap=cmap, vmin=-color_max*samples.shape[0], vmax=0)
    plt.xticks([])
    plt.yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    pass
pdm2x2_plot.size = (7,7)




def sphere_plot(
    ax, 
    samples
):
    x = samples
    x = x / torch.norm(x, dim=-1, keepdim=True)
    angle = torch.zeros(*x.shape[:-1], 2)
    angle[..., 0] = torch.atan2(x[..., 1], x[..., 0])
    angle[..., 1] = torch.arcsin(x[..., 2])
    samples = (180 * angle / np.pi).reshape(-1,2)

    lon, lat = samples[:, 0].cpu().numpy(), samples[:, 1].cpu().numpy()
    basemap = Basemap(projection='robin', lat_0=0, lon_0=0)
    x, y = basemap(lon, lat)
        
    continents_color = 'lightcyan'
    water_color = 'deepskyblue'
    size=10
    alpha=0.1
    marker="o"
    color='crimson'
    
    basemap.fillcontinents(color=continents_color, lake_color=water_color)
    basemap.drawmapboundary(fill_color=water_color)
    ax.scatter(
        x, y, 
        s=size,
        alpha=alpha,
        c=color, 
        marker=marker
    )
    pass
sphere_plot.size = (10,5)
    



def plot_simplex_result(
    dataset_samples, 
    model_samples, 
    model_name,
    graphic_path=None
):
    ax_sz = simplex_plot.size
    figsize = (2*ax_sz[0], ax_sz[1])
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(121)
    simplex_plot(
        ax, dataset_samples
    )
    ax.set_title('Original', fontsize=20)
    ax = fig.add_subplot(122)
    simplex_plot(
        ax, model_samples
    )
    ax.set_title(model_name, fontsize=20)
    if not (graphic_path is None):
        fig.savefig(os.path.join(get_repo_root(), 'results', graphic_path+'.png'))
    plt.show()




def plot_pdm2x2_result(
    dataset_samples, 
    model_samples, 
    model_name,
    red_ellipses=None,
    graphic_path=None
):
    ax_sz = pdm2x2_plot.size
    figsize = (2*ax_sz[0], ax_sz[1])
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(121)
    pdm2x2_plot(
        ax, dataset_samples,
        red_ellipses=red_ellipses
    )
    ax.set_title('Original', fontsize=20)
    ax = fig.add_subplot(122)
    pdm2x2_plot(
        ax, model_samples,
        red_ellipses=red_ellipses
    )
    ax.set_title(model_name, fontsize=20)
    if not (graphic_path is None):
        fig.savefig(os.path.join(get_repo_root(), 'results', graphic_path+'.png'))
    plt.show()




def plot_sphere_result(
    dataset_samples, 
    model_samples, 
    model_name,
    graphic_path=None
):
    ax_sz = sphere_plot.size
    figsize = (2*ax_sz[0], ax_sz[1])
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(121)
    sphere_plot(ax, dataset_samples)
    ax.set_title('Original', fontsize=20)
    ax = fig.add_subplot(122)
    sphere_plot(ax, model_samples)
    ax.set_title(model_name, fontsize=20)
    if not (graphic_path is None):
        fig.savefig(os.path.join(get_repo_root(), 'results', graphic_path+'.png'))
    plt.show()



