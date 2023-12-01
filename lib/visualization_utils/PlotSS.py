import torch
from matplotlib import pyplot as plt




def get_Gt_samples(
    ss, 
    M, 
    dataset
):
    ss.cpu()
    N = 10
    time_points = [ (ss.num_steps // (N-1)) * i for i in range(N) ]
    time_points[-1] = ss.num_steps - 1
    x0 = dataset.sample('train', M)
    collected_Gt = []
    for t in time_points:
        time_vector = t * torch.ones((x0.shape[0],), dtype=torch.long)
        Gt = ss.sample_Gt(x0, time_vector)
        collected_Gt.append(Gt)
    return collected_Gt, time_points




def get_xt_samples(
    ss, 
    M, 
    dataset
):
    ss.cpu()
    N = 10
    time_points = [ (ss.num_steps // (N-1)) * i for i in range(N) ]
    time_points[-1] = ss.num_steps - 1
    x0 = dataset.sample('train', M)
    collected_xt = []
    for t in time_points:
        time_vector = t * torch.ones((x0.shape[0],), dtype=torch.long)
        xt = ss.forward_step_sample(x0, time_vector)
        collected_xt.append(ss.to_domain(xt))
    return collected_xt, time_points




def plot_xt_samples(
    collected_xt,
    time_points,
    manifold_visualization,
    fpath,
    title='x_t'
):
    fs_suptitle = 46
    fs_timepoints = 28
    ax_sz = manifold_visualization.size
    figsize = (5*ax_sz[0], 2*ax_sz[1])
    fig = plt.figure(figsize=figsize)
    if title == 'x_t':
        fig.suptitle(r'$\bf{x_t}$', fontsize=fs_suptitle)
    elif title == 'G_t':
        fig.suptitle(r'$\bf{G_t}$', fontsize=fs_suptitle)
    for i in range(10):
        ax = plt.subplot2grid((2,5), (i//5,i%5))
        manifold_visualization(ax, collected_xt[i])
        plt.xlabel(f't = {time_points[i]}', fontsize=fs_timepoints)
    fig.savefig(fpath, bbox_inches='tight')
    plt.show()
    pass




def plot_xt_ss_process(
    ss, 
    num_objects, 
    dataset, 
    manifold_visualization,
    fpath
):
    collected_xt, time_points = get_xt_samples(
        ss=ss, 
        M=num_objects, 
        dataset=dataset
    )
    plot_xt_samples(
        collected_xt,
        time_points,
        manifold_visualization,
        fpath
    )
    pass



