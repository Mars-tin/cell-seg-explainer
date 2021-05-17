from explain_label import *


def view_full_connect():
    load_file = 'endo_seed=0.tar'
    data = 'real'
    model_name = 'gcn'
    seed = 0
    epoch, data, model = load_model(model_name, data, seed, load_file)

    edges = np.array(data.edge_index).transpose()
    all_nodes = np.arange(data.sizes[0])
    all_edges = []
    for edge in edges:
        if edge[0] in all_nodes:
            all_edges.append([edge[0], edge[1], 1])

    islet = [(11750, 12750, 6250, 7500), (10, 12)]
    view_local(islet, all_nodes, all_edges, data, 'islet_{}_{}'.format(121, 212))


def view_marker_mask(marker_strength):
    marker_strength /= np.sum(marker_strength)
    plt.figure(figsize=(12, 2))
    heatmap(np.expand_dims(marker_strength, axis=0), square=False, cmap='Blues',
            vmax=1/2, vmin=0, xticklabels=[], yticklabels=[], annot=True)
    plt.savefig('plot/' + 'masks')
    plt.show()


def view_marker_list(marker_intensity, marker_completeness, id):
    fig, axs = plt.subplots(2, figsize=(12, 6))
    for i, marker in enumerate([marker_intensity, marker_completeness]):
        heatmap(np.expand_dims(marker, axis=0), square=False, cmap='Blues', ax=axs[i],
                vmax=np.max(marker), vmin=0, xticklabels=[], yticklabels=[], annot=True)
    plt.savefig('plot/' + 'marker_{}'.format(id))
    plt.show()


view_full_connect()
