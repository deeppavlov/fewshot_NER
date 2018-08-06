from ne_sim_visualizer import NeSimVisualizer

if __name__ == '__main__':
    viz = NeSimVisualizer(color=NeSimVisualizer.get_color())
    viz.display_ne_sim([['token']], [[{'euc_dist': 0, 'dot_prod': 0, 'cosine':0.7}]])
