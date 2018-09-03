# ### Visualize similarities of tokens to NE
from IPython.core.display import display, HTML
from src.fewshot_ner_viz_component.utils import flatten_sim, zip_tokens_sim, zip_tokens_sim_list
import numpy as np

class NeSimVisualizer():
    def __init__(self, color:dict=None, height='auto', width='auto', markPredCorrect=False):
        self.bg_color = color if color != None else NeSimVisualizer.get_color()
        self.height = height
        self.width = width
        self.markPredCorrect = markPredCorrect

    @staticmethod
    def get_color(red=0, green=255, blue=0):
        return {'r': red, 'g': green, 'b': blue}

    @staticmethod
    def get_rgba_str(color, alpha=1):
        return 'rgba({},{},{},{})'.format(color['r'], color['g'], color['b'], alpha)

    def get_token_span_str(self, token, color, cf=1, correctPred=True):
        s = '<span style="padding: 0.15em; border-radius: 0.25em; background: {};">{}</span>'.format(NeSimVisualizer.get_rgba_str(color, alpha=cf), token)
        s = '<span style="padding: 0.1em; padding-bottom: 0.2em; margin: 2px; {}">{}</span>'.format('border-bottom: 3px solid red' if self.markPredCorrect and not correctPred else '', s)
        return s

    def wrap_with_style(self, html):
        return '<div style="line-height: 1.8em; border: 2px solid black; padding: 5px; height: {1}; width: {2};">{0:s}</div>'.format(html, self.height, self.width)

    def sim_transform_lin(self, sim):
        # similarity transformation for better visualization
        return (sim - self.sim_min)/(self.sim_max - self.sim_min)

    def sim_transform(self, sim, T=0.5):
        # similarity transformation with temperature for better visualization
        return (np.exp(sim/T) - np.exp(self.sim_min/T))/(np.exp(self.sim_max/T) - np.exp(self.sim_min/T))

    @staticmethod
    def color_from_gradient(color1, color2, coeff):
        color = NeSimVisualizer.get_color(0, 0, 0)
        for ch in color.keys():
            color[ch] = color1[ch]*(1-coeff) + color2[ch]*coeff
        return color

    def get_colored_results_html(self, tokens_sim: list, color, transform_sim=True, T=0.5, title=''):
        chars_on_line_max = 108
        s = ''
        if title != '':
            s += '<h3 style="margin-bottom:0.3em;">{}</h3>'.format(title)
        s += '<br/>'
        k = 0
        for i, seq in enumerate(tokens_sim):
            n_chars_on_line = 0
            for j, (token, sim) in enumerate(seq):
                n_chars_on_line += len(token) + 1
                if transform_sim:
                    sim = self.sim_transform(sim, T)
                alpha = sim
                if isinstance(color, list):
                    if len(color) > 1:
                        color_display = NeSimVisualizer.color_from_gradient(color[0], color[1], sim)
                        if sim < 0.5:
                            alpha = 1 - sim*2
                        else:
                            alpha = (sim - 0.5)*2
                    else:
                        color_display = color[0]
                else:
                    color_display = color

                correct = None
                if isinstance(self.true_labels, list) or isinstance(self.true_labels, np.ndarray):
                    correct = (sim >= 0.5 and self.true_labels[k] == 1) or ( sim < 0.5 and self.true_labels[k] == 0)

                s += self.get_token_span_str(token, color_display, cf=alpha, correctPred=correct)
                if n_chars_on_line > chars_on_line_max:
                    s += '<br/>'
                    n_chars_on_line = 0

                k += 1

            s += '<br/><br/>'
        return self.wrap_with_style(s)

    def display_ne_sim(self, tokens, sim_list, transform=True, title='', true_labels=None):
        self.true_labels = true_labels
#         print(self.true_labels)
#         print(sim_list)
        if isinstance(sim_list[0], list) and isinstance(sim_list[0][0], dict):
            tokens_sim = zip_tokens_sim(tokens, sim_list)
            sim_flat = flatten_sim(sim_list)
            self.sim_min = np.min(sim_flat['cosine'])
            self.sim_max = np.max(sim_flat['cosine'])
        else:
            tokens_sim = zip_tokens_sim_list(tokens, sim_list)
#         print(tokens_sim)
        display(HTML(self.get_colored_results_html(tokens_sim, color=self.bg_color, transform_sim=transform, title=title)))
