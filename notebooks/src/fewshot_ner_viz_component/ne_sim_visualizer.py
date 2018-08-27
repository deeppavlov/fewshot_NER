# ### Visualize similarities of tokens to NE
from IPython.core.display import display, HTML
from src.fewshot_ner_viz_component.utils import flatten_sim, zip_tokens_sim, zip_tokens_sim_list
import numpy as np

class NeSimVisualizer():
    def __init__(self, color:dict=None):
        self.bg_color = color if color != None else NeSimVisualizer.get_color()

    @staticmethod
    def get_color(red=0, green=255, blue=0):
        return {'r': red, 'g': green, 'b': blue}

    @staticmethod
    def get_rgba_str(color, alpha=1):
        return 'rgba({},{},{},{})'.format(color['r'], color['g'], color['b'], alpha)

    @staticmethod
    def get_token_span_str(token, color, cf=1):
        return '<span style="padding: 0.15em; margin-right: 4px; border-radius: 0.25em; background: {};">{}</span>'.format(NeSimVisualizer.get_rgba_str(color, alpha=cf), token)

    @staticmethod
    def wrap_with_style(html):
        return '<div style="line-height: 1.5em;">{:s}</div>'.format(html)

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
        s = '<h3 style="margin-bottom:0.3em;">{}</h3>'.format(title)
        s += '<br/><br/>'
        for seq in tokens_sim:
            for token, sim in seq:
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
                s += NeSimVisualizer.get_token_span_str(token, color_display, cf=alpha)
    #             s += ' '
            s += '<br/><br/>'
        return NeSimVisualizer.wrap_with_style(s)

    def display_ne_sim(self, tokens, sim_list, transform=True, title=''):
#         print(sim_list)
        if isinstance(sim_list[0], list) and isinstance(sim_list[0][0], dict):
            tokens_sim = zip_tokens_sim(tokens, sim_list)
            sim_flat = flatten_sim(sim_list)
            self.sim_min = np.min(sim_flat['cosine'])
            self.sim_max = np.max(sim_flat['cosine'])
        else:
            tokens_sim = zip_tokens_sim_list(tokens, sim_list)
        
        display(HTML(self.get_colored_results_html(tokens_sim, color=self.bg_color, transform_sim=transform)))