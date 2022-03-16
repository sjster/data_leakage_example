import pandas as pd
import plotly
import plotly.express as px

pattern = "_r2$"

class PlotGraphs():
    def __init__(self, color1='blue', color2='teal'):
        self.color1 = color1
        self.color2 = color2

    def plot_box(self, dataset_name):
        r"""Read and create boxplots for preprocess before split and after split $\frac{x}{y}$
        >>>print("foo")
        foo
        """
        df_true = pd.read_csv(f'df_{dataset_name}_true.csv').filter(regex=pattern)
        df_false = pd.read_csv(f'df_{dataset_name}_false.csv').filter(regex=pattern)
        fig = px.box(df_true)
        fig2 = px.box(df_false)
        fig.add_trace(fig2['data'][0])
        fig['data'][1].line.color = self.color2
        fig.update_layout(showlegend=True, title='PREPROCESS_BEFORE_SPLIT (Purple) vs. PREPROCESS_AFTER_SPLIT (Teal)')
        plotly.offline.plot(fig, filename=f'{dataset_name}.html')

plot_obj = PlotGraphs()
plot_obj.plot_box('boston')
plot_obj.plot_box('diabetes')
plot_obj.plot_box('california')
