import pandas as pd
import plotly
import plotly.express as px

pattern = "_r2$"

def plot_box(dataset_name):
    df_true = pd.read_csv(f'df_{dataset_name}_true.csv').filter(regex=pattern)
    df_false = pd.read_csv(f'df_{dataset_name}_false.csv').filter(regex=pattern)
    fig = px.box(df_true)
    fig2 = px.box(df_false)
    fig.add_trace(fig2['data'][0])
    fig['data'][1].line.color = 'teal'
    fig.update_layout(showlegend=True, title='PREPROCESS_BEFORE_SPLIT (Purple) vs. PREPROCESS_AFTER_SPLIT (Teal)')
    plotly.offline.plot(fig, filename=f'{dataset_name}.html')

plot_box('boston')
plot_box('diabetes')
plot_box('california')
