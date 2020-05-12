import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import HoverTool, ColumnDataSource, Slider, RadioButtonGroup, Div, Paragraph
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, factor_mark
from bokeh.palettes import Category20_4
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)

## load data
x = np.random.rand(100)*10 - 5
y = 5/(1 + np.exp(-x)) + np.random.randn(100)

x_min, x_max = x.min() - 0.1, x.max() + 0.1
y_min, y_max = y.min() - 0.1, y.max() + 0.1

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

# Set up data
source = ColumnDataSource(data=dict(x=[], y=[]))

# Set up plot
plot = figure(plot_height=400, plot_width=600, title='',
              tools="crosshair,pan,reset,save,wheel_zoom", 
              x_range=[x_min, x_max], y_range=[y_min, y_max],
              x_axis_label='X',
              y_axis_label='Y')

plot.line('x', 'y', source=source, line_width=2, color='orange',
        legend_label='โมเดลที่ได้')

plot.scatter(x=x_train, y=y_train, fill_alpha=0.6, size=10, 
             color='#1f77b4', legend_label='train', name='train')
plot.scatter(x=x_test, y=y_test, fill_alpha=0.4, size=10, 
             color='#aec7e8', legend_label='test', name='test')

plot.legend.location = "top_left"
plot.legend.click_policy="hide"

# Set up dashboard
title = Div(text="""<H1>โมเดลที่เฉพาะเจาะจงเกินไป VS โมเดลที่ง่ายเกินไป (Overfitting/Underfitting in Regression)</H1>""")
desc = Paragraph(text="""ในแบบฝึกหัดนี้ ให้นักเรียนลองเปลี่ยนค่า hyperparamter degree ของ Linear Regression แล้วดูว่าเมื่อใดเกิด overfitting/underfitting
โดยการวาดกราฟเส้นระหว่างค่า MSE กับค่าตัวแปร Degree ของ Linear Regression ของทั้ง training data และ test data""")
header = column(title, desc, sizing_mode="scale_both")

# Set up widgets
text = Div(text="<H3>ตัวแปร</H3>")
degree_slider = Slider(title="Degree", value=1, start=1, end=10, step=1)
mse_txt = Div(text="<H3>MSE</H3>")
mse_tr_txt = Div(text=f'MSE บน training data = ')
mse_te_txt = Div(text=f'MSE บน test data = ')

def get_fig(degree=1):
    x_deg_train = [[x**(d+1) for d in range(degree)] for x in x_train]
    x_deg_test = [[x**(d+1) for d in range(degree)] for x in x_test]
    reg = LinearRegression()
    reg = reg.fit(x_deg_train, y_train)

    x_new = np.linspace(x_min, x_max, num=200)
    x_deg = [[x**(d+1) for d in range(degree)] for x in x_new]
    y_new = reg.predict(x_deg)

    source.data = dict(x=x_new, y=y_new)

    mse_tr = mean_squared_error(y_train, reg.predict(x_deg_train))
    mse_te = mean_squared_error(y_test, reg.predict(x_deg_test))
    mse_tr_txt.text = f'MSE บน training data = {mse_tr:.2f}'
    mse_te_txt.text = f'MSE บน test data = {mse_te:.2f}'

get_fig()

# Set up layouts and add to document
inputs = column(text, degree_slider, mse_txt, mse_tr_txt, mse_te_txt)
body = row(inputs, plot, width=800)

def update_data(attrname, old, new):
    degree = degree_slider.value
    get_fig(degree)
    
degree_slider.on_change('value', update_data)

curdoc().add_root(column(header, body))
curdoc().title = "โมเดลการถดถอยที่เฉพาะเจาะจง/ง่ายเกินไป"