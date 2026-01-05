import plotly.graph_objects as go

# 文档→主题→子主题的流量
labels = ["文档", "主题A", "主题B", "子主题A1", "子主题A2", "子主题A3", "子主题B1", "子主题B2"]
source = [0, 0, 1, 1, 1, 2, 2]  # 源节点索引
target = [1, 2, 3, 4, 5, 6, 7]  # 目标节点索引
value = [600, 400, 300, 200, 100, 250, 150]  # 流量

# 绘制桑基图
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
        color=["gray", "red", "blue", "red", "red", "red", "blue", "blue"]
    ),
    link=dict(
        source=source,
        target=target,
        value=value
    )
)])

fig.update_layout(
    title_text="桑基图示例（文档→主题→子主题）",
    font_size=12,
    width=1000,
    height=600
)
fig.show()