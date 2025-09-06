import plotly.graph_objects as go


def create_probability_chart(prob_normal, prob_cataract):
    """Create interactive probability chart"""
    
    fig = go.Figure(data=[
        go.Bar(
            name='Probability',
            x=['Normal', 'Cataract'],
            y=[prob_normal, prob_cataract],
            marker_color=['#27AE60', '#E74C3C'],
            text=[f'{prob_normal*100:.1f}%', f'{prob_cataract*100:.1f}%'],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Prediction Probabilities',
        yaxis_title='Probability',
        xaxis_title='Class',
        showlegend=False,
        height=400,
        yaxis=dict(range=[0, 1])
    )
    
    return fig


def create_confidence_gauge(confidence):
    """Create confidence gauge chart"""
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level (%)"},
        delta = {'reference': 70},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig