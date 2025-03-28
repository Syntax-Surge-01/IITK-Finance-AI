# import plotly.express as px
# import plotly.io as pio
# from transformers import pipeline

# # Load Pretrained NLP Model
# explanation_model = pipeline("text2text-generation", model="t5-small")

# def generate_explanation(graph_title, summary_stats):
#     prompt = f"Explain the graph titled '{graph_title}' based on this data summary: {summary_stats}."
#     explanation = explanation_model(prompt, max_length=100, do_sample=True)[0]['generated_text']
#     return explanation

# def create_scatter_plot(df):
#     fig = px.scatter(df, x='Adjusted_net_income', y='Earning_Per_Share', color='Company', size="mk_int", size_max=40,
#                      animation_frame="Year", animation_group="Company", range_x=[-13000, 100000], range_y=[-1, 15])
#     # return pio.to_html(fig, full_html=False)
#     # Generate Explanation
#     summary_stats = f"Max Income: {df['Adjusted_net_income'].max()}, Min Income: {df['Adjusted_net_income'].min()}"
#     explanation = generate_explanation("Scatter Plot of Net Income vs EPS", summary_stats)

#     return pio.to_html(fig, full_html=False), explanation

# def create_line_chart(df):
#     fig = px.line(df, x='Year', y='Adjusted_net_income', color='Company', title='Adjusted Net Income')
#     # return pio.to_html(fig, full_html=False)
#     summary_stats = f"Yearly trends show {df['Adjusted_net_income'].diff().mean()} average change."
#     explanation = generate_explanation("Line Chart of Adjusted Net Income", summary_stats)

#     return pio.to_html(fig, full_html=False), explanation

# def create_bar_chart(df):
#     fig = px.bar(df, x='Year', y='Earning_Per_Share', color='Company', barmode="group", title='EPS')
#     return pio.to_html(fig, full_html=False)


# def create_cash_flow_by_industry(df):
#     fig = px.histogram(df, x='Industry', y='CF_margin', color='Company', barmode='group',
#                        title='Cash Flow Margin by Industry', labels={'CF_margin': 'Cash Flow Margin'})
#     return pio.to_html(fig, full_html=False)

# def create_cash_flow_by_company(df):
#     fig = px.histogram(df, x='Company', y='CF_margin', color='Year', barmode='group',
#                        title='Cash Flow Margin by Company', labels={'CF_margin': 'Cash Flow Margin'})
#     return pio.to_html(fig, full_html=False)

# def create_cash_flow_by_year(df):
#     fig = px.bar(df, x='Year', y='CF_margin', color='Company', barmode='group',
#                  title='Cash Flow Margin by Year', labels={'CF_margin': 'Cash Flow Margin'})
#     return pio.to_html(fig, full_html=False)


# def px_scatter_plot(df):
#     fig = px.scatter_3d(df, x='Earning_Per_Share', y='ROE', z='Adjusted_net_income', color='Company', 
#                          title="3D Scatter Plot")
#     return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')


import plotly.express as px
import plotly.io as pio
import google.generativeai as genai
import os
import pandas as pd
import markdown

# Configure Gemini API Key
genai.configure(api_key=os.getenv("AIzaSyCftWKY2O3vCEPLzx9WCeeXelaYJ6EKNT8"))

def generate_explanation(graph_title, df):
    """
    Uses Google Gemini to generate a detailed explanation of the given graph.
    """

    if isinstance(df, pd.DataFrame):
        summary_stats = df.describe().to_string()
    else:
        summary_stats = "Data is not a valid DataFrame."

    prompt = f"""
    Explain the following financial graph in detail:

    **Graph Title**: {graph_title}

    **Dataset Summary**:
    {summary_stats}

    Provide key insights, trends, and financial significance.
    """

    model = genai.GenerativeModel("gemini-2.0-pro-exp")
    response = model.generate_content(prompt)

    return markdown.markdown(response.text)  # Return the generated explanation



def create_scatter_plot(df):
    fig = px.scatter(df, x='Adjusted_net_income', y='Earning_Per_Share', color='Company', size="mk_int", size_max=40,
                     animation_frame="Year", animation_group="Company", range_x=[-13000, 100000], range_y=[-1, 15])
    
    summary_stats = f"Max Income: {df['Adjusted_net_income'].max()}, Min Income: {df['Adjusted_net_income'].min()}"
    explanation = generate_explanation("Scatter Plot of Net Income vs EPS", summary_stats)
    
    return pio.to_html(fig, full_html=False), explanation

def create_line_chart(df):
    fig = px.line(df, x='Year', y='Adjusted_net_income', color='Company', title='Adjusted Net Income')

    summary_stats = f"Yearly trends show {df['Adjusted_net_income'].diff().mean()} average change."
    explanation = generate_explanation("Line Chart of Adjusted Net Income", summary_stats)

    return pio.to_html(fig, full_html=False), explanation

def create_bar_chart(df):
    fig = px.bar(df, x='Year', y='Earning_Per_Share', color='Company', barmode="group", title='EPS')

    summary_stats = f"Max EPS: {df['Earning_Per_Share'].max()}, Min EPS: {df['Earning_Per_Share'].min()}"
    explanation = generate_explanation("Bar Chart of EPS by Year", summary_stats)

    return pio.to_html(fig, full_html=False), explanation

def create_cash_flow_by_industry(df):
    fig = px.histogram(df, x='Industry', y='CF_margin', color='Company', barmode='group',
                       title='Cash Flow Margin by Industry', labels={'CF_margin': 'Cash Flow Margin'})

    summary_stats = f"Avg CF Margin: {df['CF_margin'].mean()}"
    explanation = generate_explanation("Cash Flow Margin by Industry", summary_stats)

    return pio.to_html(fig, full_html=False), explanation

def create_cash_flow_by_company(df):
    fig = px.histogram(df, x='Company', y='CF_margin', color='Year', barmode='group',
                       title='Cash Flow Margin by Company', labels={'CF_margin': 'Cash Flow Margin'})

    summary_stats = f"Max CF Margin: {df['CF_margin'].max()}, Min CF Margin: {df['CF_margin'].min()}"
    explanation = generate_explanation("Cash Flow Margin by Company", summary_stats)

    return pio.to_html(fig, full_html=False), explanation

def create_cash_flow_by_year(df):
    fig = px.bar(df, x='Year', y='CF_margin', color='Company', barmode='group',
                 title='Cash Flow Margin by Year', labels={'CF_margin': 'Cash Flow Margin'})

    summary_stats = f"Annual Avg CF Margin: {df.groupby('Year')['CF_margin'].mean().to_dict()}"
    explanation = generate_explanation("Cash Flow Margin by Year", summary_stats)

    return pio.to_html(fig, full_html=False), explanation

def px_scatter_plot(df):
    fig = px.scatter_3d(df, x='Earning_Per_Share', y='ROE', z='Adjusted_net_income', color='Company', 
                         title="3D Scatter Plot")

    summary_stats = f"Max ROE: {df['ROE'].max()}, Min ROE: {df['ROE'].min()}"
    explanation = generate_explanation("3D Scatter Plot of ROE vs EPS vs Net Income", summary_stats)

    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn'), explanation
