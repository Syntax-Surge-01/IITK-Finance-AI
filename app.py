# from flask import Flask, render_template, request
# import pandas as pd
# from data_preprocessing import load_and_clean_data
# from vif_analysis import calculate_vif
# from inflation_adjustment import adjust_for_inflation
# from plots import create_scatter_plot, create_line_chart, create_bar_chart

# app = Flask(__name__)

# # Load and process data
# df = load_and_clean_data('C:\\Users\\cheta\\Documents\\financial_model\\Financial Statements.csv')
# df = adjust_for_inflation(df, 'Net_Income', 'Market_cap', 'Inflation_rate')
# df['mk_int'] = df['Adjusted_market_cap'].round().astype(int)

# @app.route('/')
# def home():
#     companies = df['Company'].unique()
#     return render_template('index.html', companies=companies)

# @app.route('/plot', methods=['POST'])
# def plot():
#     selected_company = request.form.get('company')
#     df_filtered = df[df['Company'] == selected_company]

#     scatter_html = create_scatter_plot(df_filtered)
#     line_html = create_line_chart(df_filtered)
#     bar_html = create_bar_chart(df_filtered)

#     return render_template('plot.html', scatter_html=scatter_html, line_html=line_html, bar_html=bar_html)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request
import pandas as pd
from data_preprocessing import load_and_clean_data
from vif_analysis import calculate_vif
from inflation_adjustment import adjust_for_inflation
from plots import (create_scatter_plot, create_line_chart, create_bar_chart, 
                   create_cash_flow_by_company, create_cash_flow_by_year, create_cash_flow_by_industry,px_scatter_plot)

app = Flask(__name__)

# Load and process data
df = load_and_clean_data('C:\\Users\\cheta\\Documents\\financial_model\\Financial Statements.csv')
df = adjust_for_inflation(df, 'Net_Income', 'Market_cap', 'Inflation_rate')
df['mk_int'] = df['Adjusted_market_cap'].round().astype(int)
df['CF_margin'] = df['Cash_Flow_from_Operating'] / df['Revenue']  # Compute Cash Flow Margin

@app.route('/')
def home():
    companies = df['Company'].unique()
    return render_template('index.html', companies=companies)

# @app.route('/plot', methods=['POST'])
# def plot():
#     selected_company = request.form.get('company')
#     df_filtered = df[df['Company'] == selected_company]

#     scatter_html = create_scatter_plot(df_filtered)
#     line_html = create_line_chart(df_filtered)
#     bar_html = create_bar_chart(df_filtered)
#     cf_company_html = create_cash_flow_by_company(df_filtered)
#     cf_year_html = create_cash_flow_by_year(df_filtered)
#     cf_industry_html = create_cash_flow_by_industry(df_filtered)
#     px_scatter_html = px_scatter_plot(df_filtered)

#     return render_template('plot.html', scatter_html=scatter_html, line_html=line_html, 
#                            bar_html=bar_html, cf_company_html=cf_company_html, 
#                            cf_year_html=cf_year_html, cf_industry_html=cf_industry_html,px_scatter_html = px_scatter_html)


@app.route('/plot', methods=['POST'])
def plot():
    selected_company = request.form.get('company')
    df_filtered = df[df['Company'] == selected_company]

    scatter_html, scatter_explanation = create_scatter_plot(df_filtered)
    line_html, line_explanation = create_line_chart(df_filtered)
    bar_html, bar_explanation = create_bar_chart(df_filtered)
    cf_company_html, cf_company_explanation = create_cash_flow_by_company(df_filtered)
    cf_year_html, cf_year_explanation = create_cash_flow_by_year(df_filtered)
    cf_industry_html, cf_industry_explanation = create_cash_flow_by_industry(df_filtered)
    scatter_3d_html, scatter_3d_explanation = px_scatter_plot(df_filtered)

    return render_template('plot.html', 
                           scatter_html=scatter_html, scatter_explanation=scatter_explanation,
                           line_html=line_html, line_explanation=line_explanation,
                           bar_html=bar_html, bar_explanation=bar_explanation,
                           cf_company_html=cf_company_html, cf_company_explanation=cf_company_explanation,
                           cf_year_html=cf_year_html, cf_year_explanation=cf_year_explanation,
                           cf_industry_html=cf_industry_html, cf_industry_explanation=cf_industry_explanation,
                           scatter_3d_html=scatter_3d_html, scatter_3d_explanation=scatter_3d_explanation)

if __name__ == '__main__':
    app.run(debug=True)
