import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# To set a consistent color scheme for predictions
color_map = {'Yes': '#00CC96', 'No': '#EF553B'}  # Green for 'Yes', Red for 'No'
# Define colors for seaborn correlation matrix as [(position, (red, green, blue)), ...]
colors = [(0, "#00CC96"), (1, "#EF553B")]  # Green to Red
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)


st.title('TechTrendz Customer Value Prediction')
# Explanation below the title
st.write("""
Welcome to the TechTrendz Customer Value Prediction Tool! This tool leverages a sophisticated machine learning model to analyze customer data and predict the likelihood of customers making significant financial contributions through their purchases.

**Understanding 'High Value Purchases':**
- **Definition**: In our context, 'High Value Purchases' refers to transactions or a series of transactions that significantly exceed the average spending level of the customer base.
- **Importance**: Identifying customers likely to make such purchases allows businesses to focus their marketing efforts more effectively, tailor special offers, and optimize customer relationship management strategies.
         
**What the Model Does:**
- **Predictive Analysis**: The model categorizes customers based on their predicted purchasing behavior. It uses historical data, such as past purchases, customer interactions, and demographic information, to forecast who is likely to make high-value purchases in the future.
- **Results Presentation**: After processing, the tool displays which customers are predicted as likely to make high-value purchases ('High Value') and which are not, enabling targeted business strategies.

**How to Use This Tool:**
1. **Upload Your Data**: Load your customer data file using the sidebar uploader. Ensure your file includes necessary information that the model requires.
2. **Generate Predictions**: Click the 'Predict High Value Purchases' button once your file is uploaded. The model will process the data and predict customer purchasing behaviors.
3. **Review the Predictions**: Results are split into two groups — 'High Value' and 'Others'. Review these categories to understand which customers are the most valuable in terms of expected revenue generation.

This tool is designed to help you maximize your marketing and sales effectiveness by focusing on those customers who are most likely to contribute significantly to your revenue.
""")

# Load the pre-trained model
model = joblib.load('best_model.pkl')  # Ensure this path is correct

uploaded_file = st.sidebar.file_uploader("Upload your customer data Excel here:", type=["xlsx"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    if 'Customer ID' in df.columns and 'Last Purchase Date' in df.columns:
        df_processed = model['preprocessor'].transform(df.drop(['Customer ID', 'Last Purchase Date'], axis=1))

        if st.sidebar.button('Predict High Value Purchases'):
            predictions = model['classifier'].predict(df_processed)
            df['High Value Prediction'] = ['Yes' if x == 1 else 'No' for x in predictions]

            # Split the DataFrame based on the prediction outcome
            df_yes = df[df['High Value Prediction'] == 'Yes'][['Customer ID', 'High Value Prediction']]
            df_no = df[df['High Value Prediction'] == 'No'][['Customer ID', 'High Value Prediction']]

            st.subheader("Prediction Results")

            # Pie Chart for Prediction Distribution
            fig = px.pie(df, names='High Value Prediction', title='Prediction Distribution',
                         color='High Value Prediction', color_discrete_map=color_map)
            st.plotly_chart(fig)

            # Calculate the percentages
            yes_count = sum(df['High Value Prediction'] == 'Yes')
            no_count = sum(df['High Value Prediction'] == 'No')
            total_count = len(df)
            yes_percentage = (yes_count / total_count) * 100
            no_percentage = (no_count / total_count) * 100

            # Using columns to display results side by side
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"High Value Customers: {yes_count}")
                st.dataframe(df_yes)

            with col2:
                st.subheader(f"Other Customers: {no_count}")
                st.dataframe(df_no)


            # Analysis Text
            st.subheader("Analysis of Predictions")
            st.write(f"Out of {total_count} customers, {yes_count} ({yes_percentage:.2f}%) are predicted to make high-value purchases.")
            st.write(f"This indicates that approximately {yes_percentage:.2f}% of the uploaded customer dataset could be prioritized for exclusive offers or loyalty programs to enhance customer engagement and retention.")
            st.write("Conversely, strategies to increase engagement or upsell might be needed for the remaining customers who are less likely to make high-value purchases.")


            # Scatter plot
            fig = px.scatter(df, x='Purchase Frequency', y='Annual Spend', color='High Value Prediction',
                            title='Annual Spend vs. Purchase Frequency by Prediction Outcome',
                            color_discrete_map=color_map,
                            labels={'Purchase Frequency': 'Number of Purchases per Year', 'Annual Spend': 'Total Spend per Year'})
            st.plotly_chart(fig)

            with st.expander("See detailed analysis of Spend vs. Frequency"):
                st.markdown("""
                #### Spend vs. Frequency Analysis
                - **What You See**: This scatter plot displays each customer's total annual spend against their number of purchases per year, colored by whether they are predicted to be high-value customers.
                - **Interpreting the Plot**:
                    - **Data Points**: Each point represents a customer, positioned by how often they buy and how much they spend in total.
                    - **Color Coding**: Customers predicted as high-value are differentiated from others, helping identify spending patterns associated with higher value.
                - **Key Observations**:
                    - Customers with fewer purchases but higher spend may indicate larger transaction sizes.
                    - Alternatively, customers with many small purchases might not always lead to high annual spends but could be nurtured to increase their transaction sizes.
                - **Business Insights**:
                    - **Marketing Strategies**: Develop targeted campaigns that encourage frequent purchasers to increase their spend per transaction.
                    - **Customer Segmentation**: Use these insights to segment customers more effectively, potentially creating personalized offers based on their purchasing frequency and spend.
                    - **Resource Allocation**: Prioritize engagement initiatives that align with the potential to increase overall spend among currently lower-value but frequent customers.
                """)


            # Box Plot for Age
            fig = px.box(df, x='High Value Prediction', y='Age', color='High Value Prediction', color_discrete_map=color_map,
                         title='Age Distribution by Prediction Outcome')
            st.plotly_chart(fig)
            
            with st.expander("See detailed analysis of Age Distribution"):
                st.markdown("""
                #### Age Distribution Analysis
                - **What You See**: This box plot displays the age ranges for customers based on their predicted purchasing behavior.
                - **Interpreting the Plot**:
                    - The box represents the middle 50% of ages for each group.
                    - The line inside each box shows the median age.
                    - Whiskers extend to the highest and lowest ages, excluding outliers.
                - **Key Observations**:
                    - If the median age for 'Yes' is higher, it suggests that older customers are more likely to make high-value purchases.
                    - A wider box in one category indicates more variability in age within that group.
                    - Outliers are ages that stand out because they are unusually high or low.
                - **Business Insights**:
                    - Tailor marketing strategies to target the predominant age groups identified as likely to make high-value purchases.
                    - Use this data to refine customer engagement strategies, particularly by focusing on age groups that might require more attention to increase their spending.
                    - Evaluate the effectiveness of targeted promotions or product offerings based on the age profile of your customers.
                """)
            # Bar Chart for Location
            fig = px.bar(df, x='Location', color='High Value Prediction', title='High Value Purchase Predictions by Location', color_discrete_map=color_map,
                         labels={'count':'Number of Predictions'}, barmode='group')
            st.plotly_chart(fig)
            
            with st.expander("See detailed analysis of Location-based Predictions"):
                st.markdown("""
                #### Location-based Predictions Analysis
                - **What You See**: This bar chart displays the number of customers predicted to make high-value purchases versus those who are not, segmented by location.
                - **Interpreting the Plot**:
                    - **Distribution Across Locations**: The bars represent the count of predicted outcomes, enabling a visual comparison across different geographic areas.
                    - **Color Coding**: 'Yes' predictions are distinguished from 'No' predictions, providing immediate insight into regional performance.
                - **Key Observations**:
                    - **Regional Variations**: Certain locations may exhibit a higher number of 'Yes' predictions, which could be indicative of successful market penetration or greater purchasing power.
                    - **Potential Market Gaps**: Locations with a predominant number of 'No' predictions may represent untapped or underperforming markets.
                - **Business Insights**:
                    - **Targeted Marketing Initiatives**: Increase marketing efforts in regions showing potential for growth or reinforce strategies in high-performing areas.
                    - **Resource Optimization**: Allocate resources more effectively by focusing on locations with the highest return on investment.
                    - **Strategic Expansion**: Consider expansion or increased focus in areas that are currently underperforming but have potential for high-value customer acquisition.
                """)

            # Correlation Heatmap
            correlation_matrix = df.drop(['Customer ID'], axis=1).select_dtypes(include=['number']).corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap=custom_cmap, center=0, vmin=-1, vmax=1)
            # Set the title for the heatmap with Plotly style formatting
            title = 'Correlation Heatmap of Customer Features'
            plt.text(-0.5, 1.1, title, fontsize=14.5, fontweight='bold', va='top', ha='left', transform=plt.gca().transAxes)
            # Adjust layout to add more top padding
            plt.subplots_adjust(top=0.85)  # Decrease this value to increase the top padding

            # Display the plot in Streamlit
            st.pyplot(plt)
            with st.expander("See detailed analysis of Correlation Heatmap"):
                st.markdown("""
                #### Correlation Heatmap Analysis
                - **What You See**: This heatmap displays the strength and direction of relationships between different numeric features in our dataset.
                - **Interpreting the Heatmap**:
                    - **Colors**: Warm colors (e.g., red, orange) indicate a positive correlation, while cool colors (e.g., green) indicate a negative correlation.
                    - **Intensity**: The intensity of the color shows the strength of the correlation. Stronger intensities mean stronger relationships.
                    - **Annotations**: Each cell is annotated with a correlation coefficient, ranging from -1 to 1. Values closer to 1 or -1 indicate strong correlations, while values near 0 indicate weak or no correlation.
                - **Key Observations**:
                    - **Positive Correlations**: Features that move in the same direction. For example, if annual spend and purchase frequency are strongly positive, it suggests that as one increases, so does the other.
                    - **Negative Correlations**: Features that move in opposite directions. For example, if age and engagement score are negatively correlated, older customers might engage less with certain types of marketing.
                - **Business Insights**:
                    - **Targeted Strategies**: Use insights from strong positive correlations to reinforce behaviors that drive high-value purchases.
                    - **Addressing Gaps**: Negative correlations might highlight opportunities to improve engagement or sales strategies for certain demographics or customer behaviors.
                    - **Model Refinement**: Identifying key correlating factors can help in refining the predictive model for better accuracy and effectiveness.
                """)

