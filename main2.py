import streamlit as st
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif
import io

st.set_page_config(page_title="Airline Satisfaction SVM", layout="wide")

# Initialize session state
if 'train_df' not in st.session_state:
    st.session_state.train_df = None
    st.session_state.test_df = None
    st.session_state.X_train_pca = None
    st.session_state.X_test_pca = None
    st.session_state.y_train = None
    st.session_state.y_test = None
    st.session_state.model = None
    st.session_state.cleaned_train = None
    st.session_state.cleaned_test = None
    st.session_state.selected_features = None

# Helper functions
def preprocess_data(df, missing_strategy='mean', missing_value=None, outlier_strategy='remove', outlier_threshold=3):
    df = df.copy()
    df.drop(columns=[col for col in df.columns if "Unnamed" in col], inplace=True)

    # Handle missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        if missing_strategy == 'delete':
            df = df.dropna()
        elif missing_strategy == 'mean':
            for col in missing_cols:
                if df[col].dtype != 'object':
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
        elif missing_strategy == 'median':
            for col in missing_cols:
                if df[col].dtype != 'object':
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
        elif missing_strategy == 'specific':
            for col in missing_cols:
                df[col] = df[col].fillna(missing_value)

    # Encode categorical columns
    for col in df.select_dtypes(include='object'):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Handle outliers
    numeric_cols = df.select_dtypes(include=np.number).columns
    outliers_info = {}
    
    if outlier_strategy != 'ignore':
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(df[col]))
            outliers = df[z_scores > outlier_threshold]
            outliers_info[col] = len(outliers)
            
            if outlier_strategy == 'remove':
                df = df[z_scores <= outlier_threshold]
            elif outlier_strategy == 'cap':
                lower = df[col].mean() - outlier_threshold * df[col].std()
                upper = df[col].mean() + outlier_threshold * df[col].std()
                df[col] = np.where(df[col] > upper, upper, 
                                  np.where(df[col] < lower, lower, df[col]))

    return df, missing_cols, outliers_info

def plot_boxplots(df, title):
    numeric_cols = df.select_dtypes(include=np.number).columns
    num_plots = len(numeric_cols)
    cols_per_row = 3
    rows = (num_plots + cols_per_row - 1) // cols_per_row
    
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 5*rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(col)
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle(title, y=1.02)
    st.pyplot(fig)

# Sidebar Navigation
page = st.sidebar.radio("Navigation", [
    "1. Upload Data",
    "2. EDA",
    "3. Data Cleaning",
    "4. Dimensionality Reduction",
    "5. Model Building",
    "6. Evaluation"
])

if page == "1. Upload Data":
    st.title("Upload Train and Test CSV Files")
    train_file = st.file_uploader("Upload train.csv", type=["csv"])
    test_file = st.file_uploader("Upload test.csv", type=["csv"])

    if train_file and test_file:
        st.session_state.train_df = pd.read_csv(train_file)
        st.session_state.test_df = pd.read_csv(test_file)
        st.success("Files uploaded successfully!")
        
        st.subheader("Train Set Preview")
        st.dataframe(st.session_state.train_df.head())
        
        st.subheader("Test Set Preview")
        st.dataframe(st.session_state.test_df.head())

elif page == "2. EDA":
    st.title("Exploratory Data Analysis")
    if st.session_state.train_df is not None:
        df = st.session_state.train_df
        
        st.subheader("Dataset Information")
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")
        
        # Show data types
        dtype_info = pd.DataFrame(df.dtypes, columns=['Data Type'])
        st.write("Data Types:", dtype_info)

        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()

        st.subheader("Data Info")
        st.text(info_str)
        
        st.subheader("Summary Statistics")
        st.write(df.describe(include='all'))
        
        st.subheader("Target Variable Distribution")
        fig = px.histogram(df, x='satisfaction', color='satisfaction', 
                          title='Distribution of Satisfaction')
        st.plotly_chart(fig)
        
        st.subheader("Feature Distributions")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        st.subheader("Feature Distributions")

        # Get numeric columns (exclude object/string columns)
        numeric_cols = df.select_dtypes(include=[np.number, 'int64', 'float64']).columns.tolist()

        # Set safe defaults - skip if there aren't enough columns
        default_selected = []
        if len(numeric_cols) >= 4:
            default_selected = numeric_cols[2:4]  # 3rd and 4th columns
        elif len(numeric_cols) >= 1:
            default_selected = numeric_cols[:1]  # Just first column if <4 available

        selected = st.multiselect(
            "Select numeric columns to plot",
            options=numeric_cols,
            default=default_selected
        )

        # Plot selected features
        for col in selected:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f'Distribution of {col}', pad=20)
            st.pyplot(fig)

        st.subheader("Interactive Correlation Matrix")
        numeric_df = df.select_dtypes(include=['number'])
        corr_matrix = numeric_df.corr()

        # Create hover text
        hover_text = []
        for yi, yy in enumerate(corr_matrix.columns):
            hover_row = []
            for xi, xx in enumerate(corr_matrix.columns):
                hover_row.append(f"{xx} vs {yy}: {corr_matrix.iloc[yi, xi]:.2f}")
            hover_text.append(hover_row)

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.columns.tolist(),
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            hoverongaps=True,
            hoverinfo='text',
            text=hover_text
        ))

        fig.update_layout(
            title='Correlation Matrix',
            width=800,
            height=800,
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            yaxis_autorange='reversed'
        )

        st.plotly_chart(fig)

        
        st.subheader("Feature Relationships (Pair Plot)")
        pairplot_cols = st.multiselect("Select columns for pair plot (max 5)", 
                                    numeric_df.columns, default=numeric_df.columns[:3])

        if len(pairplot_cols) > 5:
            st.warning("Please select no more than 5 columns for performance reasons.")
        elif len(pairplot_cols) > 1:
            try:
                fig = px.scatter_matrix(df, 
                                    dimensions=pairplot_cols, 
                                    color='satisfaction',
                                    title="Feature Relationships Pair Plot")
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error creating pair plot: {str(e)}")
                # Fallback to regular scatter plots if pair plot fails
                st.info("Showing individual scatter plots instead")
                for i in range(len(pairplot_cols)):
                    for j in range(i+1, len(pairplot_cols)):
                        fig = px.scatter(df, 
                                    x=pairplot_cols[i], 
                                    y=pairplot_cols[j], 
                                    color='satisfaction',
                                    title=f"{pairplot_cols[i]} vs {pairplot_cols[j]}")
                        st.plotly_chart(fig)
        else:
            st.warning("Please select at least 2 columns for pair plot")

    else:
        st.warning("Please upload data first.")

elif page == "3. Data Cleaning":
    st.title("Data Cleaning and Outlier Handling")
    if st.session_state.train_df is not None:
        df = st.session_state.train_df
        
        st.subheader("Missing Values Analysis")
        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        
        if not missing_values.empty:
            st.write("Columns with missing values:")
            st.dataframe(missing_values.rename('Missing Values'))
            st.write(f"Total columns with missing values: {len(missing_values)}")
            
            st.subheader("Missing Value Treatment")
            missing_strategy = st.radio("How would you like to handle missing values?",
                                      ['delete', 'mean', 'median', 'specific'])
            
            missing_value = None
            if missing_strategy == 'specific':
                missing_value = st.text_input("Enter the value to fill missing values with")
            
            st.subheader("Outlier Analysis")
            numeric_cols = df.select_dtypes(include=np.number).columns
            z_scores = np.abs(stats.zscore(df[numeric_cols]))
            outliers = (z_scores > 3).sum(axis=0)
            outliers = outliers[outliers > 0]
            
            if not outliers.empty:
                st.write("Outliers detected in columns:")
                st.dataframe(outliers.rename('Outlier Count'))
                st.write(f"Total outliers detected: {outliers.sum()}")
                
                st.subheader("Outlier Visualization (Before Treatment)")
                plot_boxplots(df, "Boxplots Before Outlier Treatment")
                
                st.subheader("Outlier Treatment")
                outlier_strategy = st.radio("How would you like to handle outliers?",
                                           ['ignore', 'remove', 'cap'])
                
                outlier_threshold = st.slider("Select z-score threshold for outliers", 
                                             min_value=1.0, max_value=5.0, value=3.0, step=0.5)
                
                # Process data with selected strategies
                cleaned_train, missing_cols, outliers_info = preprocess_data(
                    st.session_state.train_df,
                    missing_strategy=missing_strategy,
                    missing_value=missing_value,
                    outlier_strategy=outlier_strategy,
                    outlier_threshold=outlier_threshold
                )
                
                # Process test data with same strategies (except deletion)
                cleaned_test, _, _ = preprocess_data(
                    st.session_state.test_df,
                    missing_strategy='mean' if missing_strategy == 'delete' else missing_strategy,
                    missing_value=missing_value,
                    outlier_strategy='ignore'  # Don't remove outliers from test set
                )
                
                st.success("Data cleaning completed!")
                st.write("New train shape:", cleaned_train.shape)
                
                if outlier_strategy != 'ignore':
                    st.subheader("Outlier Visualization (After Treatment)")
                    plot_boxplots(cleaned_train, "Boxplots After Outlier Treatment")
                
                st.session_state.cleaned_train = cleaned_train
                st.session_state.cleaned_test = cleaned_test
            else:
                st.info("No outliers detected in the data.")
        else:
            st.info("No missing values found in the dataset.")
    else:
        st.warning("Please upload data first.")

elif page == "4. Dimensionality Reduction":
    st.title("Feature Selection and Dimensionality Reduction")
    if 'cleaned_train' in st.session_state:
        df_train = st.session_state.cleaned_train
        df_test = st.session_state.cleaned_test
        
        st.subheader("Feature-Target Correlation")
        numeric_cols = df_train.select_dtypes(include=np.number).columns
        corr_with_target = df_train[numeric_cols].corr()['satisfaction'].abs().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        corr_with_target.drop('satisfaction').plot(kind='bar', ax=ax)
        ax.set_title("Feature Correlation with Target")
        ax.set_ylabel("Absolute Correlation Coefficient")
        st.pyplot(fig)
        
        st.subheader("Feature Selection")
        corr_threshold = st.slider("Select correlation threshold for feature selection", 
                                  min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        
        selected_features = corr_with_target[corr_with_target > corr_threshold].index.tolist()
        if 'satisfaction' in selected_features:
            selected_features.remove('satisfaction')
        
        st.write(f"Selected {len(selected_features)} features with correlation > {corr_threshold}")
        st.write("Selected features:", selected_features)
        
        st.session_state.selected_features = selected_features
        
        st.subheader("Interactive Scatter Plot")
        if len(selected_features) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis feature", selected_features, index=0)
            with col2:
                y_axis = st.selectbox("Y-axis feature", selected_features, index=1)
            
            fig = px.scatter(df_train, x=x_axis, y=y_axis, color='satisfaction',
                             title=f"{x_axis} vs {y_axis} colored by Satisfaction")
            st.plotly_chart(fig)
        
        st.subheader("PCA Dimensionality Reduction")
        X_train = df_train[selected_features]
        y_train = df_train['satisfaction']
        X_test = df_test[selected_features]
        y_test = df_test['satisfaction']
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        pca = PCA()
        pca.fit(X_train_scaled)
        
        fig, ax = plt.subplots()
        ax.plot(np.cumsum(pca.explained_variance_ratio_))
        ax.axhline(y=0.95, color='r', linestyle='--')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.set_title('PCA Explained Variance')
        st.pyplot(fig)
        
        n_components = st.slider("Select number of PCA components", 
                                min_value=1, max_value=len(selected_features), 
                                value=min(10, len(selected_features)))
        
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        st.session_state.X_train_pca = X_train_pca
        st.session_state.X_test_pca = X_test_pca
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        
        st.success(f"PCA applied: Reduced from {X_train.shape[1]} to {X_train_pca.shape[1]} components")
        
    else:
        st.warning("Please complete data cleaning first.")

elif page == "5. Model Building":
    st.title("SVM Model Training")
    if st.session_state.X_train_pca is not None:
        # First ensure all data is properly encoded
        if not all(isinstance(x, (int, float)) for x in st.session_state.y_train[:1]):
            # Safely encode labels if needed
            le = LabelEncoder()
            y_train = le.fit_transform(st.session_state.y_train)
        else:
            y_train = st.session_state.y_train
            
        # Take smaller samples
        sample_size = min(20000, len(st.session_state.X_train_pca))
        X_train = st.session_state.X_train_pca[:sample_size]
        y_train = y_train[:sample_size]

        st.write(f"Training SVM on {sample_size} samples...")

        # Simplified parameter grid for faster training
        param_grid = {
            'C': [10],
            'kernel': ['rbf'],
            'gamma': ['scale']
        }

        # Add progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        grid = GridSearchCV(
            SVC(probability=True),
            param_grid,
            cv=3,
            n_jobs=-1,
            verbose=1
        )

        # Simulate progress
        for i in range(10):
            time.sleep(0.2)
            progress_bar.progress((i + 1) * 10)
            status_text.text(f"Preparing... {10*(i+1)}%")

        # Actual training
        with st.spinner('Training in progress...'):
            grid.fit(X_train, y_train)

        progress_bar.empty()
        status_text.empty()

        st.session_state.model = grid.best_estimator_
        st.success("Training complete!")
        st.write(f"Best parameters: {grid.best_params_}")
        st.write(f"Best score: {grid.best_score_:.4f}")
        
        # Prepare test samples
        if hasattr(st.session_state, 'X_test_pca'):
            test_size = min(5000, len(st.session_state.X_test_pca))
            st.session_state.X_test_eval = st.session_state.X_test_pca[:test_size]
            st.session_state.y_test_eval = st.session_state.y_test[:test_size]
    else:
        st.warning("Please complete dimensionality reduction first.")

elif page == "6. Evaluation":
    st.title("Model Evaluation")
    if st.session_state.model is not None:
        model = st.session_state.model
        X_test = st.session_state.X_test_pca
        y_test = st.session_state.y_test
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        st.subheader("Model Performance Metrics")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, 
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['Neutral/Dissatisfied', 'Satisfied'],
                        y=['Neutral/Dissatisfied', 'Satisfied'],
                        text_auto=True)
        fig.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig)
        
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0))
        
        st.subheader("Probability Distribution")
        fig = px.histogram(x=y_proba, color=y_test, nbins=50,
                          labels={'x': 'Predicted Probability', 'y': 'Count'},
                          title='Predicted Probability Distribution')
        st.plotly_chart(fig)
    else:
        st.warning("Please train the model first.")

