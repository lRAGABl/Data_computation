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

# Single target encoder for consistency
if 'target_le' not in st.session_state:
    st.session_state.target_le = LabelEncoder()

# Helper functions
def preprocess_data(df, missing_strategy='mean', missing_value=None, outlier_strategy='remove', outlier_threshold=3):
    df = df.copy()
    # Drop unnamed columns
    df.drop(columns=[col for col in df.columns if "Unnamed" in col], inplace=True, errors='ignore')

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

    # Encode categorical columns (except 'satisfaction')
    for col in df.select_dtypes(include='object'):
        if col == 'satisfaction':
            continue  # Skip target column
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Handle outliers (excluding 'satisfaction')
    numeric_cols = df.select_dtypes(include=np.number).columns.drop('satisfaction', errors='ignore')
    outliers_info = {}
    
    if outlier_strategy != 'ignore':
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))
            outlier_mask = z_scores > outlier_threshold
            outliers_info[col] = outlier_mask.sum()
            
            if outlier_strategy == 'remove':
                df = df[~outlier_mask]
            elif outlier_strategy == 'cap':
                lower = df[col].mean() - outlier_threshold * df[col].std()
                upper = df[col].mean() + outlier_threshold * df[col].std()
                df[col] = np.clip(df[col], lower, upper)
            elif outlier_strategy == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                df[col] = np.clip(df[col], lower, upper)

    return df, missing_cols, outliers_info

@st.cache_data
def encoded_datasets(train_df, test_df, cleaning_params):
    """Process and encode both datasets with consistent target encoding"""
    # Clean both datasets
    cleaned_train, _, _ = preprocess_data(train_df, **cleaning_params)
    
    # For test data, use mean for missing values (don't delete rows)
    test_params = cleaning_params.copy()
    if test_params.get('missing_strategy') == 'delete':
        test_params['missing_strategy'] = 'mean'
    test_params['outlier_strategy'] = 'ignore'  # Don't remove test outliers
    
    cleaned_test, _, _ = preprocess_data(test_df, **test_params)
    
    # Make sure both have the same columns
    common_cols = cleaned_train.columns.intersection(cleaned_test.columns)
    cleaned_train = cleaned_train[common_cols]
    cleaned_test = cleaned_test[common_cols]
    
    # Encode target consistently
    le = st.session_state.target_le
    if 'satisfaction' in cleaned_train.columns:
        cleaned_train['satisfaction'] = le.fit_transform(cleaned_train['satisfaction'].astype(str))
        cleaned_test['satisfaction'] = le.transform(cleaned_test['satisfaction'].astype(str))
    
    return cleaned_train, cleaned_test

def plot_boxplots(df, title):
    numeric_cols = df.select_dtypes(include=np.number).columns.drop('satisfaction', errors='ignore')
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

@st.cache_resource
def train_svm_model(X, y, param_grid):
    """Train SVM with GridSearch and cache the result"""
    grid = GridSearchCV(
        SVC(probability=True),
        param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_, grid.best_score_

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
        numeric_cols = df.select_dtypes(include=np.number).columns.drop('satisfaction', errors='ignore')

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
        numeric_df = df.select_dtypes(include=['number']).drop('satisfaction', axis=1, errors='ignore')
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
        else:
            st.info("No missing values found in the dataset.")
            missing_strategy = 'mean'  # Default
            missing_value = None
            
        st.subheader("Outlier Analysis")
        numeric_cols = df.select_dtypes(include=np.number).columns.drop('satisfaction', errors='ignore')
        
        try:
            z_scores = np.abs(stats.zscore(df[numeric_cols], nan_policy='omit'))
            
            # Convert to Series with column names
            outlier_counts = pd.Series((z_scores > 3).sum(axis=0), index=numeric_cols)
            outliers = outlier_counts[outlier_counts > 0]
            
            if not outliers.empty:
                st.write("Outliers detected in columns:")
                st.dataframe(outliers.rename('Outlier Count'))
                st.write(f"Total outliers detected: {outliers.sum()}")
                
                st.subheader("Outlier Visualization (Before Treatment)")
                plot_boxplots(df, "Boxplots Before Outlier Treatment")
                
                st.subheader("Outlier Treatment")
                outlier_strategy = st.radio("How would you like to handle outliers?",
                                           ['ignore', 'remove', 'cap', 'iqr'])
                
                outlier_threshold = st.slider("Select z-score threshold for outliers", 
                                             min_value=1.0, max_value=5.0, value=3.0, step=0.5)
            else:
                st.info("No outliers detected in the data.")
                outlier_strategy = 'ignore'
                outlier_threshold = 3.0
                
        except Exception as e:
            st.error(f"Error analyzing outliers: {str(e)}")
            outlier_strategy = 'ignore'
            outlier_threshold = 3.0
        
        if st.button("Apply Cleaning"):
            # Create parameters dictionary
            cleaning_params = {
                'missing_strategy': missing_strategy,
                'missing_value': missing_value,
                'outlier_strategy': outlier_strategy,
                'outlier_threshold': outlier_threshold
            }
            
            # Process both datasets with caching
            with st.spinner("Cleaning data..."):
                cleaned_train, cleaned_test = encoded_datasets(
                    st.session_state.train_df, 
                    st.session_state.test_df, 
                    cleaning_params
                )
            
            st.session_state.cleaned_train = cleaned_train
            st.session_state.cleaned_test = cleaned_test
            
            st.success("Data cleaning completed!")
            st.write("New train shape:", cleaned_train.shape)
            
            if outlier_strategy != 'ignore':
                st.subheader("Outlier Visualization (After Treatment)")
                try:
                    plot_boxplots(cleaned_train, "Boxplots After Outlier Treatment")
                except Exception as e:
                    st.error(f"Error plotting boxplots after cleaning: {str(e)}")
    else:
        st.warning("Please upload data first.")

elif page == "4. Dimensionality Reduction":
    st.title("Feature Selection and Dimensionality Reduction")
    if st.session_state.cleaned_train is not None:
        df_train = st.session_state.cleaned_train
        df_test = st.session_state.cleaned_test
        
        # First check if the satisfaction column exists
        if 'satisfaction' not in df_train.columns:
            st.error("The 'satisfaction' column is missing from the cleaned data. Please go back to the data cleaning step.")
            st.stop()
        
        st.subheader("Feature-Target Correlation")
        # Make sure to include all numeric columns INCLUDING 'satisfaction'
        all_numeric_cols = df_train.select_dtypes(include=np.number).columns
        
        # Calculate correlations including 'satisfaction'
        correlation_matrix = df_train[all_numeric_cols].corr()
        
        # Check if 'satisfaction' is in the correlation matrix
        if 'satisfaction' in correlation_matrix.columns:
            corr_with_target = correlation_matrix['satisfaction'].abs().sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            # Drop satisfaction from the plot (not from the data)
            if 'satisfaction' in corr_with_target.index:
                corr_with_target = corr_with_target.drop('satisfaction')
            corr_with_target.plot(kind='bar', ax=ax)
            ax.set_title("Feature Correlation with Target")
            ax.set_ylabel("Absolute Correlation Coefficient")
            st.pyplot(fig)
            
            st.subheader("Feature Selection")
            corr_threshold = st.slider("Select correlation threshold for feature selection", 
                                    min_value=0.0, max_value=1.0, value=0.1, step=0.01)
            
            selected_features = corr_with_target[corr_with_target > corr_threshold].index.tolist()
            
            st.write(f"Selected {len(selected_features)} features with correlation > {corr_threshold}")
            st.write("Selected features:", selected_features)
            
            st.session_state.selected_features = selected_features
            
            # Rest of the code...
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
                                    min_value=1, 
                                    max_value=min(len(selected_features), X_train.shape[1]), 
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
            st.error("'satisfaction' column is not available in the correlation matrix. Check data cleaning steps.")
    else:
        st.warning("Please complete data cleaning first.")
elif page == "5. Model Building":
    st.title("SVM Model Training")
    if st.session_state.X_train_pca is not None:
        # Determine sample size for training (limit to avoid memory issues)
        sample_size = min(20000, len(st.session_state.X_train_pca))
        X_train = st.session_state.X_train_pca[:sample_size]
        y_train = st.session_state.y_train[:sample_size]

        st.write(f"Training SVM on {sample_size} samples...")
        
        # Parameter selection
        col1, col2 = st.columns(2)
        with col1:
            C = st.slider("C parameter (regularization)", 0.1, 20.0, 10.0, 0.1)
            kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
        with col2:
            gamma = st.selectbox("Gamma parameter", ["scale", "auto"], index=0)
            degree = st.slider("Degree (for poly kernel)", 2, 5, 3) if kernel == "poly" else 3

        param_grid = {
            'C': [C],
            'kernel': [kernel],
            'gamma': [gamma]
        }
        
        if kernel == "poly":
            param_grid['degree'] = [degree]

        if st.button("Train Model"):
            # Add progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate initial progress
            for i in range(10):
                time.sleep(0.1)
                progress_bar.progress((i + 1) * 10)
                status_text.text(f"Preparing... {10*(i+1)}%")

            # Actual training
            with st.spinner('Training SVM...'):
                try:
                    model, best_params, best_score = train_svm_model(
                        X_train, y_train, param_grid
                    )
                    st.session_state.model = model
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success("Training complete!")
                    st.write(f"Best parameters: {best_params}")
                    st.write(f"Cross-validation score: {best_score:.4f}")
                    
                    # Prepare test samples for evaluation
                    test_size = min(5000, len(st.session_state.X_test_pca))
                    st.session_state.X_test_eval = st.session_state.X_test_pca[:test_size]
                    st.session_state.y_test_eval = st.session_state.y_test[:test_size]
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()
    else:
        st.warning("Please complete dimensionality reduction first.")

elif page == "6. Evaluation":
    st.title("Model Evaluation")
    if st.session_state.model is not None:
        model = st.session_state.model
        
        # Use cached test data if available, otherwise use the full test set
        if hasattr(st.session_state, 'X_test_eval'):
            X_test = st.session_state.X_test_eval
            y_test = st.session_state.y_test_eval
        else:
            X_test = st.session_state.X_test_pca
            y_test = st.session_state.y_test
            
        # Limit size for performance
        eval_size = min(5000, len(X_test))
        X_test = X_test[:eval_size]
        y_test = y_test[:eval_size]
        
        with st.spinner("Generating predictions..."):
            y_pred = model.predict(X_test)
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                has_proba = True
            except:
                has_proba = False
        
        st.subheader("Model Performance Metrics")
        accuracy = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{accuracy:.4f}")
        
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        
        # Get class labels from the encoder
        le = st.session_state.target_le
        class_names = le.classes_ if hasattr(le, 'classes_') else ['Negative', 'Positive']
        
        fig = px.imshow(cm, 
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=class_names,
                        y=class_names,
                        text_auto=True)
        fig.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig)
        
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0))
        
        if has_proba:
            st.subheader("Probability Distribution")
            fig = px.histogram(x=y_proba, color=y_test, nbins=50,
                              labels={'x': 'Predicted Probability', 'y': 'Count'},
                              title='Predicted Probability Distribution by Class',
                              color_discrete_map={0: 'blue', 1: 'red'})
            st.plotly_chart(fig)
            
            # ROC curve
            from sklearn.metrics import roc_curve, roc_auc_score
            
            try:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc = roc_auc_score(y_test, y_proba)
                
                fig = px.line(
                    x=fpr, y=tpr,
                    labels={"x": "False Positive Rate", "y": "True Positive Rate"},
                    title=f"ROC Curve (AUC = {auc:.4f})"
                )
                
                # Add diagonal reference line
                fig.add_shape(
                    type='line',
                    line=dict(dash='dash', width=1, color='gray'),
                    x0=0, y0=0, x1=1, y1=1
                )
                
                st.plotly_chart(fig)
            except Exception as e:
                st.warning(f"Could not generate ROC curve: {str(e)}")
        
        # Feature importance (for linear kernel only)
        if hasattr(model, 'coef_') and st.session_state.selected_features is not None:
            st.subheader("Feature Importance")
            try:
                importances = np.abs(model.coef_[0])
                feature_names = np.array(st.session_state.selected_features)
                indices = np.argsort(importances)[::-1]
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.barh(range(len(indices[:15])), importances[indices[:15]])
                ax.set_yticks(range(len(indices[:15])))
                ax.set_yticklabels(feature_names[indices[:15]])
                ax.set_title('Feature Importance (Linear Kernel)')
                st.pyplot(fig)
            except Exception as e:
                st.info("Feature importance is only available for linear kernel.")
    else:
        st.warning("Please train the model first.")
