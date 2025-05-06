import streamlit as st
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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from scipy import stats
import io

# Set page config
st.set_page_config(
    page_title="Airline Satisfaction SVM",
    layout="wide",
    initial_sidebar_state="expanded"
)

def fix_dataframe_types(df):
    """Convert DataFrame columns to Arrow-compatible types"""
    df = df.copy()
    
    # Convert object/string columns
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype('str')
    
    # Convert integer columns
    for col in df.select_dtypes(include=['int64', 'Int64']).columns:
        df[col] = df[col].astype('float64')
    
    # Specific fixes for known problematic columns
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype('float64')
    
    return df

def safe_dataframe_display(df, max_rows=5):
    """Safely display DataFrame with type conversion"""
    display_df = fix_dataframe_types(df.head(max_rows))
    st.dataframe(display_df)

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
    """Preprocess data with missing value and outlier handling"""
    df = fix_dataframe_types(df.copy())
    df.drop(columns=[col for col in df.columns if "Unnamed" in col], inplace=True)

    # Handle missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        if missing_strategy == 'delete':
            df = df.dropna()
        elif missing_strategy == 'mean':
            for col in missing_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
        elif missing_strategy == 'median':
            for col in missing_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
        elif missing_strategy == 'specific':
            for col in missing_cols:
                df[col] = df[col].fillna(missing_value)

    # Encode categorical columns safely
    for col in df.select_dtypes(include='object').columns:
        try:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        except Exception as e:
            st.warning(f"Could not encode column {col}: {str(e)}")
            df[col] = df[col].astype('category').cat.codes

    # Handle outliers using IQR (more robust than Z-score)
    numeric_cols = df.select_dtypes(include=np.number).columns
    outliers_info = {}
    
    if outlier_strategy != 'ignore':
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - outlier_threshold*iqr
            upper_bound = q3 + outlier_threshold*iqr
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outliers_info[col] = len(outliers)
            
            if outlier_strategy == 'remove':
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            elif outlier_strategy == 'cap':
                df[col] = df[col].clip(lower_bound, upper_bound)

    return fix_dataframe_types(df), missing_cols, outliers_info

def plot_boxplots(df, title):
    """Create boxplots for numeric columns"""
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        st.warning("No numeric columns to plot")
        return
    
    num_plots = len(numeric_cols)
    cols_per_row = 3
    rows = (num_plots + cols_per_row - 1) // cols_per_row
    
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 5*rows))
    axes = axes.flatten() if rows > 1 else [axes]
    
    for i, col in enumerate(numeric_cols):
        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(col)
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle(title, y=1.02)
    st.pyplot(fig)

# Sidebar Navigation
pages = [
    "1. Upload Data",
    "2. EDA",
    "3. Data Cleaning",
    "4. Dimensionality Reduction",
    "5. Model Building",
    "6. Evaluation"
]

page = st.sidebar.radio("Navigation", pages, index=0)

# Page 1: Upload Data
if page == "1. Upload Data":
    st.title("📤 Upload Train and Test CSV Files")
    
    with st.expander("ℹ Instructions"):
        st.write("""
        1. Upload your training data (train.csv)
        2. Upload your testing data (test.csv)
        3. Both files should contain a 'satisfaction' column as the target variable
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        train_file = st.file_uploader("Upload train.csv", type=["csv"], key="train_upload")
    with col2:
        test_file = st.file_uploader("Upload test.csv", type=["csv"], key="test_upload")

    if train_file and test_file:
        try:
            st.session_state.train_df = fix_dataframe_types(pd.read_csv(train_file))
            st.session_state.test_df = fix_dataframe_types(pd.read_csv(test_file))
            st.success("Files uploaded successfully!")
            
            st.subheader("Train Set Preview")
            safe_dataframe_display(st.session_state.train_df)
            
            st.subheader("Test Set Preview")
            safe_dataframe_display(st.session_state.test_df)
            
            # Check for target column
            if 'satisfaction' not in st.session_state.train_df.columns:
                st.error("❌ 'satisfaction' column not found in training data!")
            if 'satisfaction' not in st.session_state.test_df.columns:
                st.error("❌ 'satisfaction' column not found in test data!")
                
        except Exception as e:
            st.error(f"Error loading files: {str(e)}")

# Page 2: EDA
elif page == "2. EDA":
    st.title("🔍 Exploratory Data Analysis")
    
    if st.session_state.train_df is not None:
        df = fix_dataframe_types(st.session_state.train_df)
        
        with st.expander("📊 Dataset Overview"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Number of Rows", df.shape[0])
            with col2:
                st.metric("Number of Columns", df.shape[1])
            
            st.subheader("Data Types")
            dtype_info = pd.DataFrame(df.dtypes, columns=['Data Type'])
            st.table(dtype_info)
            
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
        
        with st.expander("📈 Summary Statistics"):
            st.dataframe(df.describe(include='all'))
        
        with st.expander("🎯 Target Variable Analysis"):
            if 'satisfaction' in df.columns:
                fig = px.histogram(df, x='satisfaction', color='satisfaction',
                                 title='Distribution of Satisfaction')
                st.plotly_chart(fig)
            else:
                st.error("Target column 'satisfaction' not found!")
        
        with st.expander("📉 Feature Distributions"):
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            
            if len(numeric_cols) > 0:
                default_cols = numeric_cols[:min(4, len(numeric_cols))]
                selected = st.multiselect(
                    "Select numeric columns to plot",
                    options=numeric_cols,
                    default=default_cols
                )
                
                for col in selected:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.histplot(df[col], kde=True, ax=ax)
                    ax.set_title(f'Distribution of {col}', pad=20)
                    st.pyplot(fig)
            else:
                st.warning("No numeric columns found for distribution plots")
        
        with st.expander("🔄 Correlation Analysis"):
            numeric_df = df.select_dtypes(include=np.number)
            if len(numeric_df.columns) > 0:
                corr_matrix = numeric_df.corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns.tolist(),
                    y=corr_matrix.columns.tolist(),
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1,
                    hoverongaps=True,
                    hoverinfo='text',
                    text=[[f"{y} vs {x}: {corr_matrix.iloc[i,j]:.2f}" 
                          for j, x in enumerate(corr_matrix.columns)] 
                         for i, y in enumerate(corr_matrix.columns)]
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
            else:
                st.warning("No numeric columns for correlation analysis")
        
        with st.expander("🔗 Feature Relationships"):
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) >= 2:
                pairplot_cols = st.multiselect(
                    "Select columns for pair plot (max 5)", 
                    numeric_cols, 
                    default=numeric_cols[:min(3, len(numeric_cols))]
                )
                
                if len(pairplot_cols) > 5:
                    st.warning("Please select no more than 5 columns")
                elif len(pairplot_cols) >= 2:
                    try:
                        fig = px.scatter_matrix(
                            df,
                            dimensions=pairplot_cols,
                            color='satisfaction' if 'satisfaction' in df.columns else None,
                            title="Feature Relationships"
                        )
                        st.plotly_chart(fig)
                    except Exception as e:
                        st.error(f"Could not create pair plot: {str(e)}")
                else:
                    st.warning("Select at least 2 columns")
            else:
                st.warning("Need at least 2 numeric columns for pair plot")
    else:
        st.warning("Please upload data first")

# Page 3: Data Cleaning
elif page == "3. Data Cleaning":
    st.title("🧹 Data Cleaning and Outlier Handling")
    
    if st.session_state.train_df is not None:
        df = fix_dataframe_types(st.session_state.train_df.copy())
        
        with st.expander("🔎 Missing Value Analysis"):
            missing_values = df.isnull().sum()
            missing_values = missing_values[missing_values > 0]
            
            if not missing_values.empty:
                st.write("Columns with missing values:")
                st.dataframe(missing_values.rename('Missing Values'))
                st.write(f"Total columns with missing values: {len(missing_values)}")
                
                st.subheader("Missing Value Treatment")
                missing_strategy = st.radio(
                    "How would you like to handle missing values?",
                    ['delete', 'mean', 'median', 'specific'],
                    horizontal=True
                )
                
                missing_value = None
                if missing_strategy == 'specific':
                    missing_value = st.text_input("Enter the value to fill missing values with")
            else:
                st.info("✅ No missing values found in the dataset.")
        
        with st.expander("📊 Outlier Analysis"):
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                # Using IQR method for outlier detection
                outliers_report = []
                for col in numeric_cols:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5*iqr
                    upper = q3 + 1.5*iqr
                    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                    if n_outliers > 0:
                        outliers_report.append({
                            'Column': col,
                            'Outliers': n_outliers,
                            'Percentage': f"{n_outliers/len(df)*100:.1f}%",
                            'Lower Bound': f"{lower:.2f}",
                            'Upper Bound': f"{upper:.2f}"
                        })
                
                if outliers_report:
                    outliers_df = pd.DataFrame(outliers_report)
                    st.write("Outliers detected in columns:")
                    st.dataframe(outliers_df)
                    st.write(f"Total outliers detected: {outliers_df['Outliers'].sum()}")
                    
                    st.subheader("Outlier Visualization")
                    plot_boxplots(df, "Boxplots Before Outlier Treatment")
                    
                    st.subheader("Outlier Treatment")
                    outlier_strategy = st.radio(
                        "How would you like to handle outliers?",
                        ['ignore', 'remove', 'cap'],
                        horizontal=True
                    )
                    
                    iqr_multiplier = st.slider(
                        "IQR multiplier for outlier boundaries",
                        min_value=1.0,
                        max_value=5.0,
                        value=1.5,
                        step=0.1
                    )
                else:
                    st.info("✅ No significant outliers detected using IQR method.")
                    outlier_strategy = 'ignore'
            else:
                st.warning("No numeric columns for outlier analysis")
                outlier_strategy = 'ignore'
        
        # Process data when user clicks the button
        if st.button("Apply Data Cleaning"):
            with st.spinner("Cleaning data..."):
                try:
                    # Process train data
                    cleaned_train, _, _ = preprocess_data(
                        st.session_state.train_df,
                        missing_strategy=missing_strategy,
                        missing_value=missing_value,
                        outlier_strategy=outlier_strategy,
                        outlier_threshold=iqr_multiplier
                    )
                    
                    # Process test data (without outlier removal)
                    cleaned_test, _, _ = preprocess_data(
                        st.session_state.test_df,
                        missing_strategy='mean' if missing_strategy == 'delete' else missing_strategy,
                        missing_value=missing_value,
                        outlier_strategy='ignore'
                    )
                    
                    st.session_state.cleaned_train = cleaned_train
                    st.session_state.cleaned_test = cleaned_test
                    
                    st.success("Data cleaning completed!")
                    st.write("New train shape:", cleaned_train.shape)
                    
                    if outlier_strategy != 'ignore':
                        with st.expander("📈 Post-Treatment Visualization"):
                            plot_boxplots(cleaned_train, "Boxplots After Outlier Treatment")
                
                except Exception as e:
                    st.error(f"Error during data cleaning: {str(e)}")
    else:
        st.warning("Please upload data first")

# Page 4: Dimensionality Reduction
elif page == "4. Dimensionality Reduction":
    st.title("📉 Dimensionality Reduction")
    
    if 'cleaned_train' in st.session_state:
        df_train = fix_dataframe_types(st.session_state.cleaned_train)
        df_test = fix_dataframe_types(st.session_state.cleaned_test)
        
        with st.expander("📈 Feature-Target Correlation"):
            if 'satisfaction' not in df_train.columns:
                st.error("Target column 'satisfaction' not found!")
            else:
                numeric_cols = df_train.select_dtypes(include=np.number).columns
                if len(numeric_cols) > 0:
                    try:
                        corr_matrix = df_train[numeric_cols].corr()
                        if 'satisfaction' in corr_matrix.columns:
                            corr_with_target = corr_matrix['satisfaction'].abs().sort_values(ascending=False)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            corr_with_target.drop('satisfaction', errors='ignore').plot(kind='barh', ax=ax)
                            ax.set_title("Feature Correlation with Target")
                            ax.set_xlabel("Absolute Correlation Coefficient")
                            st.pyplot(fig)
                            
                            st.dataframe(
                                corr_with_target.to_frame("Correlation")
                                .sort_values("Correlation", ascending=False)
                                .style.background_gradient(cmap='viridis')
                            )
                        else:
                            st.warning("Could not calculate correlations with target")
                    except Exception as e:
                        st.error(f"Correlation calculation failed: {str(e)}")
                else:
                    st.warning("No numeric columns for correlation analysis")
        
        with st.expander("🔍 Feature Selection"):
            if 'satisfaction' in df_train.columns:
                numeric_cols = df_train.select_dtypes(include=np.number).columns
                corr_matrix = df_train[numeric_cols].corr()
                
                if 'satisfaction' in corr_matrix.columns:
                    corr_with_target = corr_matrix['satisfaction'].abs().drop('satisfaction', errors='ignore')
                    
                    corr_threshold = st.slider(
                        "Select correlation threshold for feature selection",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.1,
                        step=0.01
                    )
                    
                    selected_features = corr_with_target[corr_with_target > corr_threshold].index.tolist()
                    
                    if len(selected_features) > 0:
                        st.write(f"Selected {len(selected_features)} features with correlation > {corr_threshold}")
                        st.write("Selected features:", selected_features)
                        
                        st.session_state.selected_features = selected_features
                        
                        # Interactive scatter plot
                        if len(selected_features) >= 2:
                            col1, col2 = st.columns(2)
                            with col1:
                                x_axis = st.selectbox("X-axis feature", selected_features, index=0)
                            with col2:
                                y_axis = st.selectbox("Y-axis feature", selected_features, index=min(1, len(selected_features)-1))
                            
                            fig = px.scatter(
                                df_train,
                                x=x_axis,
                                y=y_axis,
                                color='satisfaction',
                                title=f"{x_axis} vs {y_axis} colored by Satisfaction"
                            )
                            st.plotly_chart(fig)
                    else:
                        st.warning("No features meet the correlation threshold")
                else:
                    st.warning("Could not calculate feature correlations")
            else:
                st.error("Target column 'satisfaction' not found!")
        
        with st.expander("🧩 PCA Dimensionality Reduction"):
            if hasattr(st.session_state, 'selected_features') and st.session_state.selected_features:
                selected_features = st.session_state.selected_features
                
                X_train = df_train[selected_features]
                y_train = df_train['satisfaction']
                X_test = df_test[selected_features]
                y_test = df_test['satisfaction']
                
                # Standardize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Fit PCA
                pca = PCA()
                pca.fit(X_train_scaled)
                
                # Plot explained variance
                fig, ax = plt.subplots()
                ax.plot(np.cumsum(pca.explained_variance_ratio_))
                ax.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
                ax.set_xlabel('Number of Components')
                ax.set_ylabel('Cumulative Explained Variance')
                ax.set_title('PCA Explained Variance')
                ax.legend()
                st.pyplot(fig)
                
                # Let user select number of components
                max_components = min(20, len(selected_features))
                n_components = st.slider(
                    "Select number of PCA components",
                    min_value=1,
                    max_value=max_components,
                    value=min(10, max_components)
                )
                # Apply PCA
                pca = PCA(n_components=n_components)
                X_train_pca = pca.fit_transform(X_train_scaled)
                X_test_pca = pca.transform(X_test_scaled)
                
                # Store in session state
                st.session_state.X_train_pca = X_train_pca
                st.session_state.X_test_pca = X_test_pca
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                st.success(f"PCA applied: Reduced from {X_train.shape[1]} to {n_components} components")
            else:
                st.warning("Please select features first")
    else:
        st.warning("Please complete data cleaning first")

# Page 5: Model Building
elif page == "5. Model Building":
    st.title("🤖 Model Training")
    
    if st.session_state.X_train_pca is not None:
        # Prepare data
        X_train = st.session_state.X_train_pca
        y_train = st.session_state.y_train
        
        # Ensure y_train is numeric
        if not np.issubdtype(y_train.dtype, np.number):
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
        
        # Limit sample size for faster training
        sample_size = min(20000, len(X_train))
        X_train = X_train[:sample_size]
        y_train = y_train[:sample_size]
        
        st.write(f"Training on {len(X_train)} samples...")
        
        with st.expander("⚙ Model Configuration"):
            # Simplified parameter grid
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['rbf'],
                'gamma': ['scale']
            }
            
            # Let user choose search method
            search_method = st.radio(
                "Hyperparameter search method",
                ["Grid Search", "Random Search"],
                index=0
            )
            
            if search_method == "Random Search":
                n_iter = st.slider("Number of iterations", 5, 50, 10)
        
        # Train model
        if st.button("Train Model"):
            with st.spinner("Training in progress..."):
                progress_bar = st.progress(0)
                
                try:
                    # Initialize model
                    svm = SVC(probability=True)
                    
                    # Select search method
                    if search_method == "Grid Search":
                        search = GridSearchCV(
                            svm,
                            param_grid,
                            cv=3,
                            n_jobs=-1,
                            verbose=1
                        )
                    else:
                        search = RandomizedSearchCV(
                            svm,
                            param_distributions=param_grid,
                            n_iter=n_iter,
                            cv=3,
                            n_jobs=-1,
                            verbose=1
                        )
                    
                    # Simulate progress (remove in production)
                    for i in range(10):
                        time.sleep(0.2)
                        progress_bar.progress((i + 1) * 10)
                    
                    # Fit model
                    search.fit(X_train, y_train)
                    progress_bar.empty()
                    
                    # Store results
                    st.session_state.model = search.best_estimator_
                    
                    # Display results
                    st.success("Training complete!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("*Best Parameters:*")
                        st.json(search.best_params_)
                    with col2:
                        st.metric("Best CV Score", f"{search.best_score_:.4f}")
                    
                    # Prepare test samples
                    if hasattr(st.session_state, 'X_test_pca'):
                        test_size = min(5000, len(st.session_state.X_test_pca))
                        st.session_state.X_test_eval = st.session_state.X_test_pca[:test_size]
                        st.session_state.y_test_eval = st.session_state.y_test[:test_size]
                
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"Training failed: {str(e)}")
    else:
        st.warning("Please complete dimensionality reduction first")

# Page 6: Evaluation
elif page == "6. Evaluation":
    st.title("📊 Model Evaluation")
    
    if st.session_state.model is not None:
        model = st.session_state.model
        
        if hasattr(st.session_state, 'X_test_eval'):
            X_test = st.session_state.X_test_eval
            y_test = st.session_state.y_test_eval
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Display metrics
            st.subheader("Model Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            with col2:
                st.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
            with col3:
                st.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Neutral/Dissatisfied', 'Satisfied'],
                y=['Neutral/Dissatisfied', 'Satisfied'],
                text_auto=True
            )
            fig.update_layout(title="Confusion Matrix")
            st.plotly_chart(fig)
            
            # Classification report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0))
            
            # Probability distribution
            st.subheader("Probability Distribution")
            fig = px.histogram(
                x=y_proba,
                color=y_test,
                nbins=50,
                labels={'x': 'Predicted Probability', 'y': 'Count'},
                title='Predicted Probability Distribution'
            )
            st.plotly_chart(fig)
        else:
            st.warning("Test data not available for evaluation")
    else:
        st.warning("Please train the model first")

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
*Airline Satisfaction Prediction*  
Built with Streamlit | [GitHub Repo](#)
""")
