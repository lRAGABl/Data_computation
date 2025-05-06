import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
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
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('str')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('float64')
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype('float64')
    return df

def safe_dataframe_display(df, max_rows=5):
    """Safely display DataFrame with type conversion"""
    display_df = fix_dataframe_types(df.head(max_rows))
    st.dataframe(display_df)

# Initialize session state
session_keys = [
    'train_df', 'test_df', 'X_train_pca', 'X_test_pca', 
    'y_train', 'y_test', 'model', 'cleaned_train', 
    'cleaned_test', 'selected_features'
]
for key in session_keys:
    if key not in st.session_state:
        st.session_state[key] = None

def preprocess_data(df, missing_strategy='mean', missing_value=None, outlier_strategy='remove', outlier_threshold=1.5):
    """Preprocess data with missing value and outlier handling"""
    df = fix_dataframe_types(df.copy())
    df.drop(columns=[col for col in df.columns if "Unnamed" in col], inplace=True)

    # Handle missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        if missing_strategy == 'delete':
            df = df.dropna()
        elif missing_strategy in ['mean', 'median']:
            for col in missing_cols:
                fill_value = df[col].mean() if missing_strategy == 'mean' else df[col].median()
                df[col] = df[col].fillna(fill_value)
        elif missing_strategy == 'specific':
            for col in missing_cols:
                df[col] = df[col].fillna(missing_value)

    # Encode categorical columns
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        try:
            df[col] = le.fit_transform(df[col].astype(str))
        except Exception as e:
            st.warning(f"Could not encode column {col}: {str(e)}")
            df[col] = df[col].astype('category').cat.codes

    # Handle outliers using IQR method
    numeric_cols = df.select_dtypes(include=np.number).columns
    if outlier_strategy != 'ignore':
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - outlier_threshold * iqr
            upper_bound = q3 + outlier_threshold * iqr
            if outlier_strategy == 'remove':
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            elif outlier_strategy == 'cap':
                df[col] = df[col].clip(lower_bound, upper_bound)

    return fix_dataframe_types(df)

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
                
                fig = px.imshow(
                    corr_matrix,
                    labels={'x': "Features", 'y': "Features", 'color': "Correlation"},
                    x=corr_matrix.columns.tolist(),
                    y=corr_matrix.columns.tolist()
                )
                fig.update_xaxes(side="top")
                fig.update_layout(title="Correlation Matrix")
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
                
                if len(pairplot_cols) >= 2:
                    fig = px.scatter_matrix(
                        df,
                        dimensions=pairplot_cols,
                        color='satisfaction' if 'satisfaction' in df.columns else None,
                        title="Feature Relationships"
                    )
                    st.plotly_chart(fig)
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
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                    if n_outliers > 0:
                        outliers_report.append({
                            'Column': col,
                            'Outliers': n_outliers,
                            'Percentage': f"{n_outliers / len(df) * 100:.1f}%"
                        })
                
                if outliers_report:
                    outliers_df = pd.DataFrame(outliers_report)
                    st.write("Outliers detected in columns:")
                    st.dataframe(outliers_df)
                    
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
                    cleaned_train = preprocess_data(
                        st.session_state.train_df,
                        missing_strategy=missing_strategy,
                        missing_value=missing_value,
                        outlier_strategy=outlier_strategy,
                        outlier_threshold=iqr_multiplier
                    )
                    
                    # Process test data
                    cleaned_test = preprocess_data(
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
    
    if st.session_state.cleaned_train is not None:
        df_train = fix_dataframe_types(st.session_state.cleaned_train)
        df_test = fix_dataframe_types(st.session_state.cleaned_test)
        
        with st.expander("🧩 PCA Dimensionality Reduction"):
            numeric_features = df_train.select_dtypes(include=np.number).columns.tolist()
            
            if len(numeric_features) > 1:
                st.write("Numeric features available for PCA:")
                st.write(numeric_features)
                
                X_train = df_train[numeric_features]
                y_train = df_train['satisfaction']
                X_test = df_test[numeric_features]
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
                n_components = st.slider(
                    "Select number of PCA components",
                    min_value=1,
                    max_value=min(len(numeric_features), 20),
                    value=min(10, len(numeric_features))
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
                st.warning("Not enough numeric features for PCA")
    else:
        st.warning("Please complete data cleaning first")

# Page 5: Model Building
elif page == "5. Model Building":
    st.title("🤖 Model Training")
    
    if st.session_state.X_train_pca is not None:
        # Prepare data
        X_train = st.session_state.X_train_pca
        y_train = st.session_state.y_train.to_numpy()
        
        st.write(f"Training on {len(X_train)} samples...")
        
        with st.expander("⚙ Model Configuration"):
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['rbf'],
                'gamma': ['scale']
            }
            
            search_method = st.radio(
                "Hyperparameter search method",
                ["Grid Search", "Random Search"],
                index=0
            )
            
            n_iter = None
            if search_method == "Random Search":
                n_iter = st.slider("Number of iterations", 5, 50, 10)
        
        # Train model
        if st.button("Train Model"):
            with st.spinner("Training in progress..."):
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
                    
                    # Fit model
                    search.fit(X_train, y_train)
                    
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
                    
                    # Prepare test data for evaluation
                    st.session_state.X_test_eval = st.session_state.X_test_pca
                    st.session_state.y_test_eval = st.session_state.y_test
                
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
    else:
        st.warning("Please complete dimensionality reduction first")

# Page 6: Evaluation
elif page == "6. Evaluation":
    st.title("📊 Model Evaluation")
    
    if st.session_state.model is not None:
        model = st.session_state.model
        
        if st.session_state.X_test_eval is not None:
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
                st.metric("Precision", f"{precision_score(y_test, y_pred, average='binary'):.4f}")
            with col3:
                st.metric("Recall", f"{recall_score(y_test, y_pred, average='binary'):.4f}")
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Neutral/Dissatisfied', 'Satisfied'],
                y=['Neutral/Dissatisfied', 'Satisfied']
            )
            fig.update_layout(title="Confusion Matrix")
            st.plotly_chart(fig)
            
            # Classification report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0, subset=['precision', 'recall', 'f1-score']))
            
            # Probability distribution
            st.subheader("Probability Distribution")
            fig = px.histogram(
                x=y_proba,
                nbins=50,
                labels={'x': 'Predicted Probability', 'y': 'Count'},
                title='Predicted Probability Distribution'
            )
            st.plotly_chart(fig)
        else:
            st.warning("Test data not available for evaluation")
    else:
        st.warning("Please train the model first")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit")
