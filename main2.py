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

# Session state initialization
for k in [
    'train_df', 'test_df', 'X_train_pca', 'X_test_pca', 'y_train', 'y_test', 
    'model', 'cleaned_train', 'cleaned_test', 'selected_features', 'X_test_eval', 'y_test_eval'
]:
    if k not in st.session_state:
        st.session_state[k] = None

# Robust boxplot function, now SAFE for 1 or more numeric columns
def plot_boxplots(df, title):
    numeric_cols = df.select_dtypes(include=np.number).columns
    n = len(numeric_cols)
    if n == 0:
        st.warning("No numeric columns to plot")
        return
    
    # Remove columns with all NaN or only one unique value
    cols_to_plot = [c for c in numeric_cols if df[c].nunique(dropna=True) > 1 and df[c].notnull().sum() > 1]
    if not cols_to_plot:
        st.warning("Not enough numeric columns with >1 unique value for boxplots.")
        return

    n_plot = len(cols_to_plot)
    cols_per_row = 3
    nrows = int(np.ceil(n_plot / cols_per_row))
    fig, axes = plt.subplots(nrows=nrows, ncols=cols_per_row, figsize=(5 * cols_per_row, 5 * nrows))
    # Ensure axes is always 2D array for consistent indexing
    if nrows == 1 and cols_per_row == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.expand_dims(axes, 0)
    elif cols_per_row == 1:
        axes = np.expand_dims(axes, 1)

    axes_flat = axes.flatten()
    for i, col in enumerate(cols_to_plot):
        sns.boxplot(y=df[col], ax=axes_flat[i])
        axes_flat[i].set_title(col)
    # Hide any unused axes
    for i in range(n_plot, len(axes_flat)):
        axes_flat[i].set_axis_off()

    fig.suptitle(title)
    st.pyplot(fig)
    plt.close(fig)

def preprocess_data(df, missing_strategy='mean', missing_value=None, outlier_strategy='remove', outlier_threshold=1.5):
    df = fix_dataframe_types(df.copy())
    df.drop(columns=[col for col in df.columns if "Unnamed" in col], inplace=True)

    # Missing value handling
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        if missing_strategy == 'delete':
            df = df.dropna()
        elif missing_strategy in ['mean', 'median']:
            for col in missing_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    fill_value = df[col].mean() if missing_strategy == 'mean' else df[col].median()
                else:
                    fill_value = df[col].mode().dropna().iloc[0] if not df[col].mode().empty else ""
                df[col] = df[col].fillna(fill_value)
        elif missing_strategy == 'specific':
            for col in missing_cols:
                df[col] = df[col].fillna(missing_value)

    # Encode object columns
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        try:
            df[col] = le.fit_transform(df[col].astype(str))
        except Exception:
            df[col] = df[col].astype('category').cat.codes

    # Outlier handling (IQR)
    numeric_cols = df.select_dtypes(include=np.number).columns
    if outlier_strategy != 'ignore':
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - outlier_threshold * iqr
            upper = q3 + outlier_threshold * iqr
            if outlier_strategy == 'remove':
                df = df[(df[col] >= lower) & (df[col] <= upper)]
            elif outlier_strategy == 'cap':
                df[col] = df[col].clip(lower, upper)
    return fix_dataframe_types(df)

# Sidebar Navigation
pages = [
    "1. Upload Data",
    "2. EDA",
    "3. Data Cleaning",
    "4. Dimensionality Reduction",
    "5. Model Building",
    "6. Evaluation",
]

page = st.sidebar.radio("Navigation", pages, index=0)

# PAGE 1: Upload Data
if page == "1. Upload Data":
    st.title("📤 Upload Train and Test CSV Files")
    with st.expander("ℹ Instructions"):
        st.write("Upload your training (train.csv) and test (test.csv) data with the 'satisfaction' column.")
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
            if 'satisfaction' not in st.session_state.train_df.columns:
                st.error("❌ 'satisfaction' column not found in training data!")
            if 'satisfaction' not in st.session_state.test_df.columns:
                st.error("❌ 'satisfaction' column not found in test data!")
        except Exception as e:
            st.error(f"Error loading files: {str(e)}")

# PAGE 2: EDA
elif page == "2. EDA":
    st.title("🔍 Exploratory Data Analysis")
    if st.session_state.train_df is not None:
        df = fix_dataframe_types(st.session_state.train_df)
        with st.expander("📊 Dataset Overview"):
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            st.write(df.dtypes)
        with st.expander("Summary Statistics"):
            st.dataframe(df.describe(include='all'))
        with st.expander("Target Variable Analysis"):
            if 'satisfaction' in df.columns:
                fig = px.histogram(df, x='satisfaction', color='satisfaction', title='Distribution of Satisfaction')
                st.plotly_chart(fig)
            else:
                st.error("Target column 'satisfaction' not found!")
        with st.expander("Distributions"):
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                selected = st.multiselect("Select numeric columns to plot", options=numeric_cols, default=numeric_cols[:3])
                for col in selected:
                    fig, ax = plt.subplots()
                    sns.histplot(df[col], kde=True, ax=ax)
                    st.pyplot(fig)
            else:
                st.warning("No numeric columns for distributions")
        with st.expander("Correlation Matrix"):
            numeric_df = df.select_dtypes(include=np.number)
            if len(numeric_df.columns) > 0:
                st.plotly_chart(px.imshow(numeric_df.corr()))
        with st.expander("Pair Relationships"):
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) > 1:
                pairplot_cols = st.multiselect("Select columns (max 5)", numeric_cols, default=numeric_cols[:2])
                if 2 <= len(pairplot_cols) <= 5:
                    st.plotly_chart(px.scatter_matrix(df, dimensions=pairplot_cols, color='satisfaction' if 'satisfaction' in df.columns else None))
                else:
                    st.warning("Select 2-5 columns")
    else:
        st.warning("Please upload data first.")

# PAGE 3: Data Cleaning
elif page == "3. Data Cleaning":
    st.title("🧹 Data Cleaning and Outlier Handling")
    if st.session_state.train_df is not None:
        df = fix_dataframe_types(st.session_state.train_df.copy())
        with st.expander("Missing Value Analysis"):
            missing_values = df.isnull().sum()
            missing_values = missing_values[missing_values > 0]
            if not missing_values.empty:
                st.write("Columns with missing values:")
                st.dataframe(missing_values.rename('Missing Values'))
                st.subheader("Missing Value Treatment")
                missing_strategy = st.radio("Handle missing values?", ['delete', 'mean', 'median', 'specific'], horizontal=True)
                missing_value = st.text_input("Value to fill:", value="") if missing_strategy == 'specific' else None
            else:
                missing_strategy, missing_value = 'mean', None
                st.info("No missing values.")
        with st.expander("Outlier Analysis"):
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                outliers_report = []
                for col in numeric_cols:
                    q1, q3 = df[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                    if n_outliers > 0:
                        outliers_report.append({'Column': col, 'Outliers': n_outliers, 'Perc': f"{n_outliers / len(df) * 100:.1f}%"})
                if outliers_report:
                    outliers_df = pd.DataFrame(outliers_report)
                    st.dataframe(outliers_df)
                    st.subheader("Outlier Visualization")
                    plot_boxplots(df, "Boxplots Before Outlier Treatment")
                    st.subheader("Outlier Treatment")
                    outlier_strategy = st.radio("Handle outliers?", ['ignore', 'remove', 'cap'], horizontal=True)
                    iqr_multiplier = st.slider("IQR multiplier", 1.0, 5.0, 1.5, 0.1)
                else:
                    st.info("No outlier columns detected.")
                    outlier_strategy, iqr_multiplier = 'ignore', 1.5
            else:
                outlier_strategy, iqr_multiplier = 'ignore', 1.5
        if st.button("Apply Data Cleaning"):
            with st.spinner("Cleaning data..."):
                cleaned_train = preprocess_data(df, missing_strategy, missing_value, outlier_strategy, iqr_multiplier)
                cleaned_test = preprocess_data(
                    st.session_state.test_df,
                    'mean' if missing_strategy == 'delete' else missing_strategy,
                    missing_value,
                    'ignore'
                )
                st.session_state.cleaned_train = cleaned_train
                st.session_state.cleaned_test = cleaned_test
                st.success("Done!")
                st.write("New shape:", cleaned_train.shape)
                if outlier_strategy != 'ignore':
                    plot_boxplots(cleaned_train, "Boxplots After Outlier Treatment")
    else:
        st.warning("Please upload data first.")

# PAGE 4: Dimensionality Reduction
elif page == "4. Dimensionality Reduction":
    st.title("📉 Dimensionality Reduction")
    if st.session_state.cleaned_train is not None:
        df_train = st.session_state.cleaned_train
        df_test = st.session_state.cleaned_test
        numeric_features = df_train.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_features) > 1:
            st.write("Numeric features available for PCA:", numeric_features)
            X_train = df_train[numeric_features].copy()
            y_train = df_train['satisfaction']
            X_test = df_test[numeric_features].copy()
            y_test = df_test['satisfaction']
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            pca = PCA().fit(X_train_scaled)
            fig, ax = plt.subplots()
            ax.plot(np.cumsum(pca.explained_variance_ratio_))
            ax.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
            ax.set_xlabel('Number of Components')
            ax.set_ylabel('Cumulative Explained Variance')
            ax.set_title('PCA Explained Variance')
            ax.legend()
            st.pyplot(fig)
            n_components = st.slider("Number of PCA components", 1, min(20, len(numeric_features)), min(10, len(numeric_features)))
            pca = PCA(n_components=n_components)
            st.session_state.X_train_pca = pca.fit_transform(X_train_scaled)
            st.session_state.X_test_pca = pca.transform(X_test_scaled)
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.success(f"PCA applied: {X_train.shape[1]} → {n_components}")
        else:
            st.warning("Not enough numeric features for PCA")
    else:
        st.warning("Please complete data cleaning first.")

# ================================================================== #
# 5. MODEL TRAINING
# ================================================================== #
elif page == "5. Model Building":
    st.title("🤖 Train SVM")
    if st.session_state.X_train_pca is None:
        st.warning("Run PCA first.")
        st.stop()

    X, y = st.session_state.X_train_pca, st.session_state.y_train
    st.write(f"Training samples: **{len(X)}**")

    param_grid = {"C": [0.1, 1, 10], "kernel": ["rbf"], "gamma": ["scale"]}
    method = st.radio("Search", ["Grid", "Random"])
    n_iter = st.slider("Random-search iters", 5, 50, 10) if method == "Random" else None

    if st.button("Start training"):
        with st.spinner("Training…"):
            base = SVC(probability=True)
            if method == "Grid":
                search = GridSearchCV(base, param_grid, cv=3, n_jobs=-1)
            else:
                search = RandomizedSearchCV(base, param_grid, n_iter=n_iter, cv=3, n_jobs=-1)
            search.fit(X, y)
            st.session_state.model = search.best_estimator_
            st.success("Done!")
            st.json(search.best_params_)
            st.metric("Best CV score", f"{search.best_score_:.3f}")

# ================================================================== #
# 6. EVALUATION
# ================================================================== #
elif page == "6. Evaluation":
    st.title("📊 Evaluation")
    if st.session_state.model is None:
        st.warning("Train a model first.")
        st.stop()

    X_test = st.session_state.X_test_pca
    y_test = st.session_state.y_test
    y_pred = st.session_state.model.predict(X_test)
    y_prob = st.session_state.model.predict_proba(X_test)[:, 1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")

    st.subheader("Confusion matrix")
    st.plotly_chart(px.imshow(confusion_matrix(y_test, y_pred),
                              text_auto=True,
                              labels=dict(x="Pred", y="True", color="count")),
                    use_container_width=True)

    st.subheader("Probability histogram")
    st.plotly_chart(px.histogram(y_prob, nbins=40,
                                 labels={"value": "P(class=1)"},
                                 title="Predicted probabilities"))
# ------------------------------------------------------------------ #
st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ and Streamlit")
