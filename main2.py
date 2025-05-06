# -----------------------------  app.py  ---------------------------------
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
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score)
import io

# ------------------------------------------------------------------ #
# Streamlit page set-up
# ------------------------------------------------------------------ #
st.set_page_config(page_title="Airline Satisfaction SVM",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------------ #
# Helper utilities
# ------------------------------------------------------------------ #
def fix_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast dtypes so that Arrow / Streamlit do not choke on mixed types."""
    df = df.copy()
    # strings
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("string")
    # ints to floats (Arrow limitation)
    for c in df.select_dtypes(include=["int64", "Int64"]).columns:
        df[c] = df[c].astype("float64")
    # specific known conversion
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1}).astype("float64")
    return df


def safe_dataframe_display(df: pd.DataFrame, max_rows: int = 5) -> None:
    st.dataframe(fix_dataframe_types(df.head(max_rows)))


def preprocess_data(df: pd.DataFrame,
                    missing_strategy: str = "mean",
                    missing_value=None,
                    outlier_strategy: str = "remove",
                    outlier_threshold: float = 1.5) -> pd.DataFrame:
    """Missing values + categorical encoding + outlier handling (IQR)."""
    df = fix_dataframe_types(df.copy())
    df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True)

    # ---- missing values --------------------------------------------------
    miss_cols = df.columns[df.isnull().any()].tolist()
    if miss_cols:
        if missing_strategy == "delete":
            df = df.dropna()
        elif missing_strategy in ("mean", "median"):
            for c in miss_cols:
                if missing_strategy == "mean":
                    fill = df[c].mean()
                else:
                    fill = df[c].median()
                df[c] = df[c].fillna(fill)
        elif missing_strategy == "specific":
            for c in miss_cols:
                df[c] = df[c].fillna(missing_value)

    # ---- encode categoricals --------------------------------------------
    for c in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        try:
            df[c] = le.fit_transform(df[c].astype(str))
        except Exception:
            # fallback
            df[c] = df[c].astype("category").cat.codes

    # ---- outlier processing (IQR) ---------------------------------------
    if outlier_strategy != "ignore":
        for c in df.select_dtypes(include=np.number).columns:
            q1, q3 = df[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            low, high = q1 - outlier_threshold * iqr, q3 + outlier_threshold * iqr
            if outlier_strategy == "remove":
                df = df[(df[c] >= low) & (df[c] <= high)]
            elif outlier_strategy == "cap":
                df[c] = df[c].clip(low, high)

    return fix_dataframe_types(df)


def plot_boxplots(df: pd.DataFrame, title: str) -> None:
    """Robust box-plot grid that skips empty columns."""
    # keep only numeric columns with at least one finite value
    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns
                    if np.isfinite(df[c]).sum() > 0]

    if not numeric_cols:
        st.warning("No numeric columns with valid data to plot.")
        return

    n = len(numeric_cols)
    cols_per_row = 3
    rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row,
                             figsize=(15, 4 * rows),
                             squeeze=False)
    axes = axes.flatten()

    for idx, col in enumerate(numeric_cols):
        sns.boxplot(y=df[col], ax=axes[idx])
        axes[idx].set_title(col)

    # turn off unused axes
    for j in range(len(numeric_cols), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, y=1.02)
    st.pyplot(fig)


# ------------------------------------------------------------------ #
# Session-state placeholders
# ------------------------------------------------------------------ #
for k in (
        "train_df", "test_df",
        "cleaned_train", "cleaned_test",
        "X_train_pca", "X_test_pca",
        "y_train", "y_test",
        "model"):
    st.session_state.setdefault(k, None)

# ------------------------------------------------------------------ #
# Sidebar navigation
# ------------------------------------------------------------------ #
PAGES = ["1. Upload Data", "2. EDA", "3. Data Cleaning",
         "4. Dimensionality Reduction", "5. Model Building", "6. Evaluation"]
page = st.sidebar.radio("Navigation", PAGES, index=0)

# ================================================================== #
# 1. UPLOAD DATA
# ================================================================== #
if page == "1. Upload Data":
    st.title("📤 Upload Train and Test CSV Files")

    with st.expander("Instructions"):
        st.markdown("""
        • Upload training and testing CSV files.  
        • Each file **must contain** a **`satisfaction`** column.
        """)

    col1, col2 = st.columns(2)
    with col1:
        train_file = st.file_uploader("Train CSV", type="csv")
    with col2:
        test_file = st.file_uploader("Test CSV", type="csv")

    if train_file and test_file:
        try:
            st.session_state.train_df = fix_dataframe_types(pd.read_csv(train_file))
            st.session_state.test_df  = fix_dataframe_types(pd.read_csv(test_file))

            st.success("Files uploaded!")

            st.subheader("Train preview")
            safe_dataframe_display(st.session_state.train_df)

            st.subheader("Test preview")
            safe_dataframe_display(st.session_state.test_df)

            if "satisfaction" not in st.session_state.train_df.columns:
                st.error("`satisfaction` column missing in TRAIN file.")
            if "satisfaction" not in st.session_state.test_df.columns:
                st.error("`satisfaction` column missing in TEST file.")
        except Exception as e:
            st.exception(e)

# ================================================================== #
# 2. EDA
# ================================================================== #
elif page == "2. EDA":
    st.title("🔍 Exploratory Data Analysis")
    if st.session_state.train_df is None:
        st.warning("Upload data first.")
        st.stop()

    df = st.session_state.train_df

    # --- overview
    with st.expander("Dataset overview", expanded=True):
        st.write(f"Shape: **{df.shape[0]} rows × {df.shape[1]} cols**")
        st.dataframe(df.dtypes.rename("dtype"))

    # --- numeric distrib
    with st.expander("Numeric distributions"):
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            sel = st.multiselect("Choose numeric columns", num_cols, num_cols[:4])
            for c in sel:
                fig, ax = plt.subplots()
                sns.histplot(df[c], kde=True, ax=ax)
                ax.set_title(c)
                st.pyplot(fig)
        else:
            st.info("No numeric columns.")

    # --- correlation
    with st.expander("Correlation matrix"):
        num_df = df.select_dtypes(include=np.number)
        if not num_df.empty:
            corr = num_df.corr()
            st.plotly_chart(px.imshow(corr,
                                      labels=dict(color="corr"),
                                      x=corr.columns, y=corr.columns),
                            use_container_width=True)

# ================================================================== #
# 3. DATA CLEANING
# ================================================================== #
elif page == "3. Data Cleaning":
    st.title("🧹 Data Cleaning & Outlier Handling")
    if st.session_state.train_df is None:
        st.warning("Upload data first.")
        st.stop()

    df = st.session_state.train_df

    # ---- missing values pane -------------------------------------------
    with st.expander("Missing-value handling", expanded=True):
        miss = df.isnull().sum()
        miss = miss[miss > 0]
        if miss.empty:
            st.success("No missing values 🎉")
            missing_strategy = "none"
            missing_value = None
        else:
            st.dataframe(miss.rename("n_missing"))
            missing_strategy = st.radio("Strategy",
                                        ["delete", "mean", "median", "specific"])
            missing_value = None
            if missing_strategy == "specific":
                missing_value = st.text_input("Fill with:")

    # ---- outlier pane ---------------------------------------------------
    with st.expander("Outlier analysis", expanded=True):
        num_cols = df.select_dtypes(include=np.number).columns
        if num_cols.empty:
            st.info("No numeric columns.")
            outlier_strategy = "ignore"
            iqr_mult = 1.5
        else:
            # report
            out = []
            for c in num_cols:
                q1, q3 = df[c].quantile([0.25, 0.75])
                iqr = q3 - q1
                low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                n_out = ((df[c] < low) | (df[c] > high)).sum()
                if n_out:
                    out.append((c, n_out, n_out / len(df) * 100))
            if out:
                st.dataframe(pd.DataFrame(out, columns=["column", "n_out", "%"]))
                plot_boxplots(df, "Boxplots before treatment")
                outlier_strategy = st.radio("Outlier strategy",
                                            ["ignore", "remove", "cap"])
                iqr_mult = st.slider("IQR multiplier", 1.0, 5.0, 1.5, 0.1)
            else:
                st.success("No sizeable outliers detected.")
                outlier_strategy = "ignore"
                iqr_mult = 1.5

    # ---- apply button ---------------------------------------------------
    if st.button("Apply cleaning"):
        with st.spinner("Cleaning…"):
            st.session_state.cleaned_train = preprocess_data(
                df,
                missing_strategy=missing_strategy if miss.empty is False else "mean",
                missing_value=missing_value,
                outlier_strategy=outlier_strategy,
                outlier_threshold=iqr_mult
            )
            st.session_state.cleaned_test = preprocess_data(
                st.session_state.test_df,
                missing_strategy="mean",
                outlier_strategy="ignore"
            )
        st.success("Cleaning done!")

# ================================================================== #
# 4. PCA
# ================================================================== #
elif page == "4. Dimensionality Reduction":
    st.title("📉 PCA")
    if st.session_state.cleaned_train is None:
        st.warning("Run data-cleaning first.")
        st.stop()

    trn = st.session_state.cleaned_train
    tst = st.session_state.cleaned_test
    num_cols = trn.select_dtypes(include=np.number).columns.tolist()

    if len(num_cols) < 2:
        st.warning("Need at least two numeric features.")
        st.stop()

    # scale & PCA
    scaler = StandardScaler()
    X_trn = scaler.fit_transform(trn[num_cols])
    X_tst = scaler.transform(tst[num_cols])

    pca_full = PCA().fit(X_trn)
    fig, ax = plt.subplots()
    ax.plot(np.cumsum(pca_full.explained_variance_ratio_))
    ax.axhline(0.95, ls="--", c="r")
    ax.set_xlabel("#components")
    ax.set_ylabel("cumulative explained var.")
    st.pyplot(fig)

    n_comp = st.slider("Components", 1, min(len(num_cols), 20), 10)
    pca = PCA(n_components=n_comp)
    st.session_state.X_train_pca = pca.fit_transform(X_trn)
    st.session_state.X_test_pca  = pca.transform(X_tst)
    st.session_state.y_train = trn["satisfaction"].values
    st.session_state.y_test  = tst["satisfaction"].values

    st.success(f"PCA reduced from {len(num_cols)} → {n_comp} features.")

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
