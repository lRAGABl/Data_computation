elif page == "4. Dimensionality Reduction":
    st.title("Feature Selection and Dimensionality Reduction")
    if st.session_state.cleaned_train is not None:
        df_train = st.session_state.cleaned_train
        df_test = st.session_state.cleaned_test
        
        st.subheader("Feature-Target Correlation")
        numeric_cols = df_train.select_dtypes(include=np.number).columns.drop('satisfaction', errors='ignore')
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
