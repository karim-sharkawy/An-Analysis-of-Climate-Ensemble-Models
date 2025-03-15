# Analysis & Discussion of the Model Results

### Results:
Baseline Model RMSE: 19.8886, MAE: 16.8117, R²: -0.0000  
Random Forest RMSE: 17.4401, MAE: 14.1926, R²: 0.2311  
LSTM RMSE: 1.9114, MAE: 1.2865, R²: 0.9902  
Meta-Model Ensemble RMSE: 19.8884, MAE: 16.8122, R²: 0.0000  

Random Forest Improvement over Baseline (RMSE): 12.31%  
LSTM Improvement over Baseline (RMSE): 90.39%  
Meta-Model Ensemble Improvement over Baseline (RMSE): 0.00%  

Random Forest Improvement over Baseline (MAE): 15.58%  
LSTM Improvement over Baseline (MAE): 92.35%  
Meta-Model Ensemble Improvement over Baseline (MAE): -0.00%  

Random Forest Improvement over Baseline (R²): 0.00%  
LSTM Improvement over Baseline (R²): 0.00%  
Meta-Model Ensemble Improvement over Baseline (R²): 0.00%  

The results from the evaluation of the various models—baseline, Random Forest, LSTM, and the Meta-Model Ensemble—provide a comprehensive view of the performance of these approaches on the regression task. Each model offers different strengths and weaknesses, and understanding these is crucial for assessing their utility in this specific problem. Let’s delve into the performance of each model, analyzing aspects like overfitting, underfitting, potential improvements, and the interpretation of the metrics such as RMSE, MAE, and R².

### Model Performance Overview

Starting with the baseline model, which predicts the mean of the training target values for all test data points, we see that it has a significant shortcoming in predictive power, evidenced by an R² of 0.00% and relatively high error metrics (RMSE and MAE). The baseline is not capturing any meaningful relationship between the features and the target variable. It serves as a reference point to evaluate more sophisticated models, and any improvement in the RMSE, MAE, or R² relative to this baseline indicates better model performance.

The Random Forest model provides a substantial improvement over the baseline, reducing both RMSE and MAE. However, its R² score of 0.23 suggests that it’s only able to explain a modest portion of the variance in the target variable. This indicates that while the model is better than the baseline in terms of predictive accuracy, there’s still room for improvement in its ability to generalize to new data. The relatively low R² suggests the possibility of underfitting, as the Random Forest might not be capturing the full complexity of the data. Additionally, overfitting could still be a concern if the model is not tuned properly, particularly with the number of trees and the depth of the trees. A more refined hyperparameter search, using techniques like GridSearchCV or more exhaustive random search, could help find the optimal set of parameters to reduce underfitting and improve generalization.

The LSTM model, on the other hand, shines in this comparison. It shows an impressive reduction in RMSE and MAE, and its R² score of 0.99 indicates that the model is explaining almost all the variance in the target variable. This makes it the best-performing model in terms of predictive accuracy. The high R² suggests that the model has learned the underlying temporal patterns in the data effectively, making it a strong candidate for time-series forecasting tasks. The early stopping callback, used during training, likely prevented overfitting by halting training when validation performance began to plateau. Given these results, it's unlikely that the model is underfitting, but overfitting is always a possibility with deep learning models, especially if the model is too complex for the amount of available data. To mitigate this risk, regularization techniques, such as dropout (which is already applied), and more careful tuning of hyperparameters like batch size, number of layers, and learning rate could further improve generalization.

Despite the impressive performance of the LSTM model, the Meta-Model Ensemble, which combines predictions from both the Random Forest and LSTM models through a linear regression meta-model, does not offer any improvement over the baseline. This indicates that the ensemble model is unable to leverage the strengths of the two individual models effectively. The meta-model’s performance is essentially identical to that of the baseline, with no substantial reduction in error or improvement in R². This could be due to a variety of factors, such as a lack of diversity between the Random Forest and LSTM models’ predictions or overfitting of the linear regression model to the noise in the predictions. In this case, the ensemble approach fails to capture any additional value, suggesting that a different ensemble method, perhaps using more sophisticated models like XGBoost or Gradient Boosting Machines, could have been more effective.

### Overfitting and Underfitting

Overfitting and underfitting are common issues in machine learning models that can significantly impact the performance of predictive models, and were a main concern in the beginning of this project due to the amount of data being put into the LSTM model specifically.

Considering overfitting and underfitting, it's important to note that the Random Forest model might be prone to both, depending on how its hyperparameters are set. The relatively low R² score suggests that the model may not be capturing enough complexity in the data (underfitting). However, given its improvement over the baseline, it's also possible that it could overfit if its parameters—like the depth of trees or the number of samples per split—are not appropriately regularized. Regularization techniques, like limiting the maximum depth of trees or increasing the minimum samples required for a split, can help mitigate this risk. The LSTM model, on the other hand, performs exceptionally well, with little evidence of underfitting, as indicated by the high R² and low errors. The use of early stopping suggests that overfitting is being effectively controlled. However, as with any deep learning model, there’s always a risk of overfitting, especially with more complex architectures or insufficient data, though the LSTM model appears to have generalized well here.

The Meta-Model Ensemble, however, likely suffers from a form of overfitting, where the combination of Random Forest and LSTM predictions does not provide enough diversity or value to improve the performance over the baseline. Since the meta-model is just a linear regression, it may not be sophisticated enough to handle the nuanced relationships between the predictions of the two models. Moreover, if both models are prone to similar kinds of error or if their predictions are too highly correlated, the meta-model might not improve on the results from individual models. This suggests that better ensemble methods, such as stacking with more complex base models, might be needed to fully exploit the potential of the ensemble approach.

### What the Improvement Percentages Mean

The percentages for RMSE, MAE, and R² improvements represent the relative change in these metrics when comparing the models to the baseline model. For example, a 90.39% improvement in RMSE for the LSTM model means that the LSTM reduces the RMSE by nearly 90% compared to the baseline, demonstrating its superiority in making accurate predictions. Similarly, the 92.35% improvement in MAE suggests that the LSTM's predictions are much closer to the true values, reducing the mean absolute error by almost 92%. These numbers highlight the effectiveness of the LSTM model compared to the baseline, which simply predicts the mean target value.

However, an R² improvement of 0.00% does not necessarily mean the model is performing poorly; rather, it indicates that the model is not making any substantial improvements in terms of explaining the variance in the target variable relative to the baseline. This might be due to how R² is being calculated or the nature of the model’s output. For instance, the Random Forest and Meta-Model Ensemble both show an R² improvement of 0.00%, which suggests that while they are reducing error, they are not capturing a significant amount of the variance in the target variable. In contrast, the LSTM model, with its R² of 0.99, shows a massive improvement in explaining the data’s variance, which aligns with its superior predictive performance.

### Potential Improvements and Next Steps

There are several ways to potentially improve the models. For Random Forest, a more thorough hyperparameter tuning process could be beneficial, especially regarding the maximum tree depth and the minimum number of samples required to split a node. These adjustments could help reduce underfitting and improve generalization. Additionally, experimenting with different feature engineering techniques or adding more data could improve model performance.

For the LSTM, further tuning of the model’s architecture and hyperparameters (such as the number of LSTM units, learning rate, and batch size) might help improve accuracy, especially if more data is available. Additionally, exploring more advanced regularization methods, such as L2 regularization, or experimenting with other architectures like GRU (Gated Recurrent Units) could offer improvements.

Finally, while the Meta-Model Ensemble didn’t perform as expected, it’s worth experimenting with more complex ensemble methods, such as stacking with different base models or boosting techniques, which might be more successful in combining the strengths of the Random Forest and LSTM models.

### Conclusion

In conclusion, while the LSTM model stands out as the best-performing model, both in terms of predictive accuracy and its ability to explain variance in the target variable, there are still opportunities for improvement across all models. The Random Forest model provides a solid improvement over the baseline but could benefit from more hyperparameter tuning. The Meta-Model Ensemble, despite combining two powerful models, did not offer any significant improvement over the baseline, suggesting that more advanced ensemble methods or better diversity in model predictions could be explored. Ultimately, understanding the strengths and weaknesses of each model—along with the potential for overfitting, underfitting, and model-specific improvements—provides a clearer path for enhancing predictive accuracy in future iterations.
