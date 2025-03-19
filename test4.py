# ======================== Full Modified Training Code ===========================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # Added for saving scaler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import mlflow
import mlflow.sklearn
import ssl
import certifi
from sklearn.metrics import confusion_matrix

# --- NEW: Imports for capturing classification reports ---
from io import StringIO
import sys
# --------------------------------------------------------------------

# ======================== MLflow Setup ===========================
studentID = "st123699"
epochs = 500
batch_size = 254
ssl._create_default_https_context = ssl._create_unverified_context
mlflow.set_tracking_uri("https://mlflow.ml.brain.cs.ait.ac.th/")
os.environ["LOGNAME"] = "phankorn"
os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"
mlflow.set_experiment(experiment_name=f"{studentID}-a3")

# ======================== Data Loading & Preprocessing ===========================
# Load data
df = pd.read_csv(r'Cars.csv')
print(df.head(), df.shape)

# Preprocessing & EDA
df.rename(columns={'name': 'brand'}, inplace=True)
df = df[~df['fuel'].str.contains('CNG|LPG')]
owner_map = {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3,
             'Fourth & Above Owner': 4, 'Test Drive Car': 5}
df['owner'] = df['owner'].map(owner_map)
df = df[df['owner'] != 5]

# Clean numeric columns (remove units and convert to numeric)
for col, unit in zip(['mileage', 'engine', 'max_power'], [' kmpl', 'CC', 'bhp']):
    df[col] = pd.to_numeric(df[col].str.replace(unit, '', regex=True), errors='coerce')

# Simplify brand names and map into groups (example mapping)
df['brand'] = df['brand'].str.split().str[0]
brand_groups = {'Jeep': 'Luxury', 'Audi': 'Luxury', 'Maruti': 'Mass-Market', 'Honda': 'Mass-Market'}
df['brand'] = df['brand'].map(brand_groups)

# Label encode categorical variables and one-hot encode seller_type
le = LabelEncoder()
df['brand'] = le.fit_transform(df['brand'])
df['fuel'] = le.fit_transform(df['fuel'])
df['transmission'] = le.fit_transform(df['transmission'])
df = pd.get_dummies(df, columns=['seller_type'], drop_first=True)

# Drop irrelevant features
df.drop(['torque', 'seats', 'seller_type_Trustmark Dealer', 'transmission', 
         'seller_type_Individual', 'fuel', 'brand', 'owner'], axis=1, inplace=True)

# Discretize selling price into 4 quantile-based categories
df['price_category'] = pd.qcut(df['selling_price'], q=4, labels=[0, 1, 2, 3])

# Split features and labels
X = df[['year', 'km_driven', 'mileage', 'engine', 'max_power']]
y = df['price_category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)

# Fill missing values using .loc to avoid chained assignment warnings
X_train.loc[:, 'mileage'] = X_train['mileage'].fillna(X_train['mileage'].mean())
X_test.loc[:, 'mileage'] = X_test['mileage'].fillna(X_train['mileage'].mean())
for col in ['engine', 'max_power']:
    X_train.loc[:, col] = X_train[col].fillna(X_train[col].median())
    X_test.loc[:, col] = X_test[col].fillna(X_train[col].median())

# Scale features and add intercept term
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)

print("Train shapes:", X_train.shape, y_train.shape)
print("Test shapes:", X_test.shape, y_test.shape)
if y_train.dtype.name == 'category':
    y_train = y_train.cat.codes.values
    y_test = y_test.cat.codes.values
else:
    y_train = np.array(y_train.astype('int'))
    y_test = np.array(y_test.astype('int'))

X_train = np.array(X_train)
X_test  = np.array(X_test)
y_train = np.array(y_train)
y_test  = np.array(y_test)

# ======================== Save Scaler ===========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Add intercept term
X_train = np.concatenate((np.ones((X_train_scaled.shape[0], 1)), X_train_scaled), axis=1)
X_test = np.concatenate((np.ones((X_test_scaled.shape[0], 1)), X_test_scaled), axis=1)

# ======================== Model Definition ===========================
class LogisticRegression:
    """
    Implements a multi-class logistic regression classifier.
    If a regularization object is provided, its penalty will be added to the loss,
    and its derivative used in the gradient update.
    """
    def __init__(self, regularization=None, n_features=6, num_class=4, 
                 lr=0.001, method='batch', num_epochs=epochs, batch_size=batch_size):
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method = method  # options: 'sto', 'minibatch', 'batch'
        self.regularization = regularization
        self.n_features = n_features  # including intercept
        self.num_class = num_class
        self.moment = True

    def xavier(self, size):
        m = size
        lower, upper = -(1.0/np.sqrt(m)), (1.0/np.sqrt(m))
        return lower + np.random.rand(m, self.num_class) * (upper - lower)
    
    def one_hot(self, label):
        one_hot_vec = np.zeros(self.num_class)
        one_hot_vec[int(label)] = 1
        return one_hot_vec

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, P, y):
        n_samples = y.shape[0]
        log_likelihood = -np.log(P[range(n_samples), y] + 1e-15)
        return np.sum(log_likelihood) / n_samples

    def predict_proba(self, X):
        logits = X @ self.theta
        return self.softmax(logits)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def loss(self, y_true, logits):
        m = y_true.shape[0]
        probs = self.softmax(logits)
        base_loss = - np.sum(y_true * np.log(probs + 1e-15)) / m
        if self.regularization is not None:
            reg_loss = self.regularization(self.theta) / self.n_features
            return base_loss + reg_loss
        else:
            return base_loss

    def _train(self, X, y):
        logits = X @ self.theta
        yhat = self.softmax(logits)
        m = X.shape[0]
        grad = (X.T @ (yhat - y)) / m
        if self.regularization is not None:
            grad += self.regularization.derivation(self.theta) / self.n_features
        self.theta = self.theta - self.lr * grad
        return self.loss(y, logits)
    
    def fit(self, weight_init, momentum, X_train, y_train, X_val=None, y_val=None):
        # Initialize weights using either zero or Xavier initialization
        if weight_init == 'zero':
            self.theta = np.zeros((X_train.shape[1], self.num_class))
        else:
            self.theta = self.xavier(X_train.shape[1])
        
        self.moment = momentum
        
        # Log training parameters with MLflow.
        params = {
            "method": self.method, 
            "lr": self.lr, 
            "regularization": str(type(self.regularization).__name__) if self.regularization else "None"
        }
        mlflow.log_params(params)
        
        best_accuracy = -1.0  # Initialize best accuracy to a very low value
        best_theta = None    # To store the best model parameters
        
        # Training loop
        for epoch in range(self.num_epochs):
            # Update weights based on training method
            if self.method == 'sto':
                for i in range(X_train.shape[0]):
                    X_i = X_train[i].reshape(1, -1)
                    y_i = self.one_hot(y_train[i])
                    loss_val = self._train(X_i, y_i.reshape(1, -1))
            elif self.method == 'minibatch':
                for i in range(0, X_train.shape[0], self.batch_size):
                    X_batch = X_train[i:i+self.batch_size]
                    y_batch = np.array([self.one_hot(y) for y in y_train[i:i+self.batch_size]])
                    loss_val = self._train(X_batch, y_batch)
            else:  # batch mode
                y_onehot = np.array([self.one_hot(y) for y in y_train])
                loss_val = self._train(X_train, y_onehot)
            
            # Log the loss for this epoch to MLflow
            mlflow.log_metric("epoch_loss", loss_val, step=epoch+1)
            
            # If validation data is provided, compute accuracy on the validation set
            if X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val)
                epoch_acc = np.mean(y_val_pred == y_val)
                mlflow.log_metric("epoch_accuracy", epoch_acc, step=epoch+1)
                
                # If this is the best accuracy so far, save the model parameters
                if epoch_acc > best_accuracy:
                    best_accuracy = epoch_acc
                    best_theta = self.theta.copy()
                print(f"Epoch {epoch+1}, Loss: {loss_val:.4f}, Val Accuracy: {epoch_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}, Loss: {loss_val:.4f}")
        
        # After training, if a best model was found, restore its parameters
        if best_theta is not None:
            self.theta = best_theta
            print(f"Best model with Val Accuracy: {best_accuracy:.4f} restored.")

# class LogisticRegression:
#     """
#     Implements a multi-class logistic regression classifier.
#     If a regularization object is provided, its penalty will be added to the loss,
#     and its derivative used in the gradient update.
#     """
#     def __init__(self, regularization=None, n_features=6, num_class=4, 
#                  lr=0.001, method='batch', num_epochs=epochs, batch_size=batch_size):
#         self.lr = lr
#         self.num_epochs = num_epochs
#         self.batch_size = batch_size
#         self.method = method  # options: 'sto', 'minibatch', 'batch'
#         self.regularization = regularization
#         self.n_features = n_features  # including intercept
#         self.num_class = num_class
#         self.moment = True

#     def xavier(self, size):
#         m = size
#         lower, upper = -(1.0/np.sqrt(m)), (1.0/np.sqrt(m))
#         return lower + np.random.rand(m, self.num_class) * (upper - lower)
    
#     def one_hot(self, label):
#         one_hot_vec = np.zeros(self.num_class)
#         one_hot_vec[int(label)] = 1
#         return one_hot_vec

#     def softmax(self, z):
#         exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
#         return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
#     def cross_entropy_loss(self, P, y):
#         n_samples = y.shape[0]
#         log_likelihood = -np.log(P[range(n_samples), y] + 1e-15)
#         return np.sum(log_likelihood) / n_samples

#     def predict_proba(self, X):
#         logits = X @ self.theta
#         return self.softmax(logits)
    
#     def predict(self, X):
#         proba = self.predict_proba(X)
#         return np.argmax(proba, axis=1)
    
#     def loss(self, y_true, logits):
#         m = y_true.shape[0]
#         probs = self.softmax(logits)
#         base_loss = - np.sum(y_true * np.log(probs + 1e-15)) / m
#         if self.regularization is not None:
#             reg_loss = self.regularization(self.theta) / self.n_features
#             return base_loss + reg_loss
#         else:
#             return base_loss

#     def _train(self, X, y):
#         logits = X @ self.theta
#         yhat = self.softmax(logits)
#         m = X.shape[0]
#         grad = (X.T @ (yhat - y)) / m
#         if self.regularization is not None:
#             grad += self.regularization.derivation(self.theta) / self.n_features
#         self.theta = self.theta - self.lr * grad
#         return self.loss(y, logits)
    
# def fit(self, weight_init, momentum, X_train, y_train, X_val=None, y_val=None):
#     # Initialize weights using either zero or Xavier initialization
#     if weight_init == 'zero':
#         self.theta = np.zeros((X_train.shape[1], self.num_class))
#     else:
#         self.theta = self.xavier(X_train.shape[1])
    
#     self.moment = momentum
    
#     # Log training parameters with MLflow.
#     params = {
#         "method": self.method, 
#         "lr": self.lr, 
#         "regularization": str(type(self.regularization).__name__) if self.regularization else "None"
#     }
#     mlflow.log_params(params)
    
#     best_accuracy = -1.0  # Initialize best accuracy to a very low value
#     best_theta = None    # To store the best model parameters
    
#     # Training loop
#     for epoch in range(self.num_epochs):
#         # Update weights based on training method
#         if self.method == 'sto':
#             for i in range(X_train.shape[0]):
#                 X_i = X_train[i].reshape(1, -1)
#                 y_i = self.one_hot(y_train[i])
#                 loss_val = self._train(X_i, y_i.reshape(1, -1))
#         elif self.method == 'minibatch':
#             for i in range(0, X_train.shape[0], self.batch_size):
#                 X_batch = X_train[i:i+self.batch_size]
#                 y_batch = np.array([self.one_hot(y) for y in y_train[i:i+self.batch_size]])
#                 loss_val = self._train(X_batch, y_batch)
#         else:  # batch mode
#             y_onehot = np.array([self.one_hot(y) for y in y_train])
#             loss_val = self._train(X_train, y_onehot)
        
#         # Log the loss for this epoch to MLflow
#         mlflow.log_metric("epoch_loss", loss_val, step=epoch+1)
        
#         # If validation data is provided, compute accuracy on the validation set
#         if X_val is not None and y_val is not None:
#             y_val_pred = self.predict(X_val)
#             epoch_acc = np.mean(y_val_pred == y_val)
#             mlflow.log_metric("epoch_accuracy", epoch_acc, step=epoch+1)
            
#             # If this is the best accuracy so far, save the model parameters
#             if epoch_acc > best_accuracy:
#                 best_accuracy = epoch_acc
#                 best_theta = self.theta.copy()
#             print(f"Epoch {epoch+1}, Loss: {loss_val:.4f}, Val Accuracy: {epoch_acc:.4f}")
#         else:
#             print(f"Epoch {epoch+1}, Loss: {loss_val:.4f}")
    
#     # After training, if a best model was found, restore its parameters
#     if best_theta is not None:
#         self.theta = best_theta
#         print(f"Best model with Val Accuracy: {best_accuracy:.4f} restored.")


# ======================== Regularization Classes ===========================
class RidgePenalty:
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta):
        return self.l * np.sum(np.square(theta))
        
    def derivation(self, theta):
        return self.l * 2 * theta

class Ridge(LogisticRegression):
    def __init__(self, method, lr, l, n_features=6, num_class=4, num_epochs = epochs, batch_size = batch_size):
        regularization = RidgePenalty(l)
        super().__init__(regularization=regularization, n_features=n_features, num_class=num_class, 
                         lr=lr, method=method, num_epochs=num_epochs, batch_size=batch_size)

# ======================== Evaluation Metrics Functions ===========================
def accuracy_metric(y_pred, y_true):
    correct = sum(1 for p, t in zip(y_pred, y_true) if p == t)
    return correct / len(y_pred)

def precision_metric(y_pred, y_true, class_det):
    tp = sum(1 for p, t in zip(y_pred, y_true) if p == class_det and t == class_det)
    fp = sum(1 for p, t in zip(y_pred, y_true) if p == class_det and t != class_det)
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall_metric(y_pred, y_true, class_det):
    tp = sum(1 for p, t in zip(y_pred, y_true) if p == class_det and t == class_det)
    fn = sum(1 for p, t in zip(y_pred, y_true) if p != class_det and t == class_det)
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score(prec, rec):
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

def macro_precision(list_prec):
    return np.mean(list_prec)

def macro_recall(list_rec):
    return np.mean(list_rec)

def macro_f1(list_f1):
    return np.mean(list_f1)

def weighted_precision(list_prec):
    return 0.2 * list_prec[0] + 0.3 * list_prec[1] + 0.2 * list_prec[2] + 0.3 * list_prec[3]

def weighted_recall(list_rec):
    return 0.2 * list_rec[0] + 0.3 * list_rec[1] + 0.2 * list_rec[2] + 0.3 * list_rec[3]

def weighted_f1(list_f1):
    return 0.2 * list_f1[0] + 0.3 * list_f1[1] + 0.2 * list_f1[2] + 0.3 * list_f1[3]

def classification_report(y_pred, y_true):
    precision_0 = precision_metric(y_pred, y_true, 0)
    recall_0 = recall_metric(y_pred, y_true, 0)
    f1_score_0 = f1_score(precision_0, recall_0)
    support_0 = np.sum(y_true == 0)
    
    precision_1 = precision_metric(y_pred, y_true, 1)
    recall_1 = recall_metric(y_pred, y_true, 1)
    f1_score_1 = f1_score(precision_1, recall_1)
    support_1 = np.sum(y_true == 1)
    
    precision_2 = precision_metric(y_pred, y_true, 2)
    recall_2 = recall_metric(y_pred, y_true, 2)
    f1_score_2 = f1_score(precision_2, recall_2)
    support_2 = np.sum(y_true == 2)
    
    precision_3 = precision_metric(y_pred, y_true, 3)
    recall_3 = recall_metric(y_pred, y_true, 3)
    f1_score_3 = f1_score(precision_3, recall_3)
    support_3 = np.sum(y_true == 3)
    
    correct_predictions = sum(1 for p, t in zip(y_pred, y_true) if p == t)
    accuracy_total = correct_predictions / len(y_true)
    
    macro_avg_precision = macro_precision([precision_0, precision_1, precision_2, precision_3])
    macro_avg_recall = macro_recall([recall_0, recall_1, recall_2, recall_3])
    macro_avg_f1 = macro_f1([f1_score_0, f1_score_1, f1_score_2, f1_score_3])
    
    weighted_avg_precision = weighted_precision([precision_0, precision_1, precision_2, precision_3])
    weighted_avg_recall = weighted_recall([recall_0, recall_1, recall_2, recall_3])
    weighted_avg_f1 = weighted_f1([f1_score_0, f1_score_1, f1_score_2, f1_score_3])
    
    print(f"                precision   recall   f1-score    support")
    print(f"     class 0       {precision_0:.2f}      {recall_0:.2f}      {f1_score_0:.2f}         {support_0}")
    print(f"     class 1       {precision_1:.2f}      {recall_1:.2f}      {f1_score_1:.2f}         {support_1}")
    print(f"     class 2       {precision_2:.2f}      {recall_2:.2f}      {f1_score_2:.2f}         {support_2}")
    print(f"     class 3       {precision_3:.2f}      {recall_3:.2f}      {f1_score_3:.2f}         {support_3}")
    print()
    print(f"    accuracy                           {accuracy_total:.2f}         {len(y_true)}")
    print(f"   macro avg       {macro_avg_precision:.2f}      {macro_avg_recall:.2f}      {macro_avg_f1:.2f}         {len(y_true)}")
    print(f"weighted avg       {weighted_avg_precision:.2f}      {weighted_avg_recall:.2f}      {weighted_avg_f1:.2f}         {len(y_true)}")


# ======================== Enhanced Training & Logging ===========================
model_configs = [
    {
        "name": "Ridge-Regularization",
        "reg": "Ridge",
        "method": "minibatch",
        "lr": 0.001,
        "l_value": 0.1,
        "init_weight": "xavier"
    },
    {
        "name": "LogisticRegression-Normal",
        "reg": "Normal",
        "method": "minibatch",
        "lr": 0.001,
        "l_value": 0.0,
        "init_weight": "xavier"
    }
]

for config in model_configs:
    with mlflow.start_run(run_name=config["name"]):
        # Log parameters
        params = {
            "method": config["method"],
            "lr": config["lr"],
            "reg": config["reg"],
            "init_weight": config["init_weight"],
            "l_value": config["l_value"]
        }
        mlflow.log_params(params)
        
        # Log scaler artifact
        mlflow.log_artifact('scaler.pkl', "preprocessor")
        
        # Initialize model
        if config["reg"] == "Ridge":
            model = Ridge(method=config["method"], lr=config["lr"], l=config["l_value"],
                        n_features=X_train.shape[1], num_class=4)
        else:
            model = LogisticRegression(regularization=None, n_features=X_train.shape[1],
                                     num_class=4, lr=config["lr"], method=config["method"])
        
        # Train model
        model.fit(config["init_weight"], momentum=False, X_train=X_train, y_train=y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        acc = accuracy_metric(y_pred, y_test)
        test_loss = model.loss(np.array([model.one_hot(y) for y in y_test]), X_test @ model.theta)
        
        # Calculate class-wise metrics
        precision = []
        recall = []
        f1_scores = []
        for cls in range(4):
            prec = precision_metric(y_pred, y_test, cls)
            rec = recall_metric(y_pred, y_test, cls)
            f1 = f1_score(prec, rec)
            precision.append(prec)
            recall.append(rec)
            f1_scores.append(f1)
            mlflow.log_metric(f"precision_class_{cls}", prec)
            mlflow.log_metric(f"recall_class_{cls}", rec)
            mlflow.log_metric(f"f1_class_{cls}", f1)
        
        # Log aggregated metrics
        mlflow.log_metrics({
            "test_loss": test_loss,
            "accuracy": acc,
            "f1_macro": np.mean(f1_scores),
            "f1_weighted": weighted_f1(f1_scores),
            "precision_macro": np.mean(precision),
            "recall_macro": np.mean(recall)
        })
        
        # Confusion matrix logging
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {config['name']}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
        plt.close()
        
        # Model logging with signature
        signature = mlflow.models.infer_signature(X_train, y_proba)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=f"{studentID}-{config['reg']}-model"
        )

        from sklearn.metrics import classification_report as sk_classification_report

        # Get scikit-learn classification report and print it
        sk_report = sk_classification_report(y_test, y_pred)
        print("Sklearn Classification Report:")
        print(sk_report)

        # Print your manual classification report for comparison
        print("Manual Classification Report:")
        classification_report(y_pred, y_test)

        # -------------------- NEW: Log Classification Reports as Artifacts --------------------
        # Log the sklearn classification report
        with open("sk_classification_report.txt", "w") as f:
            f.write("Sklearn Classification Report:\n")
            f.write(sk_report)
        mlflow.log_artifact("sk_classification_report.txt", artifact_path="classification_reports")
        
        # Capture and log the manual classification report
        old_stdout = sys.stdout
        sys.stdout = manual_out = StringIO()
        classification_report(y_pred, y_test)
        sys.stdout = old_stdout
        manual_report = manual_out.getvalue()
        with open("manual_classification_report.txt", "w") as f:
            f.write("Manual Classification Report:\n")
            f.write(manual_report)
        mlflow.log_artifact("manual_classification_report.txt", artifact_path="classification_reports")
        # -----------------------------------------------------------------------------------------
