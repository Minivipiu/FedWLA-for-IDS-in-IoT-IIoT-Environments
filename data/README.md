Federated Intrusion Detection Datasets 🗄️

This directory is intended to store the preprocessed datasets used for evaluating the FedWLA aggregation strategy across different machine learning models (e.g., XGBoost, Random Forest, LightGBM, CatBoost, Decision Tree).

By placing the datasets here at the parent directory level, multiple algorithm-specific repositories can share the same underlying data files without duplicating gigabytes of information.

📊 Evaluated Datasets & Downloads

The FedWLA architecture was rigorously evaluated on four representative, publicly available datasets that cover a wide range of IoT and IIoT network traffic characteristics.

You can download the raw versions of these datasets from the following Kaggle repositories:

1. Edge-IIoTset

Environment: IoT / IIoT

Download Link: https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot

Description: A highly realistic dataset featuring traffic from various devices and multiple attack types. It is specifically designed to reflect realistic edge scenarios, making it highly suitable for evaluating intrusion detection models in both centralised and federated contexts.

2. X-IIoTID

Environment: IIoT

Download Link: https://www.kaggle.com/datasets/munaalhawawreh/xiiotid-iiot-intrusion-dataset

Description: A connectivity-agnostic and device-agnostic dataset focused on Industrial IoT. It provides statistical traffic features, network information, and protocol fields, enabling versatile multiclass evaluation.

3. WUSTL-IIoT-2021

Environment: IIoT

Download Link: https://www.kaggle.com/datasets/annaamalaiu/wustl-iiot-2021-dataset

Description: Based on real traffic captured in industrial environments using multiple protocols (Modbus, OPC UA, MQTT, etc.). It is labelled at the network flow level, comprising benign sessions and simulated attack sequences.

4. IoT-23

Environment: IoT

Download Link: https://www.kaggle.com/datasets/astralfate/iot23-dataset

Description: Contains 23 scenarios of benign and malicious traffic from IoT devices. It heavily features volumetric botnet attacks (such as Mirai and Gafgyt), facilitating robustness analysis against repetitive mass-scale attacks.

⚙️ Preprocessing Pipeline

Please note that the federated simulation code expects these datasets to be strictly preprocessed before execution. Feeding raw data directly into the simulation will result in errors or data leakage.

To faithfully replicate the methodology detailed in the manuscript (Section 4.3), ensure your preprocessing scripts apply the following pipeline to the raw CSV files:

Data Cleaning: Remove all rows containing null (NaN/missing) values and drop exact duplicate records to ensure data integrity.

Identifier Removal: Strip out any session identifiers, IP addresses, MAC addresses, and timestamps to prevent the models from memorising specific sessions (data leakage).

Categorical Encoding: Transform all remaining non-numerical attributes (e.g., categorical flags or text) into a purely numerical format using One-Hot Encoding.

Logarithmic Transformation: Identify numerical attributes exhibiting strong positive or negative skewness (e.g., skewness threshold > 3.0, such as byte counts). Apply a logarithmic scaling (e.g., log1p) to normalise these distributions.

Standardisation: Apply standard scaling (StandardScaler) to all features so that they have a mean of zero and a unit variance ($\mu = 0, \sigma = 1$).

Memory Optimisation: Downcast all 64-bit floating-point and integer columns to 32 bits. This drastically reduces RAM consumption during the distributed federated training process.

Once processed, save the resulting .csv files directly in this folder and update the dataset-path configuration in the pyproject.toml file of the corresponding experiment repository.