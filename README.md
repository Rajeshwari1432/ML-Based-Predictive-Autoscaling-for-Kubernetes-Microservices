# ML-Based Predictive Autoscaling for Kubernetes Microservices

# Introduction
Kubernetes Horizontal Pod Autoscaler (HPA) is the default mechanism for scaling microservices based on resource usage (CPU, memory). HPA is reactive—it scales only after load increases, which can lead to temporary SLA violations and inefficient resource usage.

This project implements a **predictive autoscaler** using machine learning (ML) to forecast future resource demand and scale pods proactively, improving latency, reducing SLA breaches, and optimizing resource allocation.

# Problem & Motivation
- HPA is reactive, causing temporary SLA breaches and resource waste.
- Predictive autoscaling aims to forecast future CPU/memory demand and scale proactively, improving tail-latency and resource efficiency compared to HPA.

# Objectives
- Build ML models (LSTM/GRU, XGBoost, optional hybrid) to predict CPU/memory usage (30–120s horizons).
- Implement a predictive autoscaler (controller/operator + inference service).
- Evaluate vs HPA baseline on DeathStarBench under varied loads, reporting p95/p99 latency, SLA violation %, resource waste, scaling oscillations, and autoscaler overhead.

# Datasets
  # Offline (Training/EDA)
- Google Cluster Workload Trace (ClusterData2011_2, v2.1)
  - Download: `gsutil cp gs://clusterdata-2011-2/task_usage/part-00000-of-00500.csv.gz .`
  - Optional: second shard/events for deeper analysis.

  # Online (Experiments)
- Prometheus + cAdvisor metrics during DeathStarBench runs (CPU, memory, replicas).

# Tools & Technologies
- Kubernetes, HPA
- Prometheus, cAdvisor
- Python, Pandas, NumPy
- TensorFlow/Keras, XGBoost
- Scikit-learn
- Locust, hey (load generation)
- Matplotlib (visualization)
- Docker, Helm, kubectl

# Setup & Prerequisites
- Python 3.8+
- Kubernetes cluster (Minikube or other)
- Prometheus and cAdvisor installed
- Docker, Helm, kubectl
- ML libraries: TensorFlow, XGBoost, Scikit-learn, Pandas, NumPy
- gsutil (for Google Cluster data download)
- Git (for version control)

# Methodology
  # 1. Data Collection
   - Download Google Cluster Workload Trace for offline model training:
     `gsutil cp gs://clusterdata-2011-2/task_usage/part-00000-of-00500.csv.gz .`
   - Collect real-time metrics from Prometheus/cAdvisor during Kubernetes workload runs.

  # 2. Feature Engineering
   - Aggregate metrics to per-minute time series.
   - Create sliding windows, lags, rolling stats, and time-based features.

  # 3. Model Training
   - Train LSTM/GRU and XGBoost models to predict future CPU/memory usage.
   - Validate using rolling-origin and statistical tests.

  # 4. Autoscaler Implementation
   - Build inference service and controller/operator logic for proactive scaling.

  # 5. Evaluation
   - Run DeathStarBench workloads, generate traffic with Locust/hey.
   - Compare predictive autoscaler vs. HPA on latency, SLA violations, resource waste, and scaling oscillations.

# Benchmarks & Evaluation
- DeathStarBench (Social Network or Hotel Reservation)
- Locust load generator: burst, ramp, diurnal profiles
- Metrics: p95/p99 latency, SLA violation %, resource waste, scaling oscillations, autoscaler overhead
- Success: Statistically significant improvement vs HPA (α=0.05, 95% CI, ≥5–10 runs)

# Data Collection Guide
  # Offline Data (Google ClusterData2011_2)
  1. Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install
  2. Download required files:
   `gsutil cp gs://clusterdata-2011-2/task_usage/part-00000-of-00500.csv.gz .`
   # Optional:
   `gsutil cp gs://clusterdata-2011-2/task_usage/part-00001-of-00500.csv.gz .`
   `gsutil cp gs://clusterdata-2011-2/task_events/part-00000-of-00500.csv.gz .`
  3. Do not unzip fully. Stream or chunk-read files:
   `gunzip -c part-00000-of-00500.csv.gz | head -n 20`
  4. Log for paper: Dataset name, trace version, time span, files used, total size collected.

# Online Data (Kubernetes Metrics)
  1. Deploy DeathStarBench microservices on Kubernetes.
  2. Install Prometheus + cAdvisor for metrics collection.
  3. Generate traffic using Locust (burst, ramp, diurnal workloads).
  4. Collect metrics:
   - CPU usage
   - Memory working set
   - Pod replica count
   - Scaling events
   - Autoscaler resource usage
  5. Save for each run:
   - Prometheus time-series exports
   - Locust statistics CSVs
   - Kubernetes replica logs
   - Autoscaler decision logs

# Online HPA Data Collection
  See `hpa_data_collection.txt` for step-by-step instructions to collect HPA metrics using Minikube, Docker, Prometheus, and a load generator.

# Skills & Learning Outcomes
  - Time-series forecasting
  - ML model development and evaluation
  - Feature engineering for resource metrics
  - Kubernetes autoscaling and monitoring
  - Data collection and reproducibility
  - MLOps pipeline design and deployment
  - GitHub repository management

# Importance & MLOps Value
This project demonstrates a practical MLOps workflow: data collection, feature engineering, model training, deployment, and real-world evaluation. It addresses a real DevOps challenge—proactive   autoscaling—and shows how ML can improve cloud resource efficiency and reliability.

# Future Improvements
  - Integrate more advanced models (e.g., attention, hybrid ensembles)
  - Expand to multi-metric and multi-service scaling
  - Add anomaly detection and fallback logic
  - Automate data collection and evaluation pipelines
  - Deploy as a Kubernetes operator for production use

# Conclusion
This project provides a robust, reproducible framework for ML-based predictive autoscaling in Kubernetes, outperforming HPA in real-world scenarios. It is a strong example of applied MLOps, combining ML, DevOps, and cloud-native skills.

# References
  - Google ClusterData2011_2: https://github.com/google/cluster-data
  - Kubernetes HPA docs: https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/
  - Prometheus: https://prometheus.io/docs/introduction/overview/
  - cAdvisor: https://github.com/google/cadvisor
  - DeathStarBench: https://github.com/delimitrou/DeathStarBench
  - Locust: https://docs.locust.io/

---
