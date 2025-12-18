from prometheus_client import start_http_server, Counter, Gauge, Histogram
import time
import random

# Metrics (10 total)
REQUEST_COUNT = Counter('request_count', 'Total requests')
LATENCY = Histogram('latency_seconds', 'Inference latency')
ERROR_RATE = Counter('error_rate', 'Total errors')
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage')
RAM_USAGE = Gauge('ram_usage_mb', 'RAM usage')
PRED_DIST = Counter('model_prediction_distribution', 'Failure vs No Failure', ['prediction'])
CONFIDENCE_AVG = Gauge('model_confidence_avg', 'Average confidence score')
FAILURE_RATE = Gauge('failure_prediction_rate', 'Rate of failure predicted')
THROUGHPUT = Gauge('inference_throughput', 'Requests per second')
CUSTOM_TEMP_DRIFT = Gauge('sensor_temp_drift', 'Drift in temperature sensor')

def process_request():
    REQUEST_COUNT.inc()
    with LATENCY.time():
        time.sleep(random.uniform(0.01, 0.2))
        pred = random.choice([0, 1])
        PRED_DIST.labels(prediction=str(pred)).inc()
        CONFIDENCE_AVG.set(random.uniform(0.8, 0.99))

if __name__ == '__main__':
    start_http_server(8000)
    while True:
        process_request()
        CPU_USAGE.set(random.uniform(20, 60))
        time.sleep(1)