#---------------------------------- MAIN FILE  -------------------------------
# Importing libraries
from fastapi import FastAPI, UploadFile, File, Request
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge
from PIL import Image
import numpy as np
import io, time, psutil

# Importing support functions from python scripts
from load_model import load_model
from predict_digit import predict_digit
from format_image import format_image

# Creating App
app = FastAPI(title="AE20B007: Digit Recognition App for Big Data Lab")
path = 'MNIST_Model.keras' # Specifying path of model

# Prometheus Metrics
request_counter = Counter("api_requests_total", "Total No. of API requests", ["client_ip"])
inference_time_gauge = Gauge("api_inference_time_seconds", "Time taken for Inference in seconds")
processing_time_per_char_gauge = Gauge("api_processing_time_per_char_microseconds", "Processing time per character in microseconds")
network_receive_bytes = Gauge("api_network_receive_bytes", "Total Network receive bytes")
network_transmit_bytes = Gauge("api_network_transmit_bytes", "Total Network transmit bytes")
memory_utilization = Gauge("api_memory_utilization_percent", "API Memory Utilization in percent")
cpu_utilization = Gauge("api_cpu_utilization_percent", "API CPU Utilization in percent")

# Instrumentator for exposing app
Instrumentator().instrument(app).expose(app)

# API Endpoint for predicting digits from images of any size
@app.post("/predict/")
async def predict(request: Request, file: UploadFile = File(...)):
    '''
    Function: API Endpoint which takes the uploaded input image of any size and predicts the digits

    Input:-
    file [File]: File uploaded by user

    Output:-
    [dict]: Key is "digit" and value is the digit predicted by the model
    '''
    client_ip = request.client.host
    request_counter.labels(client_ip=client_ip).inc()
    
    content = await file.read() # Content of uploaded file is read
    image = Image.open(io.BytesIO(content)) # Image is opened as an PIL Image object
    start_time = time.time()
    img = format_image(image) # Image is resized to 28X28 grayscale image

    path = 'MNIST_Model.keras' # Path of the model
    model = load_model(path) # Model is loaded

    digit = predict_digit(model, img) # Digit is predicted

    end_time = time.time()
    
    inference_time = end_time - start_time
    inference_time_gauge.set(inference_time) # Inference time is measured

    # Memory and CPU Utilization is measured
    memory_utilization.set(psutil.virtual_memory().percent)
    cpu_utilization.set(psutil.cpu_percent())

    # Effective processing time per character is measured
    input_length = len(content)  # Use the length of the file contents
    processing_time_per_char = (inference_time*1e6)/input_length  # Convert to microseconds per character
    processing_time_per_char_gauge.set(processing_time_per_char)

    # Network I/O bytes are measured
    net_io = psutil.net_io_counters()
    network_receive_bytes.set(net_io.bytes_recv)
    network_transmit_bytes.set(net_io.bytes_sent)

    return {"digit": digit} # Output is returned