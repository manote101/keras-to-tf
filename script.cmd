# run trining
docker run --gpus=1 -it --rm -v $(pwd):/workspace nvcr.io/nvidia/tensorflow:22.12-tf2-py3 python train.py

# start model serving
docker run -p 8901:8501 -v $(pwd):/models -e MODEL_NAME=2xplus_one -t tensorflow/serving

# run prediction with curl command
curl -X POST -H "Content-Type: application/json" -d '{"instances": [[5]]}' http://localhost:8901/v1/models/2xplus_one:predict