pid=$(sudo lsof /dev/nvidia-uvm | awk '/ollama/ {print $2}' | head -n1)

sudo kill -9 $pid

sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm

/home/suga/projects/llama.cpp/build/bin/llama-server -m models/Qwen3.5-9B-UD-IQ2_XXS.gguf --host 0.0.0.0 --port 8080 -ngl 99
#
