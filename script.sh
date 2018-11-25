TASK_NAME="flickr8k"
LOG_DIRECTORY="logs/$TASK_NAME"
NUMBER_RUNS=50

mkdir -p "$LOG_DIRECTORY"
for ((i=1; i<=NUMBER_RUNS; i++)); do
	MODEL_DIRECTORY="models/$TASK_NAME/run$i/"
	mkdir -p "$MODEL_DIRECTORY"
	python train.py --task "$TASK_NAME" --model_path "$MODEL_DIRECTORY" > "$LOG_DIRECTORY/run$i.txt"
done
