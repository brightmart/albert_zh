# @Author: bo.shi
# @Date:   2020-03-15 16:11:00
# @Last Modified by:   bo.shi
# @Last Modified time: 2020-04-02 17:54:05
#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
CLUE_DATA_DIR=$CURRENT_DIR/CLUEdataset
ALBERT_TINY_DIR=$CURRENT_DIR/albert_tiny

download_data(){
  TASK_NAME=$1
  if [ ! -d $CLUE_DATA_DIR ]; then
    mkdir -p $CLUE_DATA_DIR
    echo "makedir $CLUE_DATA_DIR"
  fi
  cd $CLUE_DATA_DIR
  if [ ! -d ${TASK_NAME} ]; then
    mkdir $TASK_NAME
    echo "make dataset dir $CLUE_DATA_DIR/$TASK_NAME"
  fi
  cd $TASK_NAME
  if [ ! -f "train.json" ] || [ ! -f "dev.json" ] || [ ! -f "test.json" ]; then
    rm *
    wget https://storage.googleapis.com/cluebenchmark/tasks/${TASK_NAME}_public.zip
    unzip ${TASK_NAME}_public.zip
    rm ${TASK_NAME}_public.zip
  else
    echo "data exists"
  fi
  echo "Finish download dataset."
}

download_model(){
  if [ ! -d $ALBERT_TINY_DIR ]; then
    mkdir -p $ALBERT_TINY_DIR
    echo "makedir $ALBERT_TINY_DIR"
  fi
  cd $ALBERT_TINY_DIR
  if [ ! -f "albert_config_tiny.json" ] || [ ! -f "vocab.txt" ] || [ ! -f "checkpoint" ] || [ ! -f "albert_model.ckpt.index" ] || [ ! -f "albert_model.ckpt.meta" ] || [ ! -f "albert_model.ckpt.data-00000-of-00001" ]; then
    rm *
    wget -c https://storage.googleapis.com/albert_zh/albert_tiny_489k.zip
    unzip albert_tiny_489k.zip
    rm albert_tiny_489k.zip
  else
    echo "model exists"
  fi
  echo "Finish download model."
}

run_task() {
  TASK_NAME=$1
  download_data $TASK_NAME
  download_model $MODEL_NAME
  DATA_DIR=$CLUE_DATA_DIR/${TASK_NAME}
  PREV_TRAINED_MODEL_DIR=$ALBERT_TINY_DIR
  MAX_SEQ_LENGTH=$2
  TRAIN_BATCH_SIZE=$3
  LEARNING_RATE=$4
  NUM_TRAIN_EPOCHS=$5
  SAVE_CHECKPOINTS_STEPS=$6
  OUTPUT_DIR=$CURRENT_DIR/${TASK_NAME}_output/
  COMMON_ARGS="
        --task_name=$TASK_NAME \
        --data_dir=$DATA_DIR \
        --vocab_file=$PREV_TRAINED_MODEL_DIR/vocab.txt \
        --bert_config_file=$PREV_TRAINED_MODEL_DIR/albert_config_tiny.json \
        --init_checkpoint=$PREV_TRAINED_MODEL_DIR/albert_model.ckpt \
        --max_seq_length=$MAX_SEQ_LENGTH \
        --train_batch_size=$TRAIN_BATCH_SIZE \
        --learning_rate=$LEARNING_RATE \
        --num_train_epochs=$NUM_TRAIN_EPOCHS \
        --save_checkpoints_steps=$SAVE_CHECKPOINTS_STEPS \
        --output_dir=$OUTPUT_DIR \
        --keep_checkpoint_max=0 \
  "
  cd $CURRENT_DIR
  echo "Start running..."
  python run_classifier_clue.py \
        $COMMON_ARGS \
        --do_train=true \
        --do_eval=false \
        --do_predict=false

  echo "Start predict..."
  python run_classifier_clue.py \
        $COMMON_ARGS \
        --do_train=false \
        --do_eval=true \
        --do_predict=true
}

##command##task_name##model_name##max_seq_length##train_batch_size##learning_rate##num_train_epochs##save_checkpoints_steps##tpu_ip
run_task afqmc 128 16 2e-5 3 300
run_task cmnli 128 64 3e-5 2 300
run_task csl 128 16 1e-5 5 100
run_task iflytek 128 32 2e-5 3 300
run_task tnews 128 16 2e-5 3 300
run_task wsc 128 16 1e-5 10 10