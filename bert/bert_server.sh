#!/bin/bash
bert-serving-start -model_dir ./cased_L-24_H-1024_A-16/ -num_worker=2 -max_seq_len 450 -pooling_strategy NONE -show_tokens_to_client


