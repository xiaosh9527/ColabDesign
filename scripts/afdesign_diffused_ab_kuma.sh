#!/bin/bash

module load gcc/13.2.0
module load cuda/12.4.1
module load cudnn/9.2.1.18-12

python -u design_with_em_constraints.py \
        --seed=0_42 \
        --num_diffusion_designs=3 \
        --num_afdesign_designs=5 \
        --num_mpnn_designs=5 \
        --learning_rate=0.02 \
        --target_pdb=./examples/7lo8_Z.pdb \
        --target_chain=Z \
        --target_hotspot=227 \
        --binder_pdb=./examples/7lo8_HL.pdb \
        --binder_chain=H,L \
        --rm_aa=C \
        --target_flexible \
        --rm_template_ic \
        --use_binder_template \
        --use_mpnn_loss \
        --use_multimer \
        --mpnn_weights abmpnn \
        --mpnn_model_name abmpnn
