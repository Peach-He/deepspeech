python -m intel_extension_for_pytorch.cpu.launch train.py \
    --train-manifest /root/ht/ML/tmp/training/speech_recognition/libri_train_manifest.csv \
    --val-manifest /root/ht/ML/tmp/training/speech_recognition/libri_train_manifest.csv \
    --batch-size 128