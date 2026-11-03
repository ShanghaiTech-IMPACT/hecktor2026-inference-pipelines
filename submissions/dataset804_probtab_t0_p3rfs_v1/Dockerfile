ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE} AS hecktor2026-task

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/home/user/.local/bin:$PATH \
    nnUNet_raw=/tmp/nnUNet_raw \
    nnUNet_preprocessed=/tmp/nnUNet_preprocessed \
    nnUNet_results=/tmp/nnUNet_results \
    HECKTOR_BATCH_MODE=auto \
    HECKTOR_CPU_WORKERS=3 \
    HECKTOR_LOCALROI_WORKERS=2 \
    HECKTOR_POST_WORKERS=1 \
    HECKTOR_PREFETCH=4 \
    HECKTOR_COARSE_BATCH_SIZE=2 \
    HECKTOR_FINE_TILE_STEP_SIZE=0.8 \
    HECKTOR_FINE_USE_TTA=1 \
    HECKTOR_FINE_FOLDS=0,1,2,3,4

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/requirements.txt
RUN pip install --no-cache-dir -r /opt/app/requirements.txt

ENV HECKTOR_ALLOW_BATCH_ENTRY=0 \
    HECKTOR_BATCH_MODE=sequential \
    HECKTOR_CPU_WORKERS=1 \
    HECKTOR_LOCALROI_WORKERS=1 \
    HECKTOR_POST_WORKERS=0 \
    HECKTOR_PREFETCH=1 \
    HECKTOR_COARSE_BATCH_SIZE=1 \
    HECKTOR_FINE_TILE_STEP_SIZE=0.8 \
    HECKTOR_FINE_USE_TTA=1 \
    HECKTOR_FINE_FOLDS=0,1,2,3,4

COPY --chown=user:user submissions/dataset804_probtab_t0_p3rfs_v1/src/inference.py submissions/dataset804_probtab_t0_p3rfs_v1/src/hecktor_pipeline.py submissions/dataset804_probtab_t0_p3rfs_v1/src/train_hecktor_tn_stage_baseline.py submissions/dataset804_probtab_t0_p3rfs_v1/src/submission_models.py /opt/app/
COPY --chown=user:user weights/dataset804_probtab_t0_p3rfs_v1/model/ /opt/ml/model/

RUN mkdir -p /input /output /tmp/nnUNet_raw /tmp/nnUNet_preprocessed /tmp/nnUNet_results \
    && chown -R user:user /input /output /tmp/nnUNet_raw /tmp/nnUNet_preprocessed /tmp/nnUNet_results /opt/app /opt/ml/model

USER user
ENTRYPOINT ["python", "/opt/app/inference.py"]
