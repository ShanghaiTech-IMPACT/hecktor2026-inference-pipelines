# HECKTOR 2026 Dataset804 Probability-Tabular Submission

This Docker context implements `dataset804_probtab_t0_p3rfs_v1`.

Pipeline:

1. Read official CT/PET `.mha` inputs from `/input/images/ct` and `/input/images/pet`; `.nii.gz` remains supported for local tests.
2. Orient CT/PET to LPS. CT is the reference grid; PET is resampled only when its grid differs.
3. Generate a high-recall HNROI from CT body foreground and PET brain-anchor heuristics.
4. Convert the high-recall HNROI to a Dataset804-style wide HNROI:
   - If z extent is `<= 500 mm`, keep it unchanged.
   - If z extent is `> 500 mm`, run no-label head-end QC.
   - PET brain anchor is used first; CT very-high-intensity components are only a backup.
   - If the detected head side is the expected high-z end, keep the superior/head-side `500 mm`.
   - If PET/CT evidence is missing or conflicts with the expected direction, fallback keeps the full HNROI.
5. Run Dataset804 `3d_lowres` nnU-Net folds `0-4` with TTA to generate the previous-stage mask.
6. Run Dataset804 `3d_cascade_fullres` nnU-Net folds `0-4` with TTA on the same wide HNROI, using the lowres mask as cascade input. The final probability map is kept in memory for soft-probability N-stage features.
7. Paste the wide-HNROI prediction back to the original CT grid and write `/output/images/head-neck-tumor-segmentation/output.mha`.
8. Extract shallow mask/intensity features plus allowed EHR fields. T-stage uses hard-mask features; N-stage uses hard-mask plus soft-probability features; RFS uses hard-mask features.
9. Apply the conservative T0 detector to the base T-stage prediction: only selected HPV-positive, tiny-GTVp, high nodal-burden T1 cases can be overridden to T0.

Tabular policy:

- M-stage is not used.
- GT labels, Relapse, and RFS are never used at inference.
- Treatment is allowed by the current official EHR schema and is used where the exported model expects it.
- No 3DINO/SSL image embedding is used in this version.
- T-stage: latest Dataset804 5-fold TTA hard-only shallow features, ExtraTrees, `image_plus_ehr_treatment_no_alcohol`, plus the conservative T0 detector.
- N-stage: latest Dataset804 5-fold TTA hard + soft probability features, CatBoost, `image_plus_ehr_no_treatment_full`.
- RFS: latest Dataset804 5-fold TTA hard-only shallow features, P3 fusion with 5-seed ExtraTrees relapse classifier plus ExtraSurvivalTrees leaf5.
- The RFS model emits recurrence risk internally; the submitted score is sign-flipped as `1 - recurrence_risk`.

Expected model bundle layout:

```text
model/
  VERSION.json
  nnunet/
    Dataset804_HE2026_FullOfficial_WideHNROI_Cascade/
      nnUNetTrainer__nnUNetPlans__3d_lowres/
        dataset.json
        plans.json
        fold_0/checkpoint_final.pth
        fold_1/checkpoint_final.pth
        fold_2/checkpoint_final.pth
        fold_3/checkpoint_final.pth
        fold_4/checkpoint_final.pth
      nnUNetTrainer__nnUNetPlans__3d_cascade_fullres/
        dataset.json
        plans.json
        fold_0/checkpoint_final.pth
        fold_1/checkpoint_final.pth
        fold_2/checkpoint_final.pth
        fold_3/checkpoint_final.pth
        fold_4/checkpoint_final.pth
  tabular/
    t_stage_model.joblib
    n_stage_model.joblib
    rfs_model.joblib
    t0_detector_rule.json
```

Build and package:

```powershell
cd D:\workspace\3D_VLM\docker\hecktor2026_submission_dataset804_probtab_t0_p3rfs_v1
.\prepare_model_bundle.ps1
cd .\Task
.\do_build.ps1
.\do_save.ps1
```

Versioned outputs are written to:

- `D:\workspace\3D_VLM\docker\hecktor2026_submission_dataset804_probtab_t0_p3rfs_v1\dist\hecktor2026-pipeline-dataset804_probtab_t0_p3rfs_v1.tar`
- `D:\workspace\3D_VLM\docker\hecktor2026_submission_dataset804_probtab_t0_p3rfs_v1\dist\model-dataset804_probtab_t0_p3rfs_v1.tar.gz`

Compatibility filenames are also refreshed in the same `dist` directory:

- `hecktor2026-pipeline.tar`
- `model.tar.gz`

Runtime defaults:

- Official entrypoint is single-case sequential inference.
- `HECKTOR_FINE_FOLDS=0,1,2,3,4`
- `HECKTOR_FINE_USE_TTA=1`
- `HECKTOR_FINE_TILE_STEP_SIZE=0.8`
- `HECKTOR_FINE_TILE_BATCH_SIZE=1`
