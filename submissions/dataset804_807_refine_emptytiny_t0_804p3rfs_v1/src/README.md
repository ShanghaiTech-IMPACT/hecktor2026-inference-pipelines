# HECKTOR 2026 Dataset804 + Dataset807 Refinement Submission

This Docker context implements `dataset804_807_refine_emptytiny_t0_804p3rfs_v1`.

Pipeline:

1. Read official CT/PET `.mha` inputs from `/input/images/ct` and `/input/images/pet`; `.nii.gz` is still supported for local tests.
2. Orient CT/PET to LPS. CT is the reference grid; PET is resampled only when its grid differs.
3. Generate high-recall HNROI from CT body foreground and PET brain-anchor heuristics.
4. Convert HNROI to Dataset804-style wide HNROI:
   - if z extent is `<= 500 mm`, keep it unchanged;
   - if z extent is `> 500 mm`, run no-label head-end QC;
   - PET brain anchor is used first, CT very-high-intensity components are the backup;
   - if PET/CT evidence is missing or conflicts with expected direction, fallback keeps full HNROI.
5. Run Dataset804 `3d_lowres` nnU-Net folds `0-4` with TTA using `checkpoint_final.pth`.
6. Run Dataset804 `3d_cascade_fullres` nnU-Net folds `0-4` with TTA using `checkpoint_final.pth`.
7. Build Dataset807 LocalROI from the Dataset804 hard mask with margin `(90, 90, 110) mm` and minimum FOV `(200, 200, 240) mm`, matching the Dataset807 training manifest.
8. Run Dataset807 `3d_fullres` refinement folds `0-4` with TTA using `checkpoint_best.pth`; input channels are CT, PET, GTVp hard prior, and GTVn hard prior.
9. If the Dataset807 refined mask is empty/tiny (`<0.05 ml` total/valid component), apply a conservative fallback: low-threshold Dataset804 cascade probabilities, then lowres/cascade hard priors, then low-threshold Dataset807 probabilities.
10. Paste the postprocessed Dataset807 segmentation back to the original CT grid and write `/output/images/head-neck-tumor-segmentation/output.mha`.
11. Extract hard-mask shallow features plus allowed EHR fields from the postprocessed Dataset807 LocalROI result for T/N-stage.
12. Extract Dataset804 cascade features for RFS; this version reuses the old Cascade probtab P3 RFS model and selects only the hard-shallow feature columns it was trained with.
13. Predict T-stage, N-stage, and RFS using the bundled tabular models.

Tabular policy:

- M-stage is not used.
- GT labels, Relapse, and RFS are never used at inference.
- T-stage: Dataset807 post-fallback 5-fold TTA ensemble hard-mask shallow features + EHR + Treatment, 5-seed CatBoost probability ensemble.
- T-stage postprocess: conservative T0 detector can override only T1 to T0 under HPV-positive, tiny-GTVp, high-nodal-burden conditions.
- N-stage: Dataset807 post-fallback 5-fold TTA ensemble hard-mask shallow features + EHR without Treatment, 5-seed CatBoost probability ensemble.
- RFS: Dataset804 cascade hard-mask + soft-probability features + EHR, P3 fusion using 5-seed ExtraTrees relapse classifier plus ExtraSurvivalTrees leaf5.
- The RFS model emits recurrence risk internally; submitted `rfs.json` is sign-flipped as `1 - recurrence_risk`.

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
        ...
      nnUNetTrainer__nnUNetPlans__3d_cascade_fullres/
        dataset.json
        plans.json
        fold_0/checkpoint_final.pth
        ...
    Dataset807_HE2026_FullOfficial_RefineLocalROI_TTACascade_MaskPrior/
      nnUNetTrainer__nnUNetPlans__3d_fullres/
        dataset.json
        plans.json
        fold_0/checkpoint_best.pth
        ...
  tabular/
    t_stage_model.joblib
    n_stage_model.joblib
    rfs_model.joblib
    t0_detector_rule.json
```

Build and package:

```powershell
cd D:\workspace\3D_VLM\docker\hecktor2026_submission_dataset804_807_refine_emptytiny_t0_804p3rfs_v1
powershell -ExecutionPolicy Bypass -File .\prepare_model_bundle.ps1
cd .\Task
powershell -ExecutionPolicy Bypass -File .\do_build.ps1
powershell -ExecutionPolicy Bypass -File .\do_save.ps1
```

Versioned outputs are written to:

- `D:\workspace\3D_VLM\docker\hecktor2026_submission_dataset804_807_refine_emptytiny_t0_804p3rfs_v1\dist\hecktor2026-pipeline-dataset804_807_refine_emptytiny_t0_804p3rfs_v1.tar`
- `D:\workspace\3D_VLM\docker\hecktor2026_submission_dataset804_807_refine_emptytiny_t0_804p3rfs_v1\dist\model-dataset804_807_refine_emptytiny_t0_804p3rfs_v1.tar.gz`

Compatibility filenames are also refreshed:

- `hecktor2026-pipeline.tar`
- `model.tar.gz`

