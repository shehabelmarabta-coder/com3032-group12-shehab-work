# Section 6: Author Contribution Statement

Percentages below reflect actual time committed and artefacts delivered, not equal-by-default weighting.

| Member | Overall % | What they did |
|---|---|---|
| Shomique | 20% | Created the repo. Shipped the shared infrastructure — `src/config.py` target-column fix, `requirements.txt` ML packages, `src/evaluation.py` locked-schema helpers (`recall_at_precision`, `tune_threshold_f1`, `evaluate_model_full` returning the 18-column schema including MCC and FPR). XGBoost + LogReg L2 C=0.1 individual notebook with TreeSHAP. Group Notebook 2 cross-comparison framework. Section 1, Section 5 and Section 6 prose drafting. Group coordination over WhatsApp through the modelling week. |
| Abbas | 20% | Wrote the group preprocessing notebook (`01_Group_Preprocessing.ipynb`): duplicate removal, eight engineered features, stratified 70/15/15 split, scaling, EDA figures. SVM RBF individual notebook with three imbalance strategies and KernelSHAP analysis. Section 2 prose drafting. |
| Enisa | 18% | KNN individual notebook (k tuned across `[3, 5, 7, 11, 15, 21, 31]`) with LogReg L1 (C=1.0) and KernelSHAP. Initial results-CSV schema and §4 cross-comparison work. |
| Shehab | 16% | Project scaffold — folder layout, baseline `src/` helpers, scaffold notebook template, initial `requirements.txt`. Random Forest individual notebook with three imbalance strategies and TreeSHAP analysis. |
| Suheimat | 14% | QDA individual notebook with three imbalance strategies (none, class priors, SMOTE) and the V1–V28-only ablation showing QDA's distributional sensitivity to the engineered binary features. KernelSHAP on sampled points. |
| Zachary | 12% | MLP via `sklearn.MLPClassifier` with class-weighted training, three imbalance strategies, and KernelSHAP on sampled points. |
| **Total** | **100%** | |

Shomique and Abbas contributed disproportionately to the shared infrastructure the rest of the codebase sits on. Enisa took on the §4 cross-comparison integration work that pulls the group's results together. Shehab built the scaffold structure every member's notebook uses. Suheimat and Zachary delivered their individual modelling artefacts.

All members reviewed this statement before submission. Percentages above are subject to final verification by the group lead against artefacts present in the submission package; if any member's modelling artefacts aren't in the final repo at submission time, their percentage is revised downward and the remainder redistributed.
