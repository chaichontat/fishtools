from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

from fishtools.brdu.features import (
    META_REQUIRED_COLS,
    ensure_log_means,
    feature_names,
    features_array_from_frame,
    features_array_from_row,
)
from fishtools.brdu.model import BrduEduModelBundle, save_model_bundle


@dataclass(frozen=True)
class ModelStatus:
    ok: bool
    reason: str


def _labels_signature(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return (int(st.st_mtime_ns), int(st.st_size))


def _model_params_table(
    model: RandomForestClassifier, *, target: str
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    names = feature_names(target)
    importances = np.asarray(model.feature_importances_, dtype=np.float64)
    df = pd.DataFrame({"feature": names, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    info: dict[str, float | int | str] = {
        "model": "RandomForestClassifier",
        "n_estimators": int(model.n_estimators),
        "random_state": int(model.random_state) if isinstance(model.random_state, int) else 0,
    }
    return df, info


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _load_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["cell", "brdu", "edu"])
    df = pd.read_csv(path)
    required = {"cell", "brdu", "edu"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"labels csv missing required columns: {missing}")
    df = df[["cell", "brdu", "edu"]].copy()
    df["cell"] = df["cell"].astype(str)
    df["brdu"] = df["brdu"].astype(int)
    df["edu"] = df["edu"].astype(int)
    df = df.drop_duplicates(subset=["cell"], keep="last").reset_index(drop=True)
    return df


def _fit_binary_model(
    *,
    meta: pd.DataFrame,
    labels: pd.DataFrame,
    target: str,
) -> tuple[RandomForestClassifier | None, ModelStatus]:
    if target not in {"brdu", "edu"}:
        raise ValueError(f"Unexpected target: {target!r}")
    if labels.empty:
        return None, ModelStatus(ok=False, reason="no labels yet")

    joined = labels.merge(
        meta[
            [
                "cell",
                "brdu_mean",
                "brdu_min",
                "brdu_max",
                "brdu_median",
                "log_brdu_mean",
                "log_brdu_median",
                "log_brdu_min",
                "log_brdu_max",
                "brdu_std",
                "edu_mean",
                "edu_min",
                "edu_max",
                "edu_median",
                "log_edu_mean",
                "log_edu_median",
                "log_edu_min",
                "log_edu_max",
                "edu_std",
            ]
        ],
        on="cell",
        how="inner",
    )
    if joined.empty:
        return None, ModelStatus(ok=False, reason="no labeled cells in barrage index")

    y = joined[target].to_numpy(dtype=int)
    if np.unique(y).size < 2:
        return None, ModelStatus(ok=False, reason=f"need both classes for {target} model")

    X = features_array_from_frame(joined, target=target)
    model = RandomForestClassifier(
        n_estimators=400,
        random_state=0,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X, y)
    return model, ModelStatus(ok=True, reason=f"fit on n={len(joined)}")


def _score_unlabeled(
    *,
    meta: pd.DataFrame,
    labels: pd.DataFrame,
    brdu_model: RandomForestClassifier | None,
    edu_model: RandomForestClassifier | None,
    brdu_threshold: float,
    edu_threshold: float,
) -> pd.DataFrame:
    labeled_cells = set(labels["cell"].astype(str).tolist())
    unlabeled = meta[~meta["cell"].isin(labeled_cells)].copy()
    if unlabeled.empty:
        return unlabeled

    if brdu_model is not None:
        X_brdu = features_array_from_frame(unlabeled, target="brdu")
        p = brdu_model.predict_proba(X_brdu)[:, 1]
        unlabeled["p_brdu"] = p
    else:
        unlabeled["p_brdu"] = np.nan

    if edu_model is not None:
        X_edu = features_array_from_frame(unlabeled, target="edu")
        p = edu_model.predict_proba(X_edu)[:, 1]
        unlabeled["p_edu"] = p
    else:
        unlabeled["p_edu"] = np.nan

    if brdu_model is not None and edu_model is not None:
        # AND gate: prioritize cells near both decision thresholds.
        # (Minimizing the max distance forces both distances to be small.)
        score = np.maximum(
            np.abs(unlabeled["p_brdu"].to_numpy(dtype=np.float64) - brdu_threshold),
            np.abs(unlabeled["p_edu"].to_numpy(dtype=np.float64) - edu_threshold),
        )
    elif brdu_model is not None:
        score = np.abs(unlabeled["p_brdu"].to_numpy(dtype=np.float64) - brdu_threshold)
    elif edu_model is not None:
        score = np.abs(unlabeled["p_edu"].to_numpy(dtype=np.float64) - edu_threshold)
    else:
        score = np.full((len(unlabeled),), np.inf, dtype=np.float64)

    unlabeled["uncertainty"] = score
    return unlabeled


def _score_labeled_for_audit(
    *,
    meta: pd.DataFrame,
    labels: pd.DataFrame,
    brdu_model: RandomForestClassifier | None,
    edu_model: RandomForestClassifier | None,
) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame()

    labeled = labels.merge(meta, on="cell", how="inner")
    if labeled.empty:
        return labeled

    if brdu_model is not None:
        X_brdu = features_array_from_frame(labeled, target="brdu")
        p = brdu_model.predict_proba(X_brdu)[:, 1]
        labeled["p_brdu"] = p
        brdu = labeled["brdu"].to_numpy(dtype=np.int64)
        labeled["brdu_mismatch"] = np.where(brdu == 1, 1.0 - p, p)
    else:
        labeled["p_brdu"] = np.nan
        labeled["brdu_mismatch"] = np.nan

    if edu_model is not None:
        X_edu = features_array_from_frame(labeled, target="edu")
        p = edu_model.predict_proba(X_edu)[:, 1]
        labeled["p_edu"] = p
        edu = labeled["edu"].to_numpy(dtype=np.int64)
        labeled["edu_mismatch"] = np.where(edu == 1, 1.0 - p, p)
    else:
        labeled["p_edu"] = np.nan
        labeled["edu_mismatch"] = np.nan

    labeled["audit_score"] = np.nanmax(
        np.column_stack([labeled["brdu_mismatch"].to_numpy(), labeled["edu_mismatch"].to_numpy()]),
        axis=1,
    )
    return labeled


def _suggest_next_cell(
    *,
    meta: pd.DataFrame,
    labels: pd.DataFrame,
    brdu_model: RandomForestClassifier | None,
    edu_model: RandomForestClassifier | None,
    brdu_threshold: float,
    edu_threshold: float,
) -> str | None:
    if meta.empty:
        return None

    scored = _score_unlabeled(
        meta=meta,
        labels=labels,
        brdu_model=brdu_model,
        edu_model=edu_model,
        brdu_threshold=brdu_threshold,
        edu_threshold=edu_threshold,
    )
    if scored.empty:
        return None

    if brdu_model is None and edu_model is None:
        # No model yet: follow the barrage order until we have enough labels to fit.
        return str(scored.sort_values("rank", kind="stable").iloc[0]["cell"])

    return str(scored.sort_values(["uncertainty", "rank"], kind="stable").iloc[0]["cell"])


def _label_tab_cells(*, meta: pd.DataFrame, labels: pd.DataFrame) -> list[str]:
    labeled_cells = set(labels["cell"].astype(str).tolist())
    return meta.loc[~meta["cell"].isin(labeled_cells), "cell"].astype(str).tolist()


def _label_picker_cells(*, available_cells: list[str], current_cell: str) -> list[str]:
    if current_cell in available_cells:
        return available_cells
    return [current_cell, *available_cells]


def _label_action_specs() -> list[tuple[str, int, int]]:
    return [
        ("BrdU only", 1, 0),
        ("EdU only", 0, 1),
        ("Dual pos", 1, 1),
        ("None", 0, 0),
    ]


def _audit_top_n_slider_args(n_cells: int) -> tuple[int, int, int, int] | None:
    max_top_n = min(200, n_cells)
    if max_top_n <= 10:
        return None
    return 10, max_top_n, min(50, max_top_n), 10


def main() -> None:
    st.set_page_config(page_title="BrdU/EdU Labeler", layout="wide")
    st.title("BrdU/EdU cell labeler")

    barrage_dir = Path(st.sidebar.text_input("Barrage dir", "output/brdu_edu_barrage")).resolve()
    index_path = barrage_dir / "index.csv"
    if not index_path.exists():
        st.error(f"Missing index.csv at: {index_path}")
        st.stop()

    labels_path = Path(st.sidebar.text_input("Labels CSV", str(barrage_dir / "labels.csv"))).resolve()

    brdu_threshold = float(st.sidebar.slider("BrdU threshold (prob)", 0.0, 1.0, 0.5, 0.01))
    edu_threshold = float(st.sidebar.slider("EdU threshold (prob)", 0.0, 1.0, 0.5, 0.01))

    meta = pd.read_csv(index_path)
    required = {"rank", "cell", "image_path", *META_REQUIRED_COLS}
    missing = sorted(required - set(meta.columns))
    if missing:
        st.error(f"index.csv missing required columns: {missing}")
        st.stop()
    meta["cell"] = meta["cell"].astype(str)
    ensure_log_means(meta)
    meta_by_cell = meta.set_index("cell")

    labels = _load_labels(labels_path)

    st.sidebar.subheader("Model training")
    labels_sig_now = _labels_signature(labels_path)
    trained = st.session_state.get("trained_models")
    trained_sig = trained.get("labels_sig") if isinstance(trained, dict) else None
    has_cached = (
        isinstance(trained, dict)
        and isinstance(trained.get("brdu_model"), RandomForestClassifier)
        and isinstance(trained.get("edu_model"), RandomForestClassifier)
    )
    stale = bool(has_cached and labels_sig_now is not None and trained_sig is not None and labels_sig_now != trained_sig)

    if has_cached:
        st.sidebar.write(f"Trained at: {trained.get('trained_at_utc', '(unknown)')}")
        st.sidebar.write(f"Train rows: {trained.get('train_rows', '(unknown)')}")
        if stale:
            st.sidebar.warning("Model is stale (labels changed since last training).")
    else:
        st.sidebar.write("No cached model in this session yet.")

    retrain = st.sidebar.button("Train/Refresh models")

    if retrain or (not has_cached and not labels.empty):
        with st.spinner("Training models (this is not automatic after each label)…"):
            brdu_model, brdu_status = _fit_binary_model(meta=meta, labels=labels, target="brdu")
            edu_model, edu_status = _fit_binary_model(meta=meta, labels=labels, target="edu")
            if brdu_model is not None and edu_model is not None:
                joined = labels.merge(meta, on="cell", how="inner")
                st.session_state["trained_models"] = {
                    "brdu_model": brdu_model,
                    "edu_model": edu_model,
                    "labels_sig": labels_sig_now,
                    "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                    "train_rows": int(len(joined)),
                }
    else:
        brdu_model = trained.get("brdu_model") if isinstance(trained, dict) else None
        edu_model = trained.get("edu_model") if isinstance(trained, dict) else None
        brdu_status = ModelStatus(ok=brdu_model is not None, reason="cached (not retrained)")
        edu_status = ModelStatus(ok=edu_model is not None, reason="cached (not retrained)")

    st.sidebar.write(f"Model(brdu): {brdu_status.reason}")
    st.sidebar.write(f"Model(edu): {edu_status.reason}")

    total = len(meta)
    labeled_n = int(labels["cell"].nunique()) if not labels.empty else 0
    st.sidebar.write(f"Labeled: {labeled_n}/{total}")

    suggested = _suggest_next_cell(
        meta=meta,
        labels=labels,
        brdu_model=brdu_model,
        edu_model=edu_model,
        brdu_threshold=brdu_threshold,
        edu_threshold=edu_threshold,
    )

    # Show why the suggested cell is considered "informative"
    try:
        row_s = meta_by_cell.loc[str(suggested)] if suggested is not None else None
    except KeyError:
        row_s = None

    if row_s is not None and suggested is not None:
        x_brdu_s = features_array_from_row(row_s, target="brdu")
        x_edu_s = features_array_from_row(row_s, target="edu")
        p_brdu_s = float(brdu_model.predict_proba(x_brdu_s)[:, 1][0]) if brdu_model is not None else np.nan
        p_edu_s = float(edu_model.predict_proba(x_edu_s)[:, 1][0]) if edu_model is not None else np.nan

        st.sidebar.write(f"Suggested: {suggested}")
        if np.isfinite(p_brdu_s):
            st.sidebar.write(f"p_brdu={p_brdu_s:.3f} (thr={brdu_threshold:.2f})")
        if np.isfinite(p_edu_s):
            st.sidebar.write(f"p_edu={p_edu_s:.3f} (thr={edu_threshold:.2f})")

    st.sidebar.subheader("Model feature importance")
    if brdu_model is not None:
        params_df, info = _model_params_table(brdu_model, target="brdu")
        st.sidebar.write("brdu")
        st.sidebar.dataframe(params_df, hide_index=True, use_container_width=True)
        st.sidebar.write(info)
    else:
        st.sidebar.write("brdu: (not fit yet)")

    if edu_model is not None:
        params_df, info = _model_params_table(edu_model, target="edu")
        st.sidebar.write("edu")
        st.sidebar.dataframe(params_df, hide_index=True, use_container_width=True)
        st.sidebar.write(info)
    else:
        st.sidebar.write("edu: (not fit yet)")

    st.sidebar.subheader("Export model")
    model_out_default = barrage_dir / "brdu_edu_model.joblib"
    model_out = Path(st.sidebar.text_input("Model output path", str(model_out_default))).resolve()
    if st.sidebar.button("Write model (.joblib)", disabled=brdu_model is None or edu_model is None):
        joined = labels.merge(meta, on="cell", how="inner")
        brdu_y = joined["brdu"].to_numpy(dtype=int)
        edu_y = joined["edu"].to_numpy(dtype=int)
        brdu_counts = dict(zip(*np.unique(brdu_y, return_counts=True), strict=True))
        edu_counts = dict(zip(*np.unique(edu_y, return_counts=True), strict=True))

        bundle = BrduEduModelBundle(
            feature_version=1,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            seed=0,
            n_estimators=int(brdu_model.n_estimators),
            train_rows=int(len(joined)),
            brdu_class_counts={int(k): int(v) for k, v in brdu_counts.items()},
            edu_class_counts={int(k): int(v) for k, v in edu_counts.items()},
            brdu_model=brdu_model,
            edu_model=edu_model,
        )
        save_model_bundle(bundle, path=model_out)
        st.sidebar.success(f"Wrote model: {model_out}")

    label_tab, review_tab, audit_tab = st.tabs(["Label", "Review", "Audit"])

    with label_tab:
        if suggested is None:
            st.success("All cells labeled.")
        else:
            if "current_cell" not in st.session_state:
                st.session_state["current_cell"] = suggested

            available_cells = _label_tab_cells(meta=meta, labels=labels)
            if not available_cells:
                st.success("All cells labeled.")
                return
            meta_cells = set(meta["cell"].astype(str).tolist())
            if st.session_state.get("current_cell") not in meta_cells:
                st.session_state["current_cell"] = available_cells[0]

            current_cell = str(st.session_state["current_cell"])
            picker_cells = _label_picker_cells(available_cells=available_cells, current_cell=current_cell)
            label_prev_cells: list[str] = st.session_state.setdefault("label_prev_cells", [])

            with st.sidebar:
                st.write("Navigation")
                if st.button("Jump to suggested"):
                    if not label_prev_cells or label_prev_cells[-1] != current_cell:
                        label_prev_cells.append(current_cell)
                    st.session_state["current_cell"] = suggested
                    st.rerun()

                picked = st.selectbox(
                    "Pick cell",
                    options=picker_cells,
                    index=picker_cells.index(current_cell),
                )
                if picked != current_cell:
                    if not label_prev_cells or label_prev_cells[-1] != current_cell:
                        label_prev_cells.append(current_cell)
                    st.session_state["current_cell"] = str(picked)
                    st.rerun()

            row = meta_by_cell.loc[current_cell]
            img_path = (barrage_dir / str(row["image_path"])).resolve()

            col1, col2 = st.columns([2, 1], gap="large")

            x_brdu_current = features_array_from_row(row, target="brdu")
            x_edu_current = features_array_from_row(row, target="edu")
            p_brdu_current = (
                float(brdu_model.predict_proba(x_brdu_current)[:, 1][0]) if brdu_model is not None else np.nan
            )
            p_edu_current = float(edu_model.predict_proba(x_edu_current)[:, 1][0]) if edu_model is not None else np.nan

            with col1:
                st.subheader(current_cell)
                if not img_path.exists():
                    st.error(f"Missing image: {img_path}")
                else:
                    st.image(str(img_path), use_container_width=True)

            with col2:
                st.subheader("Scores")
                m1, m2 = st.columns(2)
                with m1:
                    with st.container(border=True):
                        if np.isfinite(p_brdu_current):
                            st.metric(
                                f"BrdU p (thr={brdu_threshold:.2f})",
                                f"{p_brdu_current:.3f}",
                            )
                        else:
                            st.metric("BrdU p", "n/a")
                with m2:
                    with st.container(border=True):
                        if np.isfinite(p_edu_current):
                            st.metric(
                                f"EdU p (thr={edu_threshold:.2f})",
                                f"{p_edu_current:.3f}",
                            )
                        else:
                            st.metric("EdU p", "n/a")

                st.subheader("Stats")
                brdu_mean = float(row["brdu_mean"])
                brdu_std = float(row["brdu_std"])
                edu_mean = float(row["edu_mean"])
                edu_std = float(row["edu_std"])
                brdu_cv_pct = (
                    100.0 * brdu_std / brdu_mean if np.isfinite(brdu_mean) and brdu_mean > 0 else np.nan
                )
                edu_cv_pct = 100.0 * edu_std / edu_mean if np.isfinite(edu_mean) and edu_mean > 0 else np.nan
                st.dataframe(
                    pd.DataFrame(
                        {
                            "brdu_mean": [brdu_mean],
                            "log_brdu_mean": [float(row["log_brdu_mean"])],
                            "brdu_std": [brdu_std],
                            "brdu_cv_pct": [brdu_cv_pct],
                            "edu_mean": [edu_mean],
                            "log_edu_mean": [float(row["log_edu_mean"])],
                            "edu_std": [edu_std],
                            "edu_cv_pct": [edu_cv_pct],
                            "p_brdu": [p_brdu_current],
                            "p_edu": [p_edu_current],
                        }
                    )
                )

                prev_col, actions_col = st.columns([1, 3])
                with prev_col:
                    if st.button("Previous", disabled=len(label_prev_cells) == 0, key="label_prev"):
                        st.session_state["current_cell"] = label_prev_cells.pop()
                        st.rerun()
                with actions_col:
                    st.write("Save label")
                    action_cols = st.columns(2)
                    for idx, (label_name, brdu_val, edu_val) in enumerate(_label_action_specs()):
                        with action_cols[idx % 2]:
                            if st.button(label_name, type="primary", key=f"label_action_{current_cell}_{label_name}"):
                                if not label_prev_cells or label_prev_cells[-1] != current_cell:
                                    label_prev_cells.append(current_cell)
                                new_row = pd.DataFrame([{"cell": current_cell, "brdu": brdu_val, "edu": edu_val}])
                                labels2 = labels[labels["cell"] != current_cell].copy()
                                labels2 = pd.concat([labels2, new_row], ignore_index=True)
                                labels2 = labels2.sort_values("cell", kind="stable").reset_index(drop=True)
                                _atomic_write_csv(labels2, labels_path)
                                del st.session_state["current_cell"]
                                st.rerun()

    with review_tab:
        st.subheader("Review / edit existing labels")
        if labels.empty:
            st.info("No labels yet.")
        else:
            label_groups: dict[str, tuple[int, int]] = {
                "BrdU-/EdU- (0,0)": (0, 0),
                "BrdU+/EdU- (1,0)": (1, 0),
                "BrdU-/EdU+ (0,1)": (0, 1),
                "BrdU+/EdU+ (1,1)": (1, 1),
            }
            group_name = st.selectbox("Group", options=list(label_groups.keys()), key="review_group")
            brdu_g, edu_g = label_groups[group_name]

            group = labels[(labels["brdu"] == brdu_g) & (labels["edu"] == edu_g)].copy()
            if group.empty:
                st.info("No cells in this group yet.")
            else:
                group = (
                    group.merge(meta[["cell", "rank", "image_path"]], on="cell", how="left")
                    .sort_values(["rank", "cell"], kind="stable")
                    .reset_index(drop=True)
                )
                cells = group["cell"].astype(str).tolist()

                if st.session_state.get("review_cell") not in set(cells):
                    st.session_state["review_cell"] = cells[0]

                next_cell = st.session_state.get("review_next_cell")
                if isinstance(next_cell, str) and next_cell in set(cells):
                    st.session_state["review_cell"] = next_cell
                    del st.session_state["review_next_cell"]

                i = int(cells.index(str(st.session_state["review_cell"])))
                left, right = st.columns([1, 3], gap="large")
                with left:
                    st.write(f"{i + 1}/{len(cells)}")
                    if st.button("Prev", disabled=i == 0):
                        st.session_state["review_cell"] = cells[max(i - 1, 0)]
                        st.rerun()
                    if st.button("Next", disabled=i >= (len(cells) - 1)):
                        st.session_state["review_cell"] = cells[min(i + 1, len(cells) - 1)]
                        st.rerun()

                    st.selectbox("Cell", options=cells, index=i, key="review_cell")

                    cell = str(st.session_state["review_cell"])
                    row_l = labels.set_index("cell").loc[cell]
                    brdu_val = int(
                        st.radio(
                            "BrdU",
                            options=[0, 1],
                            horizontal=True,
                            index=int(row_l["brdu"]),
                            key=f"review_brdu_{cell}",
                        )
                    )
                    edu_val = int(
                        st.radio(
                            "EdU",
                            options=[0, 1],
                            horizontal=True,
                            index=int(row_l["edu"]),
                            key=f"review_edu_{cell}",
                        )
                    )

                    if st.button("Save edit", type="primary"):
                        new_row = pd.DataFrame([{"cell": cell, "brdu": brdu_val, "edu": edu_val}])
                        labels2 = labels[labels["cell"] != cell].copy()
                        labels2 = pd.concat([labels2, new_row], ignore_index=True)
                        labels2 = labels2.sort_values("cell", kind="stable").reset_index(drop=True)
                        _atomic_write_csv(labels2, labels_path)
                        st.session_state["review_next_cell"] = cells[min(i + 1, len(cells) - 1)]
                        st.rerun()

                with right:
                    cell = str(st.session_state["review_cell"])
                    if cell in meta_by_cell.index:
                        row_m = meta_by_cell.loc[cell]
                        img_path = (barrage_dir / str(row_m["image_path"])).resolve()
                        if img_path.exists():
                            st.image(str(img_path), use_container_width=True)
                        else:
                            st.error(f"Missing image: {img_path}")
                    else:
                        st.error("Cell not found in index.csv (barrage index).")

    with audit_tab:
        st.subheader("Audit likely mistakes")
        st.write(
            "Cells are ranked by how strongly the current model disagrees with the saved label "
            "(higher = more suspicious)."
        )

        if labels.empty:
            st.info("No labels yet.")
        elif brdu_model is None and edu_model is None:
            st.info("Need enough labels to fit a model before we can score likely mistakes.")
        else:
            score_mode = st.selectbox(
                "Score mode",
                options=["max(BrdU, EdU)", "BrdU only", "EdU only"],
                key="audit_score_mode",
            )

            scored = _score_labeled_for_audit(meta=meta, labels=labels, brdu_model=brdu_model, edu_model=edu_model)
            if scored.empty:
                st.info("No labeled cells found in this barrage index.")
            else:
                if score_mode == "BrdU only":
                    scored["audit_score"] = scored["brdu_mismatch"]
                    scored = scored[np.isfinite(scored["audit_score"].to_numpy(dtype=np.float64))].copy()
                elif score_mode == "EdU only":
                    scored["audit_score"] = scored["edu_mismatch"]
                    scored = scored[np.isfinite(scored["audit_score"].to_numpy(dtype=np.float64))].copy()

                scored = scored.sort_values(["audit_score", "rank", "cell"], ascending=[False, True, True], kind="stable")
                cells = scored["cell"].astype(str).tolist()
                if not cells:
                    st.info("No scorable cells for this mode (model not fit?).")
                else:
                    slider_args = _audit_top_n_slider_args(len(cells))
                    if slider_args is None:
                        top_n = len(cells)
                        st.caption(f"Showing all {top_n} scorable cells.")
                    else:
                        min_top_n, max_top_n, top_default, top_step = slider_args
                        top_n = int(st.slider("Show top N", min_top_n, max_top_n, top_default, top_step))

                    st.dataframe(
                        scored[
                            [
                                "cell",
                                "rank",
                                "brdu",
                                "edu",
                                "p_brdu",
                                "p_edu",
                                "brdu_mismatch",
                                "edu_mismatch",
                                "audit_score",
                            ]
                        ].head(top_n),
                        hide_index=True,
                        use_container_width=True,
                    )

                    if st.session_state.get("audit_cell") not in set(cells):
                        st.session_state["audit_cell"] = cells[0]

                    next_cell = st.session_state.get("audit_next_cell")
                    if isinstance(next_cell, str) and next_cell in set(cells):
                        st.session_state["audit_cell"] = next_cell
                        del st.session_state["audit_next_cell"]

                    i = int(cells.index(str(st.session_state["audit_cell"])))
                    left, right = st.columns([1, 3], gap="large")
                    with left:
                        st.write(f"{i + 1}/{len(cells)}")
                        if st.button("Prev", disabled=i == 0, key="audit_prev"):
                            st.session_state["audit_cell"] = cells[max(i - 1, 0)]
                            st.rerun()
                        if st.button("Next", disabled=i >= (len(cells) - 1), key="audit_next"):
                            st.session_state["audit_cell"] = cells[min(i + 1, len(cells) - 1)]
                            st.rerun()

                        st.selectbox("Cell", options=cells, index=i, key="audit_cell")

                        cell = str(st.session_state["audit_cell"])
                        row_l = labels.set_index("cell").loc[cell]

                        row_s = scored.set_index("cell").loc[cell]
                        st.dataframe(
                            pd.DataFrame(
                                {
                                    "audit_score": [float(row_s["audit_score"])],
                                    "brdu_mismatch": [float(row_s["brdu_mismatch"]) if np.isfinite(row_s["brdu_mismatch"]) else np.nan],
                                    "edu_mismatch": [float(row_s["edu_mismatch"]) if np.isfinite(row_s["edu_mismatch"]) else np.nan],
                                    "p_brdu": [float(row_s["p_brdu"]) if np.isfinite(row_s["p_brdu"]) else np.nan],
                                    "p_edu": [float(row_s["p_edu"]) if np.isfinite(row_s["p_edu"]) else np.nan],
                                }
                            ),
                            hide_index=True,
                            use_container_width=True,
                        )

                        brdu_val = int(
                            st.radio(
                                "BrdU",
                                options=[0, 1],
                                horizontal=True,
                                index=int(row_l["brdu"]),
                                key=f"audit_brdu_{cell}",
                            )
                        )
                        edu_val = int(
                            st.radio(
                                "EdU",
                                options=[0, 1],
                                horizontal=True,
                                index=int(row_l["edu"]),
                                key=f"audit_edu_{cell}",
                            )
                        )

                        if st.button("Save edit", type="primary", key="audit_save"):
                            new_row = pd.DataFrame([{"cell": cell, "brdu": brdu_val, "edu": edu_val}])
                            labels2 = labels[labels["cell"] != cell].copy()
                            labels2 = pd.concat([labels2, new_row], ignore_index=True)
                            labels2 = labels2.sort_values("cell", kind="stable").reset_index(drop=True)
                            _atomic_write_csv(labels2, labels_path)
                            st.session_state["audit_next_cell"] = cells[min(i + 1, len(cells) - 1)]
                            st.rerun()

                    with right:
                        cell = str(st.session_state["audit_cell"])
                        if cell in meta_by_cell.index:
                            row_m = meta_by_cell.loc[cell]
                            img_path = (barrage_dir / str(row_m["image_path"])).resolve()
                            if img_path.exists():
                                st.image(str(img_path), use_container_width=True)
                            else:
                                st.error(f"Missing image: {img_path}")
                        else:
                            st.error("Cell not found in index.csv (barrage index).")


if __name__ == "__main__":
    main()
