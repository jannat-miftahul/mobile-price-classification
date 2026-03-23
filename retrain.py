import warnings
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
import pickle

warnings.filterwarnings("ignore")

# ── 1. Load dataset ───────────────────────────────────────────────────────────
print("Loading train.csv …")
df = pd.read_csv("train.csv")
print(f"   {len(df)} rows, columns: {list(df.columns)}")

# ── 2. Feature engineering (must match app.py) ────────────────────────────────
df["pixel_area"]  = df["px_height"] * df["px_width"]
df["screen_area"] = df["sc_h"]      * df["sc_w"]

FEATURE_COLS = [
    "battery_power", "blue", "clock_speed", "dual_sim",
    "fc", "four_g", "int_memory", "m_dep", "mobile_wt",
    "n_cores", "pc", "px_height", "px_width", "ram",
    "sc_h", "sc_w", "talk_time", "three_g", "touch_screen", "wifi",
    "pixel_area", "screen_area",
]
TARGET_COL = "price_range"

X = df[FEATURE_COLS].astype(float)
y = df[TARGET_COL].astype(int)

# ── 3. Build & fit pipeline ───────────────────────────────────────────────────
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("clf",     RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )),
])

print("Training RandomForest pipeline …")
pipe.fit(X, y)

scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
print(f"   5-fold CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# ── 4. Save ───────────────────────────────────────────────────────────────────
PKL_PATH = "mobile_price_rf_model.pkl"
with open(PKL_PATH, "wb") as f:
    pickle.dump(pipe, f)

print(f"\n✅ Saved updated model → {PKL_PATH}")
print("   Now run:  python app.py")
