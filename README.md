# Mobile Price Range Predictor

A machine learning app that predicts the price range of a mobile phone based on its hardware specifications.

## Model

- **Algorithm:** Random Forest Classifier (tuned with GridSearchCV / RandomizedSearchCV)
- **Dataset:** Mobile Price Classification (2000 samples, 20 features)
- **Target:** `price_range` — 0 (Low), 1 (Medium-Low), 2 (Medium-High), 3 (High)

## Features Used

| Feature                                                | Description                       |
| ------------------------------------------------------ | --------------------------------- |
| `battery_power`                                        | Battery capacity in mAh           |
| `ram`                                                  | RAM in MB                         |
| `clock_speed`                                          | Processor speed in GHz            |
| `px_height` / `px_width`                               | Screen resolution                 |
| `fc` / `pc`                                            | Front / primary camera megapixels |
| `int_memory`                                           | Internal storage in GB            |
| `blue`, `dual_sim`, `4g`, `3g`, `wifi`, `touch_screen` | Connectivity flags                |
| `pixel_area`                                           | Engineered: px_height × px_width  |
| `screen_area`                                          | Engineered: sc_h × sc_w           |

## How to Use

1. Adjust the sliders and checkboxes to match your phone's specs
2. Click **Predict Price Range**
3. View the predicted class and confidence scores

## Files

- `app.py` — Gradio interface
- `mobile_price_rf_model.pkl` — Trained Random Forest pipeline
- `requirements.txt` — Python dependencies
