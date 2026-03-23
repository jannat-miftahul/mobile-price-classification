import gradio as gr
import pandas as pd
import pickle
import numpy as np

# ── 1. Load the trained model ──────────────────────────────────────────────
with open("mobile_price_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

PRICE_LABELS = {
    0: "💰 Low Cost",
    1: "💰💰 Medium-Low Cost",
    2: "💰💰💰 Medium-High Cost",
    3: "💰💰💰💰 High Cost"
}

# ── 2. Prediction function ─────────────────────────────────────────────────
def predict_price(
    battery_power, blue, clock_speed, dual_sim,
    fc, four_g, int_memory, m_dep, mobile_wt,
    n_cores, pc, px_height, px_width, ram,
    sc_h, sc_w, talk_time, three_g, touch_screen, wifi
):
    # Derived features (must match training)
    pixel_area  = px_height * px_width
    screen_area = sc_h * sc_w

    input_df = pd.DataFrame([[
        battery_power, int(blue), clock_speed, int(dual_sim),
        fc, int(four_g), int_memory, m_dep, mobile_wt,
        n_cores, pc, px_height, px_width, ram,
        sc_h, sc_w, talk_time, int(three_g), int(touch_screen), int(wifi),
        pixel_area, screen_area
    ]], columns=[
        'battery_power','blue','clock_speed','dual_sim',
        'fc','four_g','int_memory','m_dep','mobile_wt',
        'n_cores','pc','px_height','px_width','ram',
        'sc_h','sc_w','talk_time','three_g','touch_screen','wifi',
        'pixel_area','screen_area'
    ])

    pred  = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    label       = PRICE_LABELS[pred]
    confidence  = proba[pred] * 100
    proba_text  = "\n".join(
        [f"  {PRICE_LABELS[i]}: {p*100:.1f}%" for i, p in enumerate(proba)]
    )

    return (
        f"### {label}\n\n"
        f"**Confidence:** {confidence:.1f}%\n\n"
        f"**All class probabilities:**\n{proba_text}"
    )

# ── 3. Gradio Interface ────────────────────────────────────────────────────
with gr.Blocks(title="Mobile Price Predictor") as app:

    gr.Markdown(
        """
        # Mobile Price Range Predictor
        Enter your mobile phone specifications below to predict its price range.
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🔋 Battery & Performance")
            battery_power = gr.Slider(500, 2000, step=1,  value=1000, label="Battery Power (mAh)")
            ram           = gr.Slider(256,  3998, step=2,  value=2048, label="RAM (MB)")
            clock_speed   = gr.Slider(0.5,   3.0, step=0.1,value=1.5, label="Clock Speed (GHz)")
            n_cores       = gr.Slider(1,      8,  step=1,  value=4,   label="Number of Cores")

        with gr.Column():
            gr.Markdown("### 📷 Camera")
            fc = gr.Slider(0, 19, step=1, value=5, label="Front Camera (MP)")
            pc = gr.Slider(0, 20, step=1, value=8, label="Primary Camera (MP)")

        with gr.Column():
            gr.Markdown("### 📺 Screen & Dimensions")
            px_height = gr.Slider(0,  1960, step=1, value=800,  label="Pixel Height")
            px_width  = gr.Slider(500, 1998, step=1, value=1200, label="Pixel Width")
            sc_h      = gr.Slider(5,   19,  step=1, value=12,   label="Screen Height (cm)")
            sc_w      = gr.Slider(0,   18,  step=1, value=6,    label="Screen Width (cm)")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 💾 Storage & Body")
            int_memory = gr.Slider(2,   64, step=1,   value=32,  label="Internal Memory (GB)")
            mobile_wt  = gr.Slider(80, 200, step=1,   value=140, label="Mobile Weight (g)")
            m_dep      = gr.Slider(0.1, 1.0, step=0.1, value=0.5, label="Mobile Depth (cm)")
            talk_time  = gr.Slider(2,   20,  step=1,   value=10,  label="Talk Time (hours)")

        with gr.Column():
            gr.Markdown("### 📡 Connectivity")
            blue        = gr.Checkbox(label="Bluetooth",    value=True)
            dual_sim    = gr.Checkbox(label="Dual SIM",     value=True)
            four_g      = gr.Checkbox(label="4G",           value=True)
            three_g     = gr.Checkbox(label="3G",           value=True)
            touch_screen= gr.Checkbox(label="Touch Screen", value=True)
            wifi        = gr.Checkbox(label="WiFi",         value=True)

    with gr.Row():
        predict_btn = gr.Button("🔍 Predict Price Range", variant="primary", scale=1)

    output = gr.Markdown(label="Prediction Result")

    predict_btn.click(
        fn=predict_price,
        inputs=[
            battery_power, blue, clock_speed, dual_sim,
            fc, four_g, int_memory, m_dep, mobile_wt,
            n_cores, pc, px_height, px_width, ram,
            sc_h, sc_w, talk_time, three_g, touch_screen, wifi
        ],
        outputs=output
    )

    gr.Examples(
        examples=[
            [500,  False, 0.5, False, 0,  False, 4,  0.4, 185, 2, 2, 20,  756,  256, 8, 2, 5,  False, False, False],
            [1500, True,  1.5, True,  5,  True,  32, 0.6, 140, 4, 8, 800, 1200, 2048,12, 6, 12, True,  True,  True],
            [2000, True,  3.0, True,  15, True,  64, 0.8, 120, 8, 20,1900,1980, 3998,18, 9, 20, True,  True,  True],
        ],
        inputs=[
            battery_power, blue, clock_speed, dual_sim,
            fc, four_g, int_memory, m_dep, mobile_wt,
            n_cores, pc, px_height, px_width, ram,
            sc_h, sc_w, talk_time, three_g, touch_screen, wifi
        ],
        label="Example Phones (Budget / Mid-range / Flagship)"
    )

if __name__ == "__main__":
    app.launch(share=True)