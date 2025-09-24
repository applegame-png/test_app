import streamlit as st
import pandas as pd
from datetime import date, timedelta
import os, json, math

st.set_page_config(page_title="營養管理減重紀錄", page_icon="🍱", layout="wide")

# ============================
# 檔案路徑 / 編碼
# ============================
LOG_PATH = "diet_log.csv"
FOOD_DB_PATH = "food_db.csv"
LIMITS_PATH = "limits.json"
WEIGHT_PATH = "weight_log.csv"
ADVICE_PATH = "advice_log.csv"
PROFILE_PATH = "profile.json"

CSV_ENCODING = "utf-8-sig"
ALT_ENCODING = "cp932"

# ============================
# 初始資料（台灣常用表述）
# ============================
DEFAULT_FOOD_DB = [
    {"food": "白飯", "unit": "", "per": 1.0, "kcal": 168, "protein": 2.5, "fat": 0.3, "carbs": 37.1, "fiber": 0.3, "sugar": 0.1, "sodium_mg": 1},
    {"food": "糙米", "unit": "", "per": 1.0, "kcal": 165, "protein": 2.8, "fat": 1.0, "carbs": 35.6, "fiber": 1.4, "sugar": 0.5, "sodium_mg": 5},
    {"food": "吐司", "unit": "", "per": 1.0, "kcal": 264, "protein": 9.3, "fat": 4.2, "carbs": 46.7, "fiber": 2.3, "sugar": 5.0, "sodium_mg": 490},
    {"food": "雞胸（去皮・熟）", "unit": "", "per": 1.0, "kcal": 120, "protein": 26.0, "fat": 1.5, "carbs": 0.0, "fiber": 0.0, "sugar": 0.0, "sodium_mg": 65},
    {"food": "雞蛋（全蛋）", "unit": "", "per": 1.0, "kcal": 76, "protein": 6.3, "fat": 5.3, "carbs": 0.2, "fiber": 0.0, "sugar": 0.2, "sodium_mg": 62},
]
NUTRIENTS = ["kcal", "protein", "fat", "carbs", "fiber", "sugar", "sodium_mg"]
MEAL_TYPES = ["早餐", "午餐", "晚餐", "點心"]

# ============================
# 輔助：聰明讀取 CSV
# ============================
@st.cache_data
def get_default_food_df():
    return pd.DataFrame(DEFAULT_FOOD_DB)

def read_csv_smart(file_or_path, is_path=True):
    enc_list = [CSV_ENCODING, ALT_ENCODING, "utf-8"]
    last_err = None
    for enc in enc_list:
        try:
            if is_path:
                return pd.read_csv(file_or_path, encoding=enc)
            else:
                file_or_path.seek(0)
                return pd.read_csv(file_or_path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise last_err

# ============================
# 載入 / 儲存函式
# ============================

def _ensure_food_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "unit" not in df.columns:
        df["unit"] = ""
    if "per" not in df.columns:
        df["per"] = 1.0
    cols = [c for c in df.columns if c in NUTRIENTS]
    if cols:
        df[cols] = df[cols].astype(float).round(1)
    df["per"] = pd.to_numeric(df["per"], errors="coerce").fillna(1.0).round(1)
    df["unit"] = df["unit"].astype(str)
    return df

def load_food_db(path: str = FOOD_DB_PATH) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = read_csv_smart(path, is_path=True)
            return _ensure_food_df_columns(df)
        except Exception:
            pass
    df = get_default_food_df().copy()
    return _ensure_food_df_columns(df)

def save_food_db(df: pd.DataFrame, path: str = FOOD_DB_PATH):
    df = _ensure_food_df_columns(df)
    df.to_csv(path, index=False, encoding=CSV_ENCODING)

def load_log(path: str = LOG_PATH) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = read_csv_smart(path, is_path=True)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            cols = [c for c in df.columns if c in NUTRIENTS]
            if cols:
                df[cols] = df[cols].astype(float).round(1)
            if "per" in df.columns:
                df["per"] = pd.to_numeric(df["per"], errors="coerce").round(1)
            if "unit" not in df.columns:
                df["unit"] = ""
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["date", "meal", "food", "unit", "per", *NUTRIENTS])

def save_log(df: pd.DataFrame, path: str = LOG_PATH):
    cols = [c for c in df.columns if c in NUTRIENTS]
    if cols:
        df[cols] = df[cols].astype(float).round(1)
    if "per" in df.columns:
        df["per"] = pd.to_numeric(df["per"], errors="coerce").fillna(1.0).round(1)
    if "unit" in df.columns:
        df["unit"] = df["unit"].astype(str)
    df.to_csv(path, index=False, encoding=CSV_ENCODING)

def load_limits(path: str = LIMITS_PATH) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "kcal": 2000.0,
        "protein": 150.0,
        "fat": 60.0,
        "carbs": 260.0,
        "sugar": 50.0,
        "sodium_mg": 2300.0,
        "fiber": 20.0,
        "enabled": False,
    }

def save_limits(limits: dict, path: str = LIMITS_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(limits, f, ensure_ascii=False, indent=2)

def load_weight(path: str = WEIGHT_PATH) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = read_csv_smart(path, is_path=True)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if "weight_kg" in df.columns:
                df["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce").round(1)
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["date", "weight_kg"])

def save_weight(df: pd.DataFrame, path: str = WEIGHT_PATH):
    if "weight_kg" in df.columns:
        df["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce").round(1)
    df.to_csv(path, index=False, encoding=CSV_ENCODING)

def load_advice_log(path: str = ADVICE_PATH) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = read_csv_smart(path, is_path=True)
            for col in ["start_day", "last_day", "created_at"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=[
        "created_at", "model", "window", "include_foods", "simple_mode",
        "start_day", "last_day", "ai_advice"
    ])

def save_advice_log(df: pd.DataFrame, path: str = ADVICE_PATH):
    df.to_csv(path, index=False, encoding=CSV_ENCODING)

def load_profile(path: str = PROFILE_PATH) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "sex": "男",
        "age": 28,
        "height_cm": 173.0,
        "current_weight_kg": 73.0,
        "activity": "中(每週運動1-3次)",
    }

def save_profile(prof: dict, path: str = PROFILE_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prof, f, ensure_ascii=False, indent=2)

# ============================
# Session 狀態初始化
# ============================
if "food_db" not in st.session_state:
    st.session_state.food_db = load_food_db()
if "log" not in st.session_state:
    st.session_state.log = load_log()
if "date" not in st.session_state:
    st.session_state.date = date.today()
if "limits" not in st.session_state:
    st.session_state.limits = load_limits()
if "weight" not in st.session_state:
    st.session_state.weight = load_weight()
if "advice" not in st.session_state:
    st.session_state.advice = load_advice_log()
if "profile" not in st.session_state:
    st.session_state.profile = load_profile()

# ============================
# 側邊欄（設定與資料）
# ============================
with st.sidebar:
    st.header("⚙️ 設定與資料")

    # 食品資料庫 讀取（下載按鈕在下方匯出區塊）
    uploaded_food = st.file_uploader("以 CSV 匯入食品資料庫（可選）", type=["csv"], accept_multiple_files=False)
    if uploaded_food is not None:
        try:
            df_up = read_csv_smart(uploaded_food, is_path=False)
            required = {"food", *NUTRIENTS}
            if not required.issubset(df_up.columns):
                st.error("CSV 需包含欄位：food, kcal, protein, fat, carbs, fiber, sugar, sodium_mg（unit, per 為可選）")
            else:
                df_up = _ensure_food_df_columns(df_up)
                st.session_state.food_db = df_up
                save_food_db(st.session_state.food_db)
                st.success("已讀取並儲存食品資料庫")
        except Exception as e:
            st.error(f"讀取錯誤：{e}")

    # 手動新增/刪除/刪除體重資料/個人檔案
    with st.expander("🧾 手動新增食品"):
        with st.form("add_food_form", clear_on_submit=True):
            food_name = st.text_input("食品名稱")
            c = st.columns(3)
            kcal = c[0].number_input("熱量 kcal", min_value=0.0, value=100.0)
            protein = c[1].number_input("蛋白質(g)", min_value=0.0, value=5.0)
            fat = c[2].number_input("脂肪(g)", min_value=0.0, value=3.0)
            c2b = st.columns(4)
            carbs = c2b[0].number_input("碳水化合物(g)", min_value=0.0, value=15.0)
            fiber = c2b[1].number_input("膳食纖維(g)", min_value=0.0, value=1.0)
            sugar = c2b[2].number_input("糖質(g)", min_value=0.0, value=10.0)
            sodium_mg = c2b[3].number_input("鈉(毫克)", min_value=0.0, value=100.0)
            submit_food = st.form_submit_button("新增食品")
        if submit_food:
            if food_name:
                new_row = {
                    "food": food_name, "unit": "", "per": 1.0,
                    "kcal": round(kcal, 1), "protein": round(protein, 1), "fat": round(fat, 1),
                    "carbs": round(carbs, 1), "fiber": round(fiber, 1), "sugar": round(sugar, 1),
                    "sodium_mg": round(sodium_mg, 1),
                }
                st.session_state.food_db = pd.concat([st.session_state.food_db, pd.DataFrame([new_row])], ignore_index=True)
                save_food_db(st.session_state.food_db)
                st.success(f"{food_name} 已加入資料庫（已儲存）")
            else:
                st.error("請輸入食品名稱")

    with st.expander("🗑️ 刪除食品"):
        foods = sorted(st.session_state.food_db["food"].astype(str).unique().tolist())
        del_select = st.multiselect("選擇要刪除的食品", foods)
        if st.button("刪除所選食品"):
            if del_select:
                before = len(st.session_state.food_db)
                st.session_state.food_db = st.session_state.food_db[~st.session_state.food_db["food"].isin(del_select)].reset_index(drop=True)
                save_food_db(st.session_state.food_db)
                after = len(st.session_state.food_db)
                st.success(f"已刪除 {len(del_select)} 筆（{before} → {after}）")
            else:
                st.info("尚未選擇刪除目標")

    with st.expander("⚖️ 刪除體重資料"):
        if st.session_state.weight.empty:
            st.caption("尚無體重資料")
        else:
            wtmp = st.session_state.weight.copy()
            wtmp["date"] = pd.to_datetime(wtmp["date"], errors="coerce")
            w_dates = sorted(wtmp["date"].dt.date.unique().tolist())
            del_w = st.multiselect("選擇要刪除的日期", w_dates, format_func=lambda d: d.strftime("%Y-%m-%d"))
            if st.button("刪除所選體重資料"):
                if del_w:
                    keep_mask = ~wtmp["date"].dt.date.isin(del_w)
                    st.session_state.weight = wtmp.loc[keep_mask].reset_index(drop=True)
                    save_weight(st.session_state.weight)
                    st.success(f"已刪除 {len(del_w)} 筆體重資料（已儲存）")
                else:
                    st.info("尚未選擇刪除目標")

    with st.expander("👤 個人檔案（AI 參考）", expanded=True):
        p = st.session_state.profile
        colp1, colp2 = st.columns(2)
        with colp1:
            p["sex"] = st.selectbox("性別", ["男","女","其他"], index=0 if p.get("sex","男")=="男" else 1)
            p["age"] = int(st.number_input("年齡", min_value=10, max_value=100, value=int(p.get("age",28))))
            p["height_cm"] = float(st.number_input("身高(cm)", min_value=120.0, max_value=230.0, value=float(p.get("height_cm",173.0)), step=0.1, format="%.1f"))
        with colp2:
            latest_w = None
            if not st.session_state.weight.empty:
                wtmp = st.session_state.weight.copy().sort_values("date")
                if not wtmp.empty and pd.notnull(wtmp["weight_kg"].iloc[-1]):
                    latest_w = float(wtmp["weight_kg"].iloc[-1])
            default_w = float(p.get("current_weight_kg", 73.0))
            if latest_w:
                default_w = latest_w
            p["current_weight_kg"] = float(st.number_input("目前體重(kg)", min_value=30.0, max_value=200.0, value=default_w, step=0.1, format="%.1f"))
            p["activity"] = st.selectbox("活動量", ["低(久坐)","中(每週運動1-3次)","高(每週運動4次以上)"], index=["低(久坐)","中(每週運動1-3次)","高(每週運動4次以上)"].index(p.get("activity","中(每週運動1-3次)")))
        save_profile(p)

    # ---- 📏 日目標 / 上限設定（上）
    st.markdown("---")
    st.subheader("📏 每日目標 / 上限設定")
    st.session_state.limits["enabled"] = st.toggle("啟用上限檢查", value=st.session_state.limits.get("enabled", False))
    cols = st.columns(3)
    st.session_state.limits["kcal"] = cols[0].number_input("kcal 上限", value=float(st.session_state.limits["kcal"]))
    st.session_state.limits["protein"] = cols[1].number_input("蛋白質(g) 上限", value=float(st.session_state.limits["protein"]))
    st.session_state.limits["fat"] = cols[2].number_input("脂肪(g) 上限", value=float(st.session_state.limits["fat"]))
    cols2 = st.columns(4)
    st.session_state.limits["carbs"] = cols2[0].number_input("碳水化合物(g) 上限", value=float(st.session_state.limits["carbs"]))
    st.session_state.limits["fiber"] = cols2[1].number_input("膳食纖維(g) 上限", value=float(st.session_state.limits["fiber"]))
    st.session_state.limits["sugar"] = cols2[2].number_input("糖質(g) 上限", value=float(st.session_state.limits["sugar"]))
    st.session_state.limits["sodium_mg"] = cols2[3].number_input("鈉(毫克) 上限", value=float(st.session_state.limits["sodium_mg"]))
    save_limits(st.session_state.limits)

    # ---- 🤖 AI 建議 設定（下）
    st.markdown("---")
    st.subheader("🤖 AI 建議 設定")
    st.caption("請取得 OpenAI 的 API key，並填入 secrets.toml 或下方欄位")
    env_key = os.environ.get("OPENAI_API_KEY", "")
    secret_key = None
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        pass
    if env_key and not secret_key:
        st.caption("🔐 偵測到環境變數 OPENAI_API_KEY。如需可在下方覆寫。")
    if secret_key:
        st.caption("🔐 偵測到 secrets.toml 的 OPENAI_API_KEY。如需可在下方覆寫。")
    api_key_input = st.text_input("OpenAI API Key", type="password", value="")
    st.session_state.ai_api_key = (api_key_input.strip() or secret_key or env_key or None)
    st.session_state.ai_model = st.selectbox("模型", ["gpt-4o-mini", "gpt-4.1-mini"], index=0)

    ai_window_options = ["全期間", 5, 10, 15, 20]
    sel = st.session_state.get('ai_window', 10)
    preselect_idx = 0 if sel == "全期間" else ai_window_options.index(sel) if sel in ai_window_options else ai_window_options.index(10)
    st.session_state.ai_window = st.radio("分析期間", ai_window_options, index=preselect_idx, horizontal=True, help="『全期間』代表從最早紀錄到最近。")
    st.session_state.ai_include_foods = st.checkbox("將餐飲明細一併提供給 AI（建議開啟）", value=st.session_state.get('ai_include_foods', True))
    st.session_state.ai_debug = st.checkbox("🛠 除錯：顯示送出的提示詞", value=st.session_state.get('ai_debug', False))

    # ---- 📦 匯出 / 自動儲存（含食品DB下載）
    st.markdown("---")
    st.subheader("📦 匯出 / 自動儲存")

    st.download_button(
        "下載目前食品資料庫 (CSV)",
        data=st.session_state.food_db.to_csv(index=False).encode(CSV_ENCODING),
        file_name="food_db.csv",
        mime="text/csv",
        use_container_width=True,
    )

    log_all2 = st.session_state.log.dropna(subset=["date"]).copy()
    if not log_all2.empty:
        daily_all = log_all2.groupby(log_all2["date"].dt.date)[NUTRIENTS].sum().round(1)
        meals_all = log_all2.groupby(log_all2["date"].dt.date).size().rename("meals")

        w2 = st.session_state.weight.copy()
        w2["date"] = pd.to_datetime(w2["date"], errors="coerce")
        weight_all = w2.set_index(w2["date"].dt.date)[["weight_kg"]] if not w2.empty else pd.DataFrame(columns=["weight_kg"])

        combined = daily_all.join(meals_all, how="outer").join(weight_all, how="outer").sort_index()

        if st.session_state.limits.get("enabled", False):
            for n in NUTRIENTS:
                L = float(st.session_state.limits.get(n, 0) or 0)
                if L > 0 and n in combined.columns:
                    combined[n + "_remaining"] = (L - combined[n]).apply(lambda x: round(x, 1) if pd.notnull(x) and x > 0 else 0.0)

        adv2 = st.session_state.advice.copy()
        if not adv2.empty:
            adv2["date"] = pd.to_datetime(adv2["last_day"], errors="coerce").dt.date
            latest_adv = (
                adv2.sort_values("created_at")
                    .groupby("date", as_index=False)
                    .tail(1)[["date", "ai_advice"]]
                    .set_index("date")
            )
            combined = combined.join(latest_adv, how="left")

        combined = combined.round(1)
        combined.index.name = "date"
        csv_combined = combined.reset_index().to_csv(index=False).encode(CSV_ENCODING)
        st.download_button(
            "下載合併資料（全期間）CSV",
            data=csv_combined,
            file_name="combined_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.caption("目前尚無可合併的資料")

    # 個別匯出
    csv_all = st.session_state.log.round(1).to_csv(index=False).encode(CSV_ENCODING)
    st.download_button("飲食紀錄 CSV", data=csv_all, file_name=LOG_PATH, mime="text/csv", use_container_width=True)

    csv_w = st.session_state.weight.round(1).to_csv(index=False).encode(CSV_ENCODING) if not st.session_state.weight.empty else ("date,weight_kg\n".encode(CSV_ENCODING))
    st.download_button("體重紀錄 CSV", data=csv_w, file_name=WEIGHT_PATH, mime="text/csv", use_container_width=True)

    csv_adv = st.session_state.advice.to_csv(index=False).encode(CSV_ENCODING)
    st.download_button("AI 建議歷史 CSV", data=csv_adv, file_name=ADVICE_PATH, mime="text/csv", use_container_width=True)

    # 自動儲存（側欄最後）
    save_log(st.session_state.log)
    save_weight(st.session_state.weight)
    save_advice_log(st.session_state.advice)
    save_profile(st.session_state.profile)

# ============================
# 主畫面 UI
# ============================
st.title("🍱 營養管理減重紀錄")
st.caption("此為示範 App，建議自 GitHub 下載程式碼於本機執行")

# ------- 1×2：輸入表單 & 當日體重 -------
st.markdown("### 輸入 / 體重（當日）")

selected_date = st.date_input("顯示日期（共用於輸入與體重）", value=st.session_state.date, format="YYYY-MM-DD", key="display_date_main")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### 🍽️ 輸入表單")
    with st.form("input_form"):
        meal = st.selectbox("餐別", MEAL_TYPES, index=0)
        db = st.session_state.food_db
        options = db["food"].tolist()
        food = st.selectbox("選擇食品", options, index=0 if options else None)
        submitted = st.form_submit_button("➕ 新增 1 份", use_container_width=True)

    if submitted and food:
        row = st.session_state.food_db[st.session_state.food_db["food"] == food].iloc[0]
        entry = {
            "date": pd.to_datetime(selected_date),
            "meal": meal,
            "food": row.get("food", food),
            "unit": str(row.get("unit", "")),
            "per": round(float(row.get("per", 1.0)), 1),
        }
        for n in NUTRIENTS:
            entry[n] = round(float(row[n]), 1)
        st.session_state.log = pd.concat([st.session_state.log, pd.DataFrame([entry])], ignore_index=True)
        save_log(st.session_state.log)
        st.success(f"已新增 1 份 {food}（已儲存）")

with col_right:
    st.markdown("#### ⚖️ 體重紀錄（當日）")
    wdf = st.session_state.weight.copy()
    wdf["date"] = pd.to_datetime(wdf["date"], errors="coerce")
    cur = wdf[wdf["date"].dt.date == pd.to_datetime(selected_date).date()]
    def_weight = float(cur["weight_kg"].iloc[0]) if not cur.empty else float(st.session_state.profile.get("current_weight_kg", 73.0))
    input_weight = st.number_input("體重(kg)", min_value=0.0, value=round(def_weight,1), step=0.1, format="%.1f")
    if st.button("儲存體重", use_container_width=True):
        st.session_state.weight = wdf[wdf["date"].dt.date != pd.to_datetime(selected_date).date()].copy()
        new_row = pd.DataFrame({"date": [pd.to_datetime(selected_date)], "weight_kg": [round(input_weight,1)]})
        st.session_state.weight = pd.concat([st.session_state.weight, new_row], ignore_index=True)
        save_weight(st.session_state.weight)
        st.session_state.profile["current_weight_kg"] = round(input_weight,1)
        save_profile(st.session_state.profile)
        st.success("已儲存體重")

# ============================
# 當日清單與合計
# ============================
st.markdown("---")
st.subheader(f"📒 {selected_date} 的紀錄")

st.session_state.log["date"] = pd.to_datetime(st.session_state.log["date"], errors="coerce")
mask = st.session_state.log["date"].dt.date == pd.to_datetime(selected_date).date()
day_df = st.session_state.log.loc[mask].copy()

if day_df.empty:
    st.info("此日尚無紀錄，請由表單新增。")
else:
    day_df = day_df.reset_index(drop=False).rename(columns={"index": "_idx"})
    day_df["刪除"] = False
    display_cols = ["_idx", "meal", "food", *NUTRIENTS, "刪除"]
    show_df = day_df.copy()
    for c in NUTRIENTS:
        if c in show_df.columns:
            show_df[c] = pd.to_numeric(show_df[c], errors="coerce").round(1)

    edited = st.data_editor(
        show_df[display_cols],
        num_rows="dynamic",
        use_container_width=True,
        key=f"editor_{selected_date}",
        hide_index=True,
    )
    to_delete = edited[edited["刪除"] == True]["_idx"].tolist()
    if to_delete:
        st.session_state.log = st.session_state.log.drop(index=to_delete).reset_index(drop=True)
        save_log(st.session_state.log)
        st.warning(f"已刪除 {len(to_delete)} 筆（已儲存）")

    totals = edited[NUTRIENTS].sum().round(1)

    colA, colB = st.columns([1,1])
    with colA:
        st.markdown("### 🔢 當日營養合計")
        st.table(totals.to_frame(name="合計"))
    with colB:
        st.markdown("### ⏳ 距離上限的剩餘（不足量）")
        if st.session_state.limits.get("enabled", False):
            rem = {}
            over_list = []
            for n in NUTRIENTS:
                limit = float(st.session_state.limits.get(n, 0) or 0)
                val = float(totals.get(n, 0) or 0)
                if limit > 0:
                    diff = round(limit - val, 1)
                    rem[n] = diff if diff > 0 else 0.0
                    if diff < 0:
                        over_list.append((n, round(-diff,1)))
            st.table(pd.Series(rem).to_frame("剩餘").round(1))
            if over_list:
                msg = "\n".join([f"- {k}: 超過上限 {v:.1f}" for k, v in over_list])
                st.error("⚠️ 已超過上限\n" + msg)
        else:
            st.info("可於側邊欄『每日目標 / 上限設定』啟用上限檢查")

    csv_day = edited.drop(columns=["_idx", "刪除"]).round(1).to_csv(index=False).encode(CSV_ENCODING)
    st.download_button("下載此日紀錄 CSV", data=csv_day, file_name=f"diet_{selected_date}.csv", mime="text/csv", use_container_width=True)

# ============================
# 近期彙總與視覺化（皆為日單位）
# ============================
st.markdown("---")
st.subheader("📈 近期彙總與視覺化（日單位）")

log2 = st.session_state.log.dropna(subset=["date"]).copy()

def style_exceed(df: pd.DataFrame, limits: dict):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    if not limits.get("enabled", False):
        return styles
    for col in df.columns:
        if col in NUTRIENTS:
            lim = float(limits.get(col, 0) or 0)
            if lim > 0:
                mask = df[col] > lim
                styles.loc[mask, col] = "color: red; font-weight: 700;"
    return styles

def daily_meal_presence(rdf: pd.DataFrame) -> pd.DataFrame:
    # 以日期×餐別，計算是否有紀錄（0/1）
    if rdf.empty:
        return pd.DataFrame(columns=MEAL_TYPES)
    rdf["date_only"] = rdf["date"].dt.date
    pres = (
        rdf.groupby(["date_only", "meal"]).size().unstack(fill_value=0).reindex(columns=MEAL_TYPES, fill_value=0)
    )
    pres = (pres > 0).astype(int)
    pres.index.name = None
    pres = pres.rename_axis(None, axis=1)
    return pres

if log2.empty:
    st.info("目前沒有資料")
else:
    def render_window(window):
        # window: "all" 或 int（天數）
        if window == "all":
            last_day = log2["date"].dt.date.max()
            start_day = log2["date"].dt.date.min()
        else:
            last_day = max(log2["date"].dt.date.max(), date.today())
            start_day = last_day - timedelta(days=int(window)-1)

        rmask = (log2["date"].dt.date >= start_day) & (log2["date"].dt.date <= last_day)
        rdf = log2.loc[rmask].copy()

        daily = rdf.groupby(rdf["date"].dt.date)[NUTRIENTS].sum().round(1).sort_index()

        presence = daily_meal_presence(rdf)
        daily = daily.join(presence, how="left").fillna(0)
        for m in MEAL_TYPES:
            if m in daily.columns:
                daily[m] = daily[m].astype(int)

        table_df = daily.reset_index().rename(columns={"index": "日期"})
        table_df = table_df.rename(columns={"date": "日期"})
        if "日期" not in table_df.columns:
            table_df = table_df.rename(columns={table_df.columns[0]: "日期"})

        styled = table_df.style.apply(style_exceed, limits=st.session_state.limits, axis=None)
        st.caption(f"期間：{start_day} 〜 {last_day}。超過當日上限者以紅字顯示")
        st.dataframe(styled, use_container_width=True)

        # ---- 1×2：kcal 與 體重 走勢
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### kcal 走勢（每日）")
            if "kcal" in daily.columns and not daily.empty:
                st.line_chart(daily[["kcal"]])
            else:
                st.caption("無 kcal 資料")

        with c2:
            st.markdown("#### 體重走勢（每日）")
            w = st.session_state.weight.copy()
            if not w.empty:
                w["date"] = pd.to_datetime(w["date"], errors="coerce")
                wv = w[(w["date"].dt.date >= start_day) & (w["date"].dt.date <= last_day)].copy().sort_values("date")
                if not wv.empty and "weight_kg" in wv.columns:
                    wt = wv.set_index(wv["date"].dt.date)[["weight_kg"]]
                    st.line_chart(wt)
                else:
                    st.caption("此期間無體重資料")
            else:
                st.caption("無體重資料")

        st.markdown("#### 蛋白質 / 脂肪 / 碳水 走勢（每日）")
        pfc_cols = [col for col in ["protein","fat","carbs"] if col in daily.columns]
        if pfc_cols:
            st.line_chart(daily[pfc_cols])

        return rdf, start_day, last_day, daily

    windows = [5, 10, 15, 20, 30, 60, 90, "全期間"]
    tabs = st.tabs([str(w) + ("日" if isinstance(w, int) else "") for w in windows])
    for t, window in zip(tabs, windows):
        with t:
            win_key = "all" if window == "全期間" else int(window)
            render_window(win_key)

# ============================
# 🤖 AI 減重建議（OpenAI API）
# ============================
st.markdown("---")
st.subheader("🤖 AI 減重建議（OpenAI API）")

ai_key = st.session_state.get('ai_api_key')
ai_model = st.session_state.get('ai_model', 'gpt-4o-mini')
ai_window_sel = st.session_state.get('ai_window', 10)  # "全期間" 或 int
ai_include_foods = bool(st.session_state.get('ai_include_foods', True))
ai_debug = bool(st.session_state.get('ai_debug', False))
profile = st.session_state.profile

col_ai1, col_ai2 = st.columns([1,1])
with col_ai1:
    run_ai = st.button("產生 AI 摘要與建議")
with col_ai2:
    simple_mode = st.checkbox("精簡摘要（重點式）", value=True)

def calc_bmi(height_cm: float, weight_kg: float):
    h_m = max(0.5, float(height_cm)/100.0)
    if weight_kg is None or math.isnan(weight_kg): return None
    return round(weight_kg / (h_m*h_m), 1)

def std_weight(height_cm: float):
    h_m = max(0.5, float(height_cm)/100.0)
    return round(22.0 * h_m * h_m, 1)

if run_ai:
    if not ai_key:
        st.error("請輸入 OpenAI API Key（見側邊欄）")
    else:
        base = st.session_state.log.dropna(subset=["date"]).copy()
        if base.empty:
            st.info("尚無飲食資料")
        else:
            if ai_window_sel == "全期間":
                start_day = base["date"].dt.date.min()
                last_day = base["date"].dt.date.max()
                window_days = (last_day - start_day).days + 1
            else:
                window_days = int(ai_window_sel)
                last_day = max(base["date"].dt.date.max(), date.today())
                start_day = last_day - timedelta(days=window_days-1)

            rmask = (base["date"].dt.date >= start_day) & (base["date"].dt.date <= last_day)
            rdf = base.loc[rmask].copy()
            daily = rdf.groupby(rdf["date"].dt.date)[NUTRIENTS].sum().round(1).sort_index()

            w = st.session_state.weight.copy()
            if not w.empty:
                w["date"] = pd.to_datetime(w["date"], errors="coerce")
                wmask = (w["date"].dt.date >= start_day) & (w["date"].dt.date <= last_day)
                wv = w.loc[wmask].copy()
                weight_series = wv.set_index(wv["date"].dt.date)["weight_kg"] if not wv.empty else pd.Series(dtype=float)
            else:
                weight_series = pd.Series(dtype=float)

            today_mask = st.session_state.log["date"].dt.date == last_day
            today_tot = st.session_state.log.loc[today_mask, NUTRIENTS].sum().round(1)
            limits = st.session_state.limits
            remaining = {}
            if limits.get("enabled", False):
                for n in NUTRIENTS:
                    L = float(limits.get(n, 0) or 0)
                    v = float(today_tot.get(n, 0) or 0)
                    if L>0:
                        diff = round(L - v, 1)
                        remaining[n] = diff if diff>0 else 0.0

            food_detail_json = None
            if ai_include_foods and not rdf.empty:
                freq = rdf["food"].value_counts().head(30).reset_index()
                freq.columns = ["food", "count"]
                food_sum = rdf.groupby("food")[NUTRIENTS].sum().round(1)
                def top_by(col, n=12):
                    if col not in food_sum.columns:
                        return []
                    return (
                        food_sum[col]
                        .sort_values(ascending=False)
                        .head(n)
                        .reset_index()
                        .rename(columns={col: f"total_{col}"})
                        .to_dict(orient="records")
                    )
                top_dict = {col: top_by(col) for col in NUTRIENTS}
                recent = rdf.sort_values("date").tail(80)[["date", "meal", "food"]].copy()
                recent["date"] = pd.to_datetime(recent["date"]).dt.strftime("%Y-%m-%d")
                recent_records = recent.to_dict(orient="records")
                food_detail = {
                    "食品頻率TOP": freq.to_dict(orient="records"),
                    "依營養素的前幾名食品": top_dict,
                    "最近的飲食明細": recent_records,
                }
                food_detail_json = json.dumps(food_detail, ensure_ascii=False)

            p = profile
            p_sex = p.get("sex","男")
            p_age = int(p.get("age", 28))
            p_h = float(p.get("height_cm", 173.0))
            p_w = float(p.get("current_weight_kg", 73.0))
            p_act = p.get("activity","中(每週運動1-3次)")
            p_bmi = calc_bmi(p_h, p_w)
            p_std = std_weight(p_h)

            df_for_prompt = daily.reset_index().rename(columns={"date":"日期"})
            df_for_prompt["日期"] = df_for_prompt["日期"].astype(str)
            weight_dict = {str(k): float(v) for k, v in weight_series.to_dict().items()}

            system_msg = (
                "你是一位具備臨床營養觀點、使用繁體中文（台灣用語）的減重教練。"
                "提供安全、可執行的建議，避免極端減重與醫療判斷。"
                "請根據飲食紀錄、體重走勢、上限設定與個人檔案，提出具體可行的飲食建議。"
            )
            style = "以 3〜5 點條列重點" if simple_mode else "先摘要趨勢，後給建議（含小標題）"

            prof_block = {
                "性別": p_sex, "年齡": p_age, "身高_cm": p_h, "目前體重_kg": p_w,
                "BMI": p_bmi, "標準體重_kg(BMI22)": p_std, "活動量": p_act
            }

            base_block = f"""
【個人檔案】{json.dumps(prof_block, ensure_ascii=False)}
【分析期間】{start_day}〜{last_day}（{window_days}日）
【每日合計（kcal/蛋白質/脂肪/碳水/膳食纖維/糖質/鈉）】
{df_for_prompt.to_json(orient='records', force_ascii=False)}
【體重(kg) 走勢】{json.dumps(weight_dict, ensure_ascii=False)}
【上限設定】{json.dumps({k: float(limits.get(k, 0) or 0) for k in NUTRIENTS}, ensure_ascii=False)}
【今日剩餘量（距離上限，沒有則為0）】{json.dumps(remaining, ensure_ascii=False)}
"""
            if food_detail_json:
                base_block += (
                    f"\n【基於食品名稱的參考資訊（頻率/前幾名/近期明細）】{food_detail_json}\n"
                    "請直接引用上方食品名稱，並提出替代選項、外食或超商選擇的具體方法。\n"
                )

            user_msg = (
                base_block +
                f"請先{style}說明『期間趨勢』，"
                "接著以繁中（台灣用語）提出『改善行動』。"
                "務必包含『體重相關建議』（合理的每週減重幅度、建議能量攝取範圍、每日蛋白質建議(g)、減重注意事項）。"
                "最後附上一行注意事項。"
            )

            if ai_debug:
                with st.expander("🛠 除錯：送出的提示詞（system / user）", expanded=False):
                    st.code(system_msg, language="markdown")
                    st.code(user_msg, language="markdown")

            try:
                from openai import OpenAI
                client = OpenAI(api_key=ai_key)
                resp = client.chat.completions.create(
                    model=ai_model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.6,
                )
                advice = resp.choices[0].message.content
                st.success("已產生 AI 建議")
                st.markdown(advice)
                st.caption("※ 本內容僅供一般資訊，若有慢性病或用藥請諮詢醫療專業人員。")

                new_adv = pd.DataFrame([{
                    "created_at": pd.Timestamp.now(tz="Asia/Taipei"),
                    "model": ai_model,
                    "window": "all" if ai_window_sel == "全期間" else int(ai_window_sel),
                    "include_foods": bool(ai_include_foods),
                    "simple_mode": bool(simple_mode),
                    "start_day": pd.to_datetime(start_day),
                    "last_day": pd.to_datetime(last_day),
                    "ai_advice": advice,
                }])
                st.session_state.advice = pd.concat([st.session_state.advice, new_adv], ignore_index=True)
                save_advice_log(st.session_state.advice)
                st.success("已儲存 AI 建議（advice_log.csv）")

            except ModuleNotFoundError:
                st.error("找不到 `openai` 套件。請先 `pip install openai`，並在 requirements.txt 加入 `openai>=1.42.0`。")
            except Exception as e:
                st.error(f"呼叫 OpenAI API 時發生錯誤：{e}")

# ============================
# 📝 最近一次 AI 建議
# ============================
st.markdown("---")
st.subheader("📝 最近一次 AI 建議")

adv_hist = st.session_state.advice.copy()
if adv_hist.empty:
    st.caption("尚無 AI 建議紀錄。請於上方產生。")
else:
    adv_hist["created_at"] = pd.to_datetime(adv_hist["created_at"], errors="coerce")
    latest = adv_hist.sort_values("created_at").iloc[-1]
    created_s = pd.to_datetime(latest["created_at"]).strftime("%Y-%m-%d %H:%M") if pd.notnull(latest.get("created_at")) else ""
    model_s = str(latest.get("model", ""))
    window_s = latest.get("window", 0)
    if str(window_s) == "all":
        window_disp = "全期間"
    else:
        window_disp = f"{int(window_s)}日"
    period_s = ""
    if pd.notnull(latest.get("start_day")) and pd.notnull(latest.get("last_day")):
        sd = pd.to_datetime(latest["start_day"], errors="coerce")
        ld = pd.to_datetime(latest["last_day"], errors="coerce")
        if pd.notnull(sd) and pd.notnull(ld):
            period_s = f"{sd.date()} 〜 {ld.date()}"
    st.caption(f"產生時間：{created_s} / 模型：{model_s} / 期間：{window_disp}" + (f"（{period_s}）" if period_s else ""))
    st.info(str(latest.get("ai_advice", "")))
