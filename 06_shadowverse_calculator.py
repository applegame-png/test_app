# app.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import altair as alt
from datetime import date
from io import StringIO
import os
import plotly.express as px

# -------------------------
# 基本設定
# -------------------------
st.set_page_config(page_title="對戰記錄應用", page_icon="🎮", layout="wide")

DATA_FILE = "shadowverse_results.csv"
DEFAULT_CATEGORIES = ["復仇者", "主教", "龍族", "夜魔", "巫師", "皇家", "妖精"]
DEFAULT_ACCOUNTS = ["勝利", "敗北"]

COLS = ["日期","對戰類型","對手牌型","備註","結果","金額"]

# -------------------------
# 工具函式
# -------------------------
def yen(x):
    try:
        return f"¥{int(round(x)):,}"
    except Exception:
        return x

def load_data(path=DATA_FILE):
    if not os.path.exists(path):
        df = pd.DataFrame(columns=COLS)
        df.to_csv(path, index=False)
        return df
    df = pd.read_csv(path)
    # 型別整理
    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce").dt.date
    # 補足缺少的欄位
    for c in COLS:
        if c not in df.columns:
            df[c] = 0.0 if c == "金額" else ""
    # 金額為 float
    df["金額"] = pd.to_numeric(df["金額"], errors="coerce").fillna(0.0)
    return df[COLS]

def save_data(df, path=DATA_FILE):
    df.to_csv(path, index=False)

def ensure_session():
    if "df" not in st.session_state:
        st.session_state.df = load_data()
    if "categories" not in st.session_state:
        st.session_state.categories = DEFAULT_CATEGORIES.copy()
    if "accounts" not in st.session_state:
        st.session_state.accounts = DEFAULT_ACCOUNTS.copy()
    if "budgets" not in st.session_state:
        st.session_state.budgets = {}

ensure_session()

# -------------------------
# 側邊欄（改）
# -------------------------
with st.sidebar:
    st.subheader("📅 期間與搜尋")
    mindate = date(2010,1,1)
    maxdate = date.today()

    # 日期範圍
    col_from, col_to = st.columns(2)
    with col_from:
        d_from = st.date_input(
            "開始日期",
            value=st.session_state.get("filter_from", date(date.today().year, date.today().month, 1)),
            min_value=mindate,
            max_value=maxdate,
            key="filter_from",
        )
    with col_to:
        d_to = st.date_input(
            "結束日期",
            value=st.session_state.get("filter_to", maxdate),
            min_value=mindate,
            max_value=maxdate,
            key="filter_to",
        )

    # 關鍵字 + 清除
    kw = st.text_input("關鍵字（搜尋備註）", st.session_state.get("filter_kw",""), key="filter_kw")
    if st.button("清除關鍵字", key="btn_kw_clear"):
        st.session_state.filter_kw = ""

    st.divider()

    # -------------------------
    # 🏷️ 主檔維護
    # -------------------------
    st.subheader("🏷️ 主檔維護")
    with st.expander("對手牌型維護", expanded=False):
        new_cat = st.text_input("新增對手牌型", "", key="cat_add")
        if st.button("新增對手牌型", key="cat_add_btn"):
            if new_cat and new_cat not in st.session_state.categories:
                st.session_state.categories.append(new_cat)
                st.success(f"已新增對手牌型「{new_cat}」。")
        del_cat = st.selectbox("欲刪除的對手牌型", ["---請選擇---"] + st.session_state.categories, key="cat_del_sel")
        if st.button("刪除對手牌型", key="cat_del_btn"):
            if del_cat in st.session_state.categories:
                st.session_state.categories.remove(del_cat)
                st.warning(f"已刪除對手牌型「{del_cat}」。")

    with st.expander("結果維護", expanded=False):
        new_acc = st.text_input("新增結果（例：平手）", "", key="acc_add")
        if st.button("新增結果", key="acc_add_btn"):
            if new_acc and new_acc not in st.session_state.accounts:
                st.session_state.accounts.append(new_acc)
                st.success(f"已新增結果「{new_acc}」。")
        del_acc = st.selectbox("欲刪除的結果", ["---請選擇---"] + st.session_state.accounts, key="acc_del_sel")
        if st.button("刪除結果", key="acc_del_btn"):
            if del_acc in st.session_state.accounts:
                st.session_state.accounts.remove(del_acc)
                st.warning(f"已刪除結果「{del_acc}」。")

    st.divider()

    # -------------------------
    # 🗑️ 資料操作（連動目前篩選結果）
    # -------------------------
    st.subheader("🗑️ 資料操作")
    df_for_delete = st.session_state.df.copy()

    if len(df_for_delete) > 0:
        # 日期篩選
        df_for_delete = df_for_delete[
            (pd.to_datetime(df_for_delete["日期"]).dt.date >= st.session_state["filter_from"]) &
            (pd.to_datetime(df_for_delete["日期"]).dt.date <= st.session_state["filter_to"])
        ]
        # 關鍵字篩選
        if st.session_state["filter_kw"]:
            df_for_delete = df_for_delete[
                df_for_delete["備註"].fillna("").str.contains(st.session_state["filter_kw"], case=False, na=False)
            ]

    if len(df_for_delete) > 0:
        st.caption("可從期間/搜尋結果中選擇要刪除的資料（會同步反映到實體檔案）。")
        df_reset = df_for_delete.reset_index()  # 保留原本 index（st.session_state.df 的 index）
        df_reset["顯示"] = df_reset.apply(
            lambda r: f"{r['日期']} | {r['對戰類型']} | {r['對手牌型']} | {r['備註']} | {r['結果']} | {yen(r['金額'])}",
            axis=1
        )

        # 全選 / 全不選
        b1, b2 = st.columns(2)
        with b1:
            if st.button("全選", key="btn_select_all"):
                st.session_state.rows_to_delete = df_reset["顯示"].tolist()
        with b2:
            if st.button("全不選", key="btn_clear_sel"):
                st.session_state.rows_to_delete = []

        target = st.multiselect(
            "選擇要刪除的列",
            options=df_reset["顯示"].tolist(),
            default=st.session_state.get("rows_to_delete", []),
            key="rows_to_delete"
        )

        delete_disabled = len(target) == 0
        if st.button("刪除所選列", type="primary", disabled=delete_disabled, key="delete_rows_btn"):
            to_drop = df_reset[df_reset["顯示"].isin(target)]["index"].tolist()
            st.session_state.df = st.session_state.df.drop(index=to_drop).reset_index(drop=True)
            save_data(st.session_state.df)
            st.success(f"已刪除 {len(to_drop)} 筆資料。")


# -------------------------
# 主畫面：輸入表單
# -------------------------
st.title("對戰記錄應用 🎮")
with st.container():
    st.subheader("📥 新增記錄")
    with st.form("add_form", clear_on_submit=True):
        # 金額目前未使用，欄位結構維持不變；c4 為「結果」
        c1, c2, c3, c4 = st.columns([1,1,1,1.4])
        with c1:
            t_date = st.date_input("日期", value=date.today())
        with c2:
            t_type = st.selectbox("對戰類型", ["階級","2pick"])
        with c3:
            t_cat = st.selectbox("對手牌型", st.session_state.categories)
        with c4:
            t_acc = st.selectbox("結果", st.session_state.accounts)

        memo = st.text_input("備註", "")

        submitted = st.form_submit_button("新增", use_container_width=True)
        if submitted:
            new_row = pd.DataFrame([{
                "日期": t_date,
                "對戰類型": t_type,
                "對手牌型": t_cat,
                "備註": memo,
                "結果": t_acc
            }])
            st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
            save_data(st.session_state.df)
            st.success("已新增！")

st.divider()

df = st.session_state.df.copy()

# -------------------------
# 勝率摘要（依目前篩選結果）
# -------------------------
st.subheader("📈 勝率摘要")

def _calc_rate(_df):
    total = len(_df)
    wins = (_df["結果"] == "勝利").sum()
    losses = (_df["結果"] == "敗北").sum()
    rate = (wins / total * 100) if total > 0 else 0.0
    return total, wins, losses, rate

# 以日期昇冪排序，從尾端取 N 筆作為「近 N 場」
df_sorted = df.sort_values("日期", ascending=True)

def _recent_rate(n):
    sub = df_sorted.tail(n)
    return _calc_rate(sub)

# 近 5/10/15/20 場勝率
n_list = [5, 10, 15, 20]
cols = st.columns(len(n_list) + 1)

total_available = len(df_sorted)

for i, n in enumerate(n_list):
    if total_available < n:
        # 可用對戰數不足時顯示提示
        cols[i].metric(
            f"近{n}場 勝率",
            "次數不足",
            help=f"需要：{n} 場 / 目前：{total_available} 場"
        )
    else:
        total, wins, losses, rate = _recent_rate(n)
        cols[i].metric(
            f"近{n}場 勝率",
            f"{rate:.1f}%",
            help=f"母數：{total}（勝：{wins} / 敗：{losses}）"
        )

# 整體勝率
total_all, wins_all, losses_all, rate_all = _calc_rate(df_sorted)
cols[-1].metric("整體勝率", f"{rate_all:.1f}%", help=f"母數：{total_all}（勝：{wins_all} / 敗：{losses_all}）")

st.caption("※ 以上計算僅針對側邊欄期間/關鍵字篩選後的對戰。")

# -------------------------
# 依對手牌型的勝率（視覺化）
# -------------------------
st.subheader("🗂️ 對手牌型別 勝率")

if len(df_sorted) == 0:
    st.info("沒有符合的資料。請調整期間或搜尋條件。")
else:
    rows = []
    for name, sub in df_sorted.groupby("對手牌型"):
        total, wins, losses, rate = _calc_rate(sub)
        rows.append({
            "對手牌型": name,
            "對戰數": int(total),
            "勝利": int(wins),
            "敗北": int(losses),
            "勝率(%)": float(round(rate, 1)),
        })
    by_cat = pd.DataFrame(rows)

    if by_cat.empty:
        st.info("沒有符合的資料。")
    else:
        # 依對戰數、勝率由小到大（可依需求調整）
        by_cat = by_cat.sort_values(["對戰數", "勝率(%)"], ascending=[True, True]).reset_index(drop=True)
        y_sort = by_cat["對手牌型"].tolist()

        base = alt.Chart(by_cat)

        bars = base.mark_bar().encode(
            y=alt.Y("對手牌型:N", sort=y_sort, title="對手牌型"),
            x=alt.X("勝率(%)", type="quantitative", title="勝率 (%)",
                    scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("對手牌型:N", title="對手牌型").legend(None),  # 不顯示凡例
            tooltip=["對手牌型", "對戰數", "勝利", "敗北", "勝率(%)"]
        )

        labels = base.mark_text(align="left", baseline="middle", dx=3).encode(
            y=alt.Y("對手牌型:N", sort=y_sort),
            x=alt.X("勝率(%)", type="quantitative"),
            text=alt.Text("勝率(%)", format=".1f")
        )

        chart = (bars + labels).properties(
            height=max(500, 28 * len(by_cat)),
            width="container",
            title="對手牌型別 勝率"
        )

        st.altair_chart(chart, use_container_width=True)

# -------------------------
# 對手牌型比例（圓餅圖）
# -------------------------
st.subheader("🥧 對手牌型比例")

if 'by_cat' in locals() and not by_cat.empty:
    # 依對戰數多寡決定顯示順序（圖例與扇形順序）
    order = by_cat.sort_values("對戰數", ascending=False)["對手牌型"].tolist()

    fig = px.pie(
        by_cat,
        names="對手牌型",
        values="對戰數",
        category_orders={"對手牌型": order},
        hole=0.3  # 想改成甜甜圈就設 0.3
    )

    # 內部標籤：顯示 牌型 / 佔比 / 對戰數
    fig.update_traces(
        textposition="inside",
        texttemplate="%{label}<br>%{percent:.1%}",
        hovertemplate="對手牌型：%{label}<br>對戰數：%{value}<br>比例：%{percent:.1%}<extra></extra>",
        textfont_size=18

    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        legend_title_text="對手牌型",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("沒有可用資料來繪製圓餅圖。")



# -------------------------
# 篩選 & 顯示（主清單）
# -------------------------

# 期間篩選
df = df[(pd.to_datetime(df["日期"]).dt.date >= st.session_state["filter_from"]) &
        (pd.to_datetime(df["日期"]).dt.date <= st.session_state["filter_to"])]

# 關鍵字篩選
if st.session_state["filter_kw"]:
    df = df[df["備註"].fillna("").str.contains(st.session_state["filter_kw"], case=False, na=False)]

# 列表顯示
st.subheader("📋 記錄列表")
show_df = df.copy()
show_df = show_df[["日期","對戰類型","對手牌型","備註","結果"]].sort_values("日期", ascending=False)
st.dataframe(show_df, use_container_width=True, hide_index=True)

# -------------------------
# CSV 匯入/匯出（欄位固定：日期, 對戰類型, 對手牌型, 備註, 結果）
# -------------------------
st.subheader("📄 CSV 匯入/匯出")

# 固定欄位（已移除「金額」）
COLS = ["日期", "對戰類型", "對手牌型", "備註", "結果"]

c_up, c_down = st.columns(2)

# 工具：欄位/型別整理
def _normalize_df(_df: pd.DataFrame) -> pd.DataFrame:
    # 若缺必填欄位則補齊
    for c in COLS:
        if c not in _df.columns:
            _df[c] = ""
    # 型別整理
    _df["日期"] = pd.to_datetime(_df["日期"], errors="coerce").dt.date
    # 多餘欄位丟棄並固定順序
    return _df[COLS]

with c_up:
    st.markdown("**📥 匯入 CSV**")
    enc = st.selectbox("文字編碼", ["utf-8", "utf-8-sig（含 BOM）", "cp932（Excel 日文）"], index=0, key="csv_enc")
    mode = st.radio("匯入模式", ["追加（append）", "覆寫（replace）"], horizontal=True, key="csv_mode")
    dedup = st.checkbox("去除重複（鍵：日期, 對戰類型, 對手牌型, 備註, 結果）", value=True, key="csv_dedup")

    uploaded = st.file_uploader(
        "選擇 CSV（必填欄位：日期, 對戰類型, 對手牌型, 備註, 結果）",
        type=["csv"],
        key="csv_uploader"
    )

    if uploaded is not None:
        try:
            # pandas 讀取 file-like 無法直接指定 encoding；改以 bytes → str
            raw = uploaded.read()
            if st.session_state.csv_enc == "cp932（Excel 日文）":
                text = raw.decode("cp932", errors="ignore")
            elif st.session_state.csv_enc == "utf-8-sig（含 BOM）":
                text = raw.decode("utf-8-sig", errors="ignore")
            else:
                text = raw.decode("utf-8", errors="ignore")

            updf = pd.read_csv(StringIO(text))
            need = set(COLS)
            if not need.issubset(set(updf.columns)):
                st.error("欄位名稱不足。必須：日期, 對戰類型, 對手牌型, 備註, 結果")
            else:
                updf = _normalize_df(updf)

                # 自動補主檔：遇到新牌型/新結果就加入
                new_cats = sorted(set(updf["對手牌型"]) - set(st.session_state.categories))
                if new_cats:
                    st.session_state.categories.extend([c for c in new_cats if isinstance(c, str)])
                new_accs = sorted(set(updf["結果"]) - set(st.session_state.accounts))
                if new_accs:
                    st.session_state.accounts.extend([a for a in new_accs if isinstance(a, str)])

                if mode == "覆寫（replace）":
                    base_df = pd.DataFrame(columns=COLS)
                else:
                    base_df = st.session_state.df.copy()

                merged = pd.concat([base_df, updf[COLS]], ignore_index=True)

                # 去除重複
                if dedup:
                    before = len(merged)
                    merged = merged.drop_duplicates(
                        subset=["日期", "對戰類型", "對手牌型", "備註", "結果"],
                        keep="last"
                    ).reset_index(drop=True)
                    removed_dups = before - len(merged)
                else:
                    removed_dups = 0

                st.session_state.df = merged
                save_data(st.session_state.df)

                st.success(
                    f"已匯入 CSV：匯入 {len(updf)} 行 / "
                    f"目前總筆數 {len(st.session_state.df)} 行 / "
                    f"移除重複 {removed_dups} 行"
                )

                # 匯入預覽
                st.caption("匯入資料前 5 筆（整理後）")
                st.dataframe(updf.head(5), use_container_width=True, hide_index=True)

        except Exception as e:
            st.exception(e)

with c_down:
    st.markdown("**📤 匯出 CSV**")

    # 若此區塊無局部變數 df，可用下列方式重建目前篩選結果：
    # df = st.session_state.df.copy()
    # df = df[(pd.to_datetime(df["日期"]).dt.date >= st.session_state["filter_from"]) &
    #         (pd.to_datetime(df["日期"]).dt.date <= st.session_state["filter_to"])]
    # if st.session_state["filter_kw"]:
    #     df = df[df["備註"].fillna("").str.contains(st.session_state["filter_kw"], case=False, na=False)]

    # 輸出編碼
    out_enc = st.selectbox("輸出文字編碼", ["utf-8", "utf-8-sig（Excel 建議）"], index=1, key="csv_out_enc")

    col_a, col_b = st.columns(2)
    with col_a:
        # 全部資料
        dl_all = st.session_state.df.copy()
        dl_all["日期"] = pd.to_datetime(dl_all["日期"]).dt.strftime("%Y-%m-%d")
        all_bytes = dl_all[COLS].to_csv(index=False).encode("utf-8-sig" if out_enc.endswith("建議）") else "utf-8")
        st.download_button(
            label="匯出全部資料",
            data=all_bytes,
            file_name="shadowverse_export_all.csv",
            mime="text/csv",
            use_container_width=True,
            key="btn_export_all"
        )

    with col_b:
        # 目前篩選結果
        dl_f = df.copy() if 'df' in locals() else st.session_state.df.copy()
        if 'df' not in locals():
            dl_f = dl_f[
                (pd.to_datetime(dl_f["日期"]).dt.date >= st.session_state["filter_from"]) &
                (pd.to_datetime(dl_f["日期"]).dt.date <= st.session_state["filter_to"])
            ]
            if st.session_state["filter_kw"]:
                dl_f = dl_f[
                    dl_f["備註"].fillna("").str.contains(st.session_state["filter_kw"], case=False, na=False)
                ]

        dl_f["日期"] = pd.to_datetime(dl_f["日期"]).dt.strftime("%Y-%m-%d")
        filt_bytes = dl_f[COLS].to_csv(index=False).encode("utf-8-sig" if out_enc.endswith("建議）") else "utf-8")
        st.download_button(
            label="匯出目前篩選結果",
            data=filt_bytes,
            file_name="shadowverse_export_filtered.csv",
            mime="text/csv",
            use_container_width=True,
            key="btn_export_filtered"
        )

    # 備份與範本
    st.divider()
    bb1, bb2 = st.columns(2)
    with bb1:
        if st.button("備份目前全部資料（CSV）", key="btn_backup"):
            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            bak = st.session_state.df.copy()
            bak["日期"] = pd.to_datetime(bak["日期"]).dt.strftime("%Y-%m-%d")
            bytes_bak = bak[COLS].to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "下載備份",
                data=bytes_bak,
                file_name=f"shadowverse_backup_{ts}.csv",
                mime="text/csv",
                use_container_width=True,
                key="btn_backup_dl"
            )
            st.info("請從下方的下載按鈕儲存。")

    with bb2:
        # 空白範本
        tmpl = pd.DataFrame(columns=COLS)
        bytes_tmpl = tmpl.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下載空白範本（僅表頭）",
            data=bytes_tmpl,
            file_name="shadowverse_template.csv",
            mime="text/csv",
            use_container_width=True,
            key="btn_template"
        )
