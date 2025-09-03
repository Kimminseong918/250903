# app.py
# 실행: streamlit run app.py

import io
import gc
from datetime import timedelta
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

# Altair 행수 제한 해제(큰 데이터 경고 완화)
alt.data_transformers.disable_max_rows()

# =========================
# 0) 기본 설정
# =========================
SESSION_GAP_MIN = 30  # (데이터에 session_id가 없을 때만 사용)
BIG_FILE_MB = 120     # 이 크기 이상이면 Lite 모드 기본 ON
DEFAULT_SAMPLE_FRAC = 0.35

st.set_page_config(page_title="RARRA Dashboard", layout="wide")
st.title("📊 RARRA Dashboard (Retention • Activation • Referral • Revenue • Acquisition)")
st.caption("Kaggle GA Customer Revenue Dataset | 세션 단위 분석 (메모리 최적화)")

# =========================
# 1) 데이터 로딩 & 세션 테이블 구성
# =========================
WANTED_COLS = [
    "event_time","fullVisitorId","session_id",
    "totals_pageviews","totals_hits","totals_bounces","totals_newVisits",
    "totals_transactionRevenue","visitNumber","session_duration","is_transaction",
    "channelGrouping","device_deviceCategory",
    "trafficSource_source","trafficSource_medium","trafficSource_referralPath",
    "geoNetwork_continent","device_operatingSystem","date",
]

def _read_csv_safely(file_or_bytes, usecols=None):
    """업로더/경로 모두 지원."""
    if isinstance(file_or_bytes, (str, bytes, io.BytesIO)):
        return pd.read_csv(file_or_bytes, encoding="utf-8-sig", dtype=str,
                           low_memory=False, usecols=usecols)
    # Streamlit UploadedFile
    raw = file_or_bytes.getvalue()
    return pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", dtype=str,
                       low_memory=False, usecols=usecols)

@st.cache_data(show_spinner=True)
def load_df_memory_smart(file_or_bytes, use_lite: bool, sample_frac: float) -> pd.DataFrame:
    # 헤더만 읽어 필수 확인 + usecols 확정
    if hasattr(file_or_bytes, "getvalue"):
        head = pd.read_csv(io.BytesIO(file_or_bytes.getvalue()), nrows=0, dtype=str)
    else:
        head = pd.read_csv(file_or_bytes, nrows=0, dtype=str)
    required = ["event_time", "fullVisitorId"]
    if not all(c in head.columns for c in required):
        missing = [c for c in required if c not in head.columns]
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    usecols = [c for c in head.columns if c in WANTED_COLS]
    df = _read_csv_safely(file_or_bytes, usecols=usecols)

    # Lite 샘플링
    if use_lite and 0 < sample_frac < 1.0:
        rng = np.random.RandomState(42)
        df = df.loc[rng.rand(len(df)) < sample_frac].copy()

    # 시간 처리
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df["event_time_naive"] = df["event_time"].dt.tz_convert("UTC").dt.tz_localize(None)

    # 숫자형 변환 + downcast
    def to_num(cols):
        for c in cols:
            if c in df.columns:
                s = pd.to_numeric(df[c], errors="coerce").fillna(0)
                if pd.api.types.is_float_dtype(s):
                    df[c] = pd.to_numeric(s, downcast="float")
                else:
                    df[c] = pd.to_numeric(s, downcast="integer")
    to_num(["totals_pageviews","totals_hits","totals_bounces","totals_newVisits",
            "totals_transactionRevenue","visitNumber","session_duration","is_transaction"])

    # 문자열 기본값
    df["fullVisitorId"] = df["fullVisitorId"].astype(str)
    for c in ["channelGrouping","device_deviceCategory","trafficSource_source",
              "trafficSource_medium","trafficSource_referralPath",
              "geoNetwork_continent","device_operatingSystem","date"]:
        if c not in df.columns:
            df[c] = "Unknown"
        else:
            df[c] = df[c].fillna("Unknown")

    # session_id 생성(없다면 30분 룰)
    if "session_id" not in df.columns:
        t = df[["fullVisitorId","event_time_naive"]].sort_values(["fullVisitorId","event_time_naive"]).reset_index(drop=True)
        diff = t.groupby("fullVisitorId")["event_time_naive"].diff().dt.total_seconds()
        new_sess = (diff.isna()) | (diff > SESSION_GAP_MIN*60)
        sess_num = new_sess.groupby(t["fullVisitorId"]).cumsum().astype("int32")
        df = df.loc[t.index].copy()
        df["session_id"] = df["fullVisitorId"] + "_" + sess_num.astype(str)

    # 자주 쓰는 문자열 → category (메모리 절감)
    for c in ["channelGrouping","device_deviceCategory","trafficSource_source",
              "trafficSource_medium","trafficSource_referralPath",
              "geoNetwork_continent","device_operatingSystem",
              "session_id","fullVisitorId"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    return df

@st.cache_data(show_spinner=False)
def build_session_table(df: pd.DataFrame) -> pd.DataFrame:
    sess = (df.groupby("session_id", as_index=False)
              .agg(
                  fullVisitorId=("fullVisitorId","first"),
                  session_start=("event_time_naive","min"),
                  session_end=("event_time_naive","max"),
                  pv=("totals_pageviews","sum"),
                  hits=("totals_hits","sum"),
                  bounces=("totals_bounces","max"),
                  newVisit=("totals_newVisits","max"),
                  visitNumber=("visitNumber","max"),
                  revenue=("totals_transactionRevenue","sum"),
                  channel=("channelGrouping","first"),
                  device=("device_deviceCategory","first"),
                  session_duration=("session_duration","max"),
                  source=("trafficSource_source","first"),
                  medium=("trafficSource_medium","first"),
                  referral_path=("trafficSource_referralPath","first"),
              ))

    # 결측/형 보정
    for c in ["pv","hits","bounces","visitNumber","session_duration","revenue"]:
        if c in sess.columns:
            sess[c] = pd.to_numeric(sess[c], errors="coerce").fillna(0)

    for c in ["source","medium","referral_path"]:
        if c in sess.columns:
            sess[c] = sess[c].astype(str).fillna("Unknown")

    # 파생
    sess["session_date"] = pd.to_datetime(sess["session_start"]).dt.date
    sess["session_hour"] = pd.to_datetime(sess["session_start"]).dt.hour
    sess["first_week"] = pd.to_datetime(sess["session_start"]).dt.to_period("W")
    sess["is_transaction"] = (sess["revenue"] > 0).astype(int)

    # 30일 내 재방문 누적 카운트
    s = sess.sort_values(["fullVisitorId","session_start"]).copy()
    first = s.groupby("fullVisitorId")["session_start"].transform("min")
    s["within_30d"] = (s["session_start"] > first) & (s["session_start"] <= first + pd.Timedelta(days=30))
    s["revisit_count_30d"] = s.groupby("fullVisitorId")["within_30d"].cumsum()
    sess = sess.merge(s[["session_id","revisit_count_30d"]], on="session_id", how="left")
    sess["revisit_count_30d"] = sess["revisit_count_30d"].fillna(0).astype(int)

    # 첫 방문 라벨
    first_idx = (sess.sort_values(["fullVisitorId","session_start"])
                      .groupby("fullVisitorId", as_index=False).head(1))
    sess["first_channel"] = sess["fullVisitorId"].map(dict(zip(first_idx["fullVisitorId"], first_idx["channel"])))
    sess["first_device"]  = sess["fullVisitorId"].map(dict(zip(first_idx["fullVisitorId"], first_idx["device"])))
    return sess

# =========================
# 업로더 + Lite 옵션
# =========================
with st.sidebar:
    st.header("데이터 업로드")
    up = st.file_uploader("CSV 파일 업로드 (utf-8/utf-8-sig 권장)", type=["csv"])
    st.caption("필수 컬럼: event_time, fullVisitorId")

if up is None:
    st.info("왼쪽에서 CSV를 업로드하면 대시보드가 생성됩니다.")
    st.stop()

file_size_mb = getattr(up, "size", 0) / (1024 * 1024)
with st.sidebar:
    st.markdown("---")
    st.subheader("메모리 세이프 옵션")
    lite_default = file_size_mb >= BIG_FILE_MB
    use_lite = st.checkbox("Lite 모드(대용량 권장)", value=lite_default,
                           help="필요 컬럼만 로드, 숫자 downcast, 문자열 category, 샘플링 적용")
    sample_frac = st.slider("샘플 비율", 0.05, 1.0,
                            DEFAULT_SAMPLE_FRAC if lite_default else 1.0, 0.05)
    st.caption(f"업로드 크기: ~{file_size_mb:.1f} MB")

# 실제 로드
df = load_df_memory_smart(up, use_lite=use_lite, sample_frac=sample_frac)
sess = build_session_table(df)

# 메모리 회수(원본 df는 필요한 최소만 유지)
keep_cols_df = [c for c in ["session_id","geoNetwork_continent","device_operatingSystem",
                            "channelGrouping","device_deviceCategory",
                            "totals_pageviews","totals_newVisits","date"] if c in df.columns]
df = df[keep_cols_df].copy()
gc.collect()

# 30일 이내 재방문 여부(사용자 단위)
def label_user_revisit_30d(sess: pd.DataFrame) -> pd.Series:
    s = sess.copy()
    first = s.groupby("fullVisitorId")["session_start"].transform("min")
    s["within_30d"] = (s["session_start"] > first) & (s["session_start"] <= first + pd.Timedelta(days=30))
    return s.groupby("fullVisitorId")["within_30d"].any().rename("revisit_30d")

# 7일 이동평균
def moving_avg(series: pd.Series, k: int = 7) -> pd.Series:
    return series.rolling(k, min_periods=1).mean()

# =========================
# 2) 사이드바 필터
# =========================
min_d = sess["session_date"].min()
max_d = sess["session_date"].max()

with st.sidebar:
    st.markdown("---")
    st.header("필터")
    start_d = st.date_input("시작일", min_d, min_value=min_d, max_value=max_d)
    end_d   = st.date_input("종료일", max_d, min_value=min_d, max_value=max_d)
    channels = sorted(sess["channel"].dropna().unique().tolist())
    devices  = sorted(sess["device"].dropna().unique().tolist())
    sel_channels = st.multiselect("채널", channels, default=channels)
    sel_devices  = st.multiselect("디바이스", devices, default=devices)
    min_day_sessions = st.number_input("일별 표본 최소 세션수(스파이크 가드레일)", 0, value=500, step=50)

# 필터 적용
mask = (
    (sess["session_date"] >= start_d) &
    (sess["session_date"] <= end_d) &
    (sess["channel"].isin(sel_channels)) &
    (sess["device"].isin(sel_devices))
)
sf = sess.loc[mask].copy()

# --- Activation 플래그 계산 (세션 단위, 고정 기준) ---
sf["act_pageviews"] = (sf["pv"] >= 3).astype(int)
sf["act_duration"]  = (sf["session_duration"] >= 180).astype(int)
sf["act_nonbounce"] = (sf["bounces"] == 0).astype(int)
sf["act_hits"]      = (sf["hits"] >= 10).astype(int)
sf["act_revisit"]   = (sf["revisit_count_30d"] >= 2).astype(int)
sf["activation"] = (
    (sf["act_pageviews"] == 1) &
    (sf["act_duration"]  == 1) &
    (sf["act_nonbounce"] == 1) &
    (sf["act_hits"]      == 1) &
    (sf["act_revisit"]   == 1)
).astype(int)

# 상단 KPI
k1,k2,k3,k4,k5,k6 = st.columns(6)
with k1: st.metric("사용자 수", f"{sf['fullVisitorId'].nunique():,}")
with k2: st.metric("세션 수", f"{sf['session_id'].nunique():,}")
with k3: st.metric("평균 PV/세션", f"{sf['pv'].mean():.2f}")
with k4: st.metric("Bounce Rate", f"{sf['bounces'].mean():.2%}")
with k5: st.metric("Median Duration(s)", f"{sf['session_duration'].median():.0f}")
with k6: st.metric("Activation 성공비율", f"{sf['activation'].mean():.2%}")
st.caption("Activation 기준: PV≥3 · 지속시간≥180초 · Hits≥10 · Bounce=0 · 30일 내 재방문≥2회")

# =========================
# 3) 탭 구성
# =========================
tabR, tabA, tabRef, tabRev, tabAcq = st.tabs(
    ["Retention", "Activation", "Referral", "Revenue", "Acquisition"]
)

# -------------------------
# Tab 1. Retention
# -------------------------
with tabR:
    st.subheader("1) 사용자 단위 Retention (30일 이내 재방문)")
    user_revisit = label_user_revisit_30d(sf)
    st.metric("30일 재방문율(사용자 기준)", f"{user_revisit.mean():.2%}")
    st.write("• 정의: 각 사용자의 첫 세션 이후 30일 안에 **한 번이라도** 재방문하면 성공으로 간주")

    # 첫 방문 채널/디바이스별 Retention
    first_session = sf.sort_values(["fullVisitorId","session_start"]) \
                      .groupby("fullVisitorId", as_index=False) \
                      .head(1)[["fullVisitorId","channel","device","session_start"]]
    first_map_ch  = dict(zip(first_session["fullVisitorId"], first_session["channel"]))
    first_map_dev = dict(zip(first_session["fullVisitorId"], first_session["device"]))
    user_df = user_revisit.reset_index()
    user_df["first_channel"] = user_df["fullVisitorId"].map(first_map_ch)
    user_df["first_device"]  = user_df["fullVisitorId"].map(first_map_dev)

    c1,c2 = st.columns(2)
    with c1:
        ch_tbl = user_df.groupby("first_channel")["revisit_30d"].mean().sort_values(ascending=False).reset_index()
        st.write("2) **첫 방문 채널별** 30일 재방문율"); st.dataframe(ch_tbl)
        st.bar_chart(ch_tbl.set_index("first_channel")["revisit_30d"])
    with c2:
        dev_tbl = user_df.groupby("first_device")["revisit_30d"].mean().sort_values(ascending=False).reset_index()
        st.write("3) **첫 방문 디바이스별** 30일 재방문율"); st.dataframe(dev_tbl)
        st.bar_chart(dev_tbl.set_index("first_device")["revisit_30d"])

    st.subheader("5) 채널 × PV구간별 구매 전환율")
    pv_bins = [0, 1, 3, 5, 10, 20, 50, 100, np.inf]
    pv_cut = pd.cut(sf["pv"], bins=pv_bins, right=False)
    pv_order = pv_cut.cat.categories.astype(str).tolist()
    sf["pv_bin"] = pv_cut

    conv_flag = sf["is_transaction"] if "is_transaction" in sf.columns else (sf["revenue"] > 0).astype(int)
    sf["_conv_flag"] = conv_flag.astype(int)

    conv_tbl = (sf.groupby(["channel", "pv_bin"], observed=False)
                  .agg(sessions=("session_id", "count"),
                       conversions=("_conv_flag", "sum"))
                  .reset_index())
    conv_tbl["conv_rate"] = np.where(conv_tbl["sessions"] > 0,
                                     conv_tbl["conversions"] / conv_tbl["sessions"], np.nan)
    conv_tbl["pv_bin"] = conv_tbl["pv_bin"].astype(str)

    heat = alt.Chart(conv_tbl).mark_rect().encode(
        x=alt.X("pv_bin:N", sort=pv_order, title="PV 구간"),
        y=alt.Y("channel:N", sort="-x", title="채널"),
        color=alt.Color("conv_rate:Q", title="전환율"),
        tooltip=[alt.Tooltip("channel:N", title="채널"),
                 alt.Tooltip("pv_bin:N", title="PV 구간"),
                 alt.Tooltip("sessions:Q", title="세션수", format=",.0f"),
                 alt.Tooltip("conv_rate:Q", title="전환율", format=".2%")]
    ).properties(height=320)
    st.altair_chart(heat, use_container_width=True)
    sf.drop(columns=["_conv_flag"], inplace=True, errors="ignore")

# -------------------------
# Tab 2. Activation
# -------------------------
with tabA:
    st.markdown("""
    **✅ Activation 성공 기준**  
    1) PV ≥ 3 · 2) 지속시간 ≥ 180초 · 3) Bounce=0 · 4) Hits ≥ 10 · 5) 30일 내 재방문 ≥ 2회
    """)
    st.subheader("1) 세션단위 Activation 성공 비율 (일별)")
    daily = (sf.groupby("session_date", as_index=False)
               .agg(activation=("activation","mean"),
                    sessions=("session_id","count")))
    daily["activation_ma7"] = moving_avg(daily["activation"], 7)
    daily["low_sample"] = daily["sessions"] < min_day_sessions

    base = alt.Chart(daily).encode(x="session_date:T")
    bars = base.mark_bar(opacity=0.3).encode(y=alt.Y("sessions:Q", axis=alt.Axis(title="Sessions")))
    line = base.mark_line().encode(y=alt.Y("activation:Q", axis=alt.Axis(title="Activation Rate", format="~%")))
    ma7  = base.mark_line(strokeDash=[4,4]).encode(y="activation_ma7:Q", color=alt.value("gray"))
    pts  = base.mark_circle(size=20).encode(y="activation:Q",
            color=alt.condition("datum.low_sample", alt.value("#aaaaaa"), alt.value("#1f77b4")))
    st.altair_chart(alt.layer(bars, line, ma7, pts).resolve_scale(y='independent')
                    .properties(height=320, width="container"), use_container_width=True)
    st.caption(f"점 색상: 일 세션수 < {min_day_sessions} (회색)")

    st.subheader("2) 채널/디바이스별 Activation")
    ch_act = (sf.groupby("channel", observed=False)
                .agg(sessions=("session_id","count"), act_rate=("activation","mean"))
                .sort_values("act_rate", ascending=False)).reset_index()
    st.dataframe(ch_act); st.bar_chart(ch_act.set_index("channel")["act_rate"])

    dev_act = (sf.groupby("device", observed=False)
                 .agg(sessions=("session_id","count"), act_rate=("activation","mean"))
                 .sort_values("act_rate", ascending=False)).reset_index()
    st.dataframe(dev_act); st.bar_chart(dev_act.set_index("device")["act_rate"])

    st.subheader("3) 심층 지표 & 퍼널")
    pv_bins2 = [0,1,3,5,10,20,50,100,np.inf]; pv_cut = pd.cut(sf["pv"], pv_bins2, right=False)
    dur_bins = [0,30,60,120,180,300,600,1200,1800,3600,np.inf]; dur_cut = pd.cut(sf["session_duration"], dur_bins, right=False)
    hit_bins = [0,5,10,20,50,100,200,np.inf]; hit_cut = pd.cut(sf["hits"], hit_bins, right=False)
    pv_act = sf.groupby(pv_cut, observed=False)["activation"].mean().reset_index()
    pv_act["pv_bin"] = pv_act.iloc[:,0].astype(str); pv_act["rate"] = pv_act["activation"]
    st.bar_chart(pv_act.set_index("pv_bin")["rate"])
    funnel = (sf.groupby(pv_cut, observed=False)
                .agg(sessions=("session_id","count"), act_sessions=("activation","sum"))
                .reset_index())
    funnel = funnel.rename(columns={funnel.columns[0]:"pv_bin"})
    funnel["pv_bin"] = funnel["pv_bin"].astype(str)
    funnel["activation_rate"] = np.where(funnel["sessions"]>0, funnel["act_sessions"]/funnel["sessions"], np.nan)
    st.dataframe(funnel)

# -------------------------
# Tab 3. Referral
# -------------------------
with tabRef:
    st.subheader("Referral 분석")
    ref = sf[(sf["medium"].str.lower() == "referral") | (sf["channel"] == "Referral")].copy()
    if ref.empty:
        st.warning("Referral 세션이 없습니다. 사이드바 필터를 확인해 주세요.")
        st.stop()

    top_sessions = (ref.groupby("source", observed=False)["session_id"]
                      .count().reset_index(name="sessions")
                      .sort_values("sessions", ascending=False).head(10))
    st.markdown("### Top 10 Referral Sources (세션 수)")
    ch_sessions = (alt.Chart(top_sessions).mark_bar().encode(
        x=alt.X("sessions:Q", title="Number of Sessions"),
        y=alt.Y("source:N", sort="-x", title="Source"),
        tooltip=["source","sessions"]
    ).properties(height=380))
    st.altair_chart(ch_sessions, use_container_width=True)

    ref_top = ref.merge(top_sessions, on="source", how="inner")
    by_source = (ref_top.groupby("source", observed=False)
                  .agg(avg_pv=("pv","mean"), avg_hits=("hits","mean"),
                       bounce_rate=("bounces","mean"), sessions=("session_id","count"))
                  .reset_index().sort_values("sessions", ascending=False).head(10))
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Avg Pageviews (Top 10)")
        st.altair_chart(alt.Chart(by_source).mark_bar().encode(
            x=alt.X("avg_pv:Q", title="Average Pageviews"),
            y=alt.Y("source:N", sort="-x", title="Source"),
            tooltip=["source", alt.Tooltip("avg_pv:Q", format=".2f")]
        ).properties(height=420), use_container_width=True)
    with c2:
        st.markdown("#### Avg Hits (Top 10)")
        st.altair_chart(alt.Chart(by_source).mark_bar().encode(
            x=alt.X("avg_hits:Q", title="Average Hits"),
            y=alt.Y("source:N", sort="-x", title="Source"),
            tooltip=["source", alt.Tooltip("avg_hits:Q", format=".2f")]
        ).properties(height=420), use_container_width=True)
    with c3:
        st.markdown("#### Bounce Rate (Top 10)")
        st.altair_chart(alt.Chart(by_source.assign(br_pct=by_source["bounce_rate"]*100))
            .mark_bar().encode(
            x=alt.X("br_pct:Q", title="Bounce Rate (%)"),
            y=alt.Y("source:N", sort="-x", title="Source"),
            tooltip=["source", alt.Tooltip("br_pct:Q", format=".1f")]
        ).properties(height=420), use_container_width=True)
    st.markdown("#### Raw Table")
    st.dataframe(by_source[["source","sessions","avg_pv","avg_hits","bounce_rate"]]
                 .rename(columns={"avg_pv":"avg_pageviews","bounce_rate":"bounce_rate(0-1)"}))

# -------------------------
# Tab 4. Revenue
# -------------------------
with tabRev:
    st.subheader("🌍 Distribution of Continent")
    df_f = df[df["session_id"].isin(sf["session_id"])].copy()
    if "geoNetwork_continent" in df_f.columns:
        cont_tbl = (df_f["geoNetwork_continent"].fillna("Unknown")
                    .value_counts().rename_axis("continent").reset_index(name="sessions"))
        chart_cont = alt.Chart(cont_tbl).mark_bar().encode(
            x=alt.X("sessions:Q", title="Sessions"),
            y=alt.Y("continent:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("continent:N", title="Continent"),
                     alt.Tooltip("sessions:Q", title="Sessions", format=",.0f")]
        ).properties(width=420, height=260)
        st.altair_chart(chart_cont, use_container_width=False)
    else:
        st.info("continent 컬럼이 없어 분포 차트를 건너뜁니다.")

    st.subheader("📱 Mobile vs. Desktop Traffic Share")
    dev_cnt = sf["device"].value_counts()
    obs_mobile = int(dev_cnt.get("mobile", 0)); obs_desktop = int(dev_cnt.get("desktop", 0))
    obs_total_md = max(obs_mobile + obs_desktop, 1)
    obs_mobile_share = obs_mobile / obs_total_md * 100.0
    obs_desktop_share = obs_desktop / obs_total_md * 100.0
    market_mobile_share, market_desktop_share = 64.3, 35.7

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    x = [0, 1]; labels = ["2016–2017 (Observed)", "2025 (Market)"]
    ax.bar(x[0], obs_mobile_share, label="Mobile"); ax.bar(x[0], obs_desktop_share, bottom=obs_mobile_share, label="Desktop")
    ax.bar(x[1], market_mobile_share); ax.bar(x[1], market_desktop_share, bottom=market_mobile_share)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 105); ax.set_ylabel("Traffic Share (%)")
    for i,(m,d) in enumerate([(obs_mobile_share,obs_desktop_share),(market_mobile_share,market_desktop_share)]):
        ax.text(x[i], m/2, f"{m:.1f}%", ha="center", va="center", fontsize=10, weight="bold")
        ax.text(x[i], m+d/2, f"{d:.1f}%", ha="center", va="center", fontsize=10, weight="bold")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.20), frameon=False)
    st.pyplot(fig, use_container_width=False)

    st.markdown("""
    **발견:** 데이터셋에 기록된 거래 수익이 부족/없어 직접 수익 예측은 어렵습니다.  
    **시사점:** 이탈률 감소와 참여도 증대(페이지뷰/체류시간↑)가 전환/수익 증대의 선행 과제입니다.
    """)

# -------------------------
# Tab 5. Acquisition (간단 버전: 시각화 위주)
# -------------------------
with tabAcq:
    plt.rc('font', family='Malgun Gothic'); plt.rc('axes', unicode_minus=False)
    st.header("1️⃣ 유입 규모 & 신규 고객")

    # 채널별 세션 & 신규
    if {"channelGrouping","totals_newVisits"}.issubset(df.columns):
        channel_summary = (df.groupby("channelGrouping")
                             .agg(sessions=("session_id","nunique"),
                                  new_sessions=("totals_newVisits","sum")).reset_index())
        channel_summary["new_visit_ratio"] = channel_summary["new_sessions"] / channel_summary["sessions"]
        fig, ax1 = plt.subplots(figsize=(8,5))
        ax1.bar(channel_summary["channelGrouping"], channel_summary["sessions"], alpha=0.7)
        ax1.set_ylabel("Sessions", color="blue")
        ax2 = ax1.twinx()
        ax2.plot(channel_summary["channelGrouping"], channel_summary["new_visit_ratio"], color="red", marker="o")
        ax2.set_ylabel("New Visit Ratio", color="red")
        plt.xticks(rotation=45); st.pyplot(fig)
    else:
        st.info("채널/신규 방문 컬럼이 부족합니다.")

    # Device × Channel 히트맵
    if {"device_deviceCategory","channelGrouping","totals_pageviews"}.issubset(df.columns):
        pivot = df.pivot_table(index="device_deviceCategory", columns="channelGrouping",
                               values="totals_pageviews", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(10,5))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
        plt.title("Average Pageviews by Device × Channel"); st.pyplot(fig)

    st.header("4️⃣ 채널별 신규 vs 재방문 비율")
    if {"channelGrouping","totals_newVisits"}.issubset(df.columns):
        channel_visits = (df.groupby("channelGrouping")["totals_newVisits"]
                            .agg(sessions="count", new_sessions="sum").reset_index())
        channel_visits["repeat_sessions"] = channel_visits["sessions"] - channel_visits["new_sessions"]
        channel_visits["new_ratio"] = channel_visits["new_sessions"] / channel_visits["sessions"]
        channel_visits["repeat_ratio"] = channel_visits["repeat_sessions"] / channel_visits["sessions"]
        fig, ax = plt.subplots(figsize=(10,6))
        ax.bar(channel_visits["channelGrouping"], channel_visits["new_ratio"], label="신규")
        ax.bar(channel_visits["channelGrouping"], channel_visits["repeat_ratio"],
               bottom=channel_visits["new_ratio"], label="재방문")
        ax.set_ylabel("비율"); ax.set_xlabel("채널"); plt.xticks(rotation=45); plt.legend()
        plt.title("채널별 신규 vs 재방문 비율"); st.pyplot(fig)

st.success("완료! 왼쪽 사이드바에서 기간/채널/디바이스를 바꿔가며 Retention~Acquisition을 탐색하세요.")
