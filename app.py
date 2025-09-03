# app.py
# 실행: streamlit run app.py

import io
import gc
from datetime import timedelta
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---- Altair 큰 데이터 경고 완화
alt.data_transformers.disable_max_rows()

# =========================
# 0) 기본 설정
# =========================
SESSION_GAP_MIN = 30               # session_id가 없을 때 보수적 세션화(30분 룰)
BIG_FILE_MB = 120                  # 이 크기 이상이면 Lite 모드 권장
DEFAULT_SAMPLE_FRAC = 0.35         # Lite 모드 기본 샘플 비율

st.set_page_config(page_title="RARRA Dashboard", layout="wide")
st.title("📊 RARRA Dashboard (Retention • Activation • Referral • Revenue • Acquisition)")
st.caption("Kaggle GA Customer Revenue Dataset | 세션 단위 분석 (Memory-Optimized)")

# =========================
# 1) 업로더
# =========================
with st.sidebar:
    st.header("데이터 업로드")
    up = st.file_uploader("CSV 파일을 업로드하세요 (utf-8/utf-8-sig 권장)", type=["csv"])
    st.caption("필수 컬럼: event_time, fullVisitorId (string). 기타 컬럼은 있으면 활용합니다.")

if up is None:
    st.info("왼쪽에서 CSV를 업로드하면 대시보드가 생성됩니다.")
    st.stop()

# =========================
# 2) 메모리 세이프 옵션
# =========================
file_size_mb = getattr(up, "size", 0) / (1024 * 1024)
with st.sidebar:
    st.markdown("---")
    st.subheader("메모리 세이프 옵션")
    lite_default = file_size_mb >= BIG_FILE_MB
    use_lite = st.checkbox("Lite 모드로 로딩(권장: 대용량)", value=lite_default,
                           help="필요 컬럼만 로드하고, 숫자 downcast/문자열 category/샘플링을 적용해 메모리 사용을 줄입니다.")
    sample_frac = st.slider("샘플 비율", 0.05, 1.0, DEFAULT_SAMPLE_FRAC if lite_default else 1.0, 0.05,
                            help="Lite 모드일 때 적용. 1.0이면 샘플링 없음.")
    st.caption(f"업로드 크기: ~{file_size_mb:.1f} MB")

# =========================
# 3) 로딩 함수
# =========================
REQUIRED = ["event_time", "fullVisitorId"]
OPTIONAL = [
    "session_id","totals_pageviews","totals_hits","totals_bounces","totals_newVisits",
    "totals_transactionRevenue","session_duration","visitNumber",
    "channelGrouping","device_deviceCategory",
    "trafficSource_source","trafficSource_medium","trafficSource_referralPath",
    "geoNetwork_continent","device_operatingSystem","date","is_transaction"
]
WANTED = REQUIRED + OPTIONAL

@st.cache_data(show_spinner=True)
def load_df_memory_smart(file, use_lite: bool, sample_frac: float) -> pd.DataFrame:
    # 1) 헤더만 먼저 읽어서 존재 컬럼 파악
    file.seek(0)
    first_bytes = file.getvalue() if hasattr(file, "getvalue") else file.read()
    bio = io.BytesIO(first_bytes)
    head = pd.read_csv(io.BytesIO(first_bytes), nrows=0, dtype=str)
    usecols = [c for c in head.columns if c in WANTED]

    if not all(c in head.columns for c in REQUIRED):
        missing = [c for c in REQUIRED if c not in head.columns]
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    # 2) 필요한 컬럼만 로드
    df = pd.read_csv(io.BytesIO(first_bytes), dtype=str, low_memory=False, usecols=usecols)

    # 3) 샘플링 (Lite 모드)
    if use_lite and 0 < sample_frac < 1.0:
        # event_time 기준으로 균등 샘플링(시간분포 보존용)
        if "event_time" in df.columns:
            # 빠르게 난수 마스크
            mask = np.random.RandomState(42).rand(len(df)) < sample_frac
            df = df.loc[mask].copy()
        else:
            df = df.sample(frac=sample_frac, random_state=42).copy()

    # 4) 타입 변환
    # 시간
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df["event_time_naive"] = df["event_time"].dt.tz_convert("UTC").dt.tz_localize(None)

    # 숫자
    def to_num(cols):
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
                # downcast
                if pd.api.types.is_float_dtype(df[c]):
                    df[c] = pd.to_numeric(df[c], downcast="float")
                elif pd.api.types.is_integer_dtype(df[c]):
                    df[c] = pd.to_numeric(df[c], downcast="integer")
    to_num([
        "totals_pageviews","totals_hits","totals_bounces","totals_newVisits",
        "totals_transactionRevenue","session_duration","visitNumber","is_transaction"
    ])

    # 문자열
    if "fullVisitorId" in df.columns:
        df["fullVisitorId"] = df["fullVisitorId"].astype(str)

    # 결측 기본값
    for c in ["channelGrouping","device_deviceCategory","trafficSource_source",
              "trafficSource_medium","trafficSource_referralPath","geoNetwork_continent",
              "device_operatingSystem","date"]:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown")
        else:
            df[c] = "Unknown"

    # session_id 없으면: 메모리 절약을 위해 여기서는 ‘간이 세션화’ (유저별 시간정렬 → 30분 룰)
    if "session_id" not in df.columns:
        # 정렬 시 메모리 피크를 줄이기 위해 필요한 컬럼만 뽑아서 정렬 후 다시 병합
        tmp = df[["fullVisitorId","event_time_naive"]].copy()
        tmp = tmp.sort_values(["fullVisitorId","event_time_naive"]).reset_index(drop=True)
        diff = tmp.groupby("fullVisitorId")["event_time_naive"].diff().dt.total_seconds()
        new_sess = (diff.isna()) | (diff > SESSION_GAP_MIN*60)
        sess_num = new_sess.groupby(tmp["fullVisitorId"]).cumsum().astype("int32")
        df = df.loc[tmp.index].copy()                  # 동일 순서 보장
        df["session_id"] = df["fullVisitorId"].astype(str) + "_" + sess_num.astype(str)

    # 문자열을 category로 (메모리 절감)
    for c in ["channelGrouping","device_deviceCategory","trafficSource_source",
              "trafficSource_medium","trafficSource_referralPath",
              "geoNetwork_continent","device_operatingSystem","session_id","fullVisitorId"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    return df

# 실제 로딩
df = load_df_memory_smart(up, use_lite=use_lite, sample_frac=sample_frac)

# =========================
# 4) 세션 테이블 구성
# =========================
@st.cache_data(show_spinner=True)
def build_session_table(df: pd.DataFrame) -> pd.DataFrame:
    sess = (
        df.groupby("session_id", as_index=False)
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
        )
    )

    for c in ["source","medium","referral_path"]:
        if c in sess.columns:
            sess[c] = sess[c].astype(str).fillna("Unknown")

    # 파생
    sess["session_date"] = pd.to_datetime(sess["session_start"]).dt.date
    sess["session_hour"] = pd.to_datetime(sess["session_start"]).dt.hour
    sess["first_week"] = pd.to_datetime(sess["session_start"]).dt.to_period("W")
    sess["is_transaction"] = (pd.to_numeric(sess["revenue"], errors="coerce").fillna(0) > 0).astype(int)

    # 30일 내 재방문 누적 카운트
    s = sess.sort_values(["fullVisitorId","session_start"]).copy()
    first = s.groupby("fullVisitorId")["session_start"].transform("min")
    s["within_30d"] = (s["session_start"] > first) & (s["session_start"] <= first + pd.Timedelta(days=30))
    s["revisit_count_30d"] = s.groupby("fullVisitorId")["within_30d"].cumsum()
    sess = sess.merge(s[["session_id","revisit_count_30d"]], on="session_id", how="left")
    sess["revisit_count_30d"] = sess["revisit_count_30d"].fillna(0).astype("int16")

    # 숫자 보정 & downcast
    for c in ["pv","hits","bounces","visitNumber","session_duration"]:
        if c in sess.columns:
            sess[c] = pd.to_numeric(sess[c], errors="coerce").fillna(0)
            sess[c] = pd.to_numeric(sess[c], downcast="integer")

    # 첫 방문 라벨
    first_idx = (
        sess.sort_values(["fullVisitorId","session_start"])
        .groupby("fullVisitorId", as_index=False)
        .head(1)
    )
    sess["first_channel"] = sess["fullVisitorId"].map(dict(zip(first_idx["fullVisitorId"], first_idx["channel"])))
    sess["first_device"]  = sess["fullVisitorId"].map(dict(zip(first_idx["fullVisitorId"], first_idx["device"])))

    return sess

sess = build_session_table(df)

# 원본 DF는 필요한 최소 컬럼만 남기고 슬림화하여 메모리 회수
keep_cols_df = [c for c in ["session_id","geoNetwork_continent","device_operatingSystem","channelGrouping",
                            "device_deviceCategory","totals_pageviews","totals_newVisits","date"] if c in df.columns]
df = df[keep_cols_df].copy()
gc.collect()

# =========================
# 5) 사이드바(필터)
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

    st.subheader("Activation 기준 (고정)")
    st.caption("PV≥3 · 지속시간≥180초 · Hits≥10 · Bounce=0 · 30일 내 재방문≥2회")
    min_day_sessions = st.number_input("일별 표본 최소 세션수(스파이크 가드레일)", 0, value=500, step=50)

# 필터 적용
mask = (
    (sess["session_date"] >= start_d) &
    (sess["session_date"] <= end_d) &
    (sess["channel"].isin(sel_channels)) &
    (sess["device"].isin(sel_devices))
)
sf = sess.loc[mask].copy()

# =========================
# 6) Activation 플래그 계산
# =========================
sf["act_pageviews"] = (sf["pv"] >= 3).astype(int)
sf["act_duration"]  = (sf["session_duration"] >= 180).astype(int)
sf["act_nonbounce"] = (sf["bounces"] == 0).astype(int)
sf["act_hits"]      = (sf["hits"] >= 10).astype(int)
sf["act_revisit"]   = (sf["revisit_count_30d"] >= 2).astype(int)

sf["activation"] = (
    (sf["act_pageviews"]==1) &
    (sf["act_duration"]==1) &
    (sf["act_nonbounce"]==1) &
    (sf["act_hits"]==1) &
    (sf["act_revisit"]==1)
).astype(int)

# =========================
# 7) KPI
# =========================
k1,k2,k3,k4,k5,k6 = st.columns(6)
with k1: st.metric("사용자 수", f"{sf['fullVisitorId'].nunique():,}")
with k2: st.metric("세션 수", f"{sf['session_id'].nunique():,}")
with k3: st.metric("평균 PV/세션", f"{sf['pv'].mean():.2f}")
with k4: st.metric("Bounce Rate", f"{sf['bounces'].mean():.2%}")
with k5: st.metric("Median Duration(s)", f"{sf['session_duration'].median():.0f}")
with k6: st.metric("Activation 성공비율", f"{sf['activation'].mean():.2%}")

st.caption("Activation 기준: PV≥3 · 지속시간≥180초 · Hits≥10 · Bounce=0 · 30일 내 재방문≥2회")

# =========================
# 8) 탭
# =========================
tabR, tabA, tabRef, tabRev, tabAcq = st.tabs(
    ["Retention", "Activation", "Referral", "Revenue", "Acquisition"]
)

# -------------------------
# Retention
# -------------------------
def label_user_revisit_30d(sess_local: pd.DataFrame) -> pd.Series:
    s = sess_local.copy()
    first = s.groupby("fullVisitorId")["session_start"].transform("min")
    s["within_30d"] = (s["session_start"] > first) & (s["session_start"] <= first + pd.Timedelta(days=30))
    return s.groupby("fullVisitorId")["within_30d"].any().rename("revisit_30d")

with tabR:
    st.subheader("1) 사용자 단위 Retention (30일 이내 재방문)")
    user_revisit = label_user_revisit_30d(sf)
    st.metric("30일 재방문율(사용자 기준)", f"{user_revisit.mean():.2%}")
    st.write("• 정의: 각 사용자의 첫 세션 이후 30일 안에 **한 번이라도** 재방문하면 성공")

    # 첫 방문 채널/디바이스별
    first_session = (
        sf.sort_values(["fullVisitorId","session_start"])
          .groupby("fullVisitorId", as_index=False).head(1)
          [["fullVisitorId","channel","device","session_start"]]
    )
    first_map_ch = dict(zip(first_session["fullVisitorId"], first_session["channel"]))
    first_map_dev = dict(zip(first_session["fullVisitorId"], first_session["device"]))
    user_df = user_revisit.reset_index()
    user_df["first_channel"] = user_df["fullVisitorId"].map(first_map_ch)
    user_df["first_device"]  = user_df["fullVisitorId"].map(first_map_dev)

    c1,c2 = st.columns(2)
    with c1:
        ch_tbl = user_df.groupby("first_channel")["revisit_30d"].mean().sort_values(ascending=False).reset_index()
        st.dataframe(ch_tbl)
        st.bar_chart(ch_tbl.set_index("first_channel")["revisit_30d"])
    with c2:
        dev_tbl = user_df.groupby("first_device")["revisit_30d"].mean().sort_values(ascending=False).reset_index()
        st.dataframe(dev_tbl)
        st.bar_chart(dev_tbl.set_index("first_device")["revisit_30d"])

    st.subheader("5) 채널 × PV구간별 구매 전환율")
    pv_bins = [0,1,3,5,10,20,50,100, np.inf]
    pv_cut = pd.cut(sf["pv"], bins=pv_bins, right=False)
    pv_order = pv_cut.cat.categories.astype(str).tolist()
    sf["pv_bin"] = pv_cut

    conv_flag = sf["is_transaction"].astype(int) if "is_transaction" in sf.columns else (sf["revenue"] > 0).astype(int)
    sf["_conv_flag"] = conv_flag

    conv_tbl = (
        sf.groupby(["channel","pv_bin"], observed=False)
          .agg(sessions=("session_id","count"), conversions=("_conv_flag","sum"))
          .reset_index()
    )
    conv_tbl["conv_rate"] = np.where(conv_tbl["sessions"]>0, conv_tbl["conversions"]/conv_tbl["sessions"], np.nan)
    conv_tbl["pv_bin"] = conv_tbl["pv_bin"].astype(str)

    heat = alt.Chart(conv_tbl).mark_rect().encode(
        x=alt.X("pv_bin:N", sort=pv_order, title="PV 구간"),
        y=alt.Y("channel:N", sort="-x", title="채널"),
        color=alt.Color("conv_rate:Q", title="전환율"),
        tooltip=[
            alt.Tooltip("channel:N", title="채널"),
            alt.Tooltip("pv_bin:N", title="PV 구간"),
            alt.Tooltip("sessions:Q", title="세션수", format=",.0f"),
            alt.Tooltip("conv_rate:Q", title="전환율", format=".2%"),
        ],
    ).properties(height=320)
    st.altair_chart(heat, use_container_width=True)
    sf.drop(columns=["_conv_flag"], inplace=True, errors="ignore")

# -------------------------
# Activation
# -------------------------
def moving_avg(s, k=7): return s.rolling(k, min_periods=1).mean()

with tabA:
    st.markdown("""
**✅ Activation 성공 기준**  
1. 페이지뷰(PV) ≥ 3  
2. 세션 지속시간 ≥ 180초 (3분)  
3. Bounce = 0  
4. Hits ≥ 10  
5. 30일 내 재방문 횟수 ≥ 2회
""")
    st.subheader("1) 세션단위 Activation 성공 비율 (일별)")
    daily = sf.groupby("session_date", as_index=False).agg(activation=("activation","mean"),
                                                          sessions=("session_id","count"))
    daily["activation_ma7"] = moving_avg(daily["activation"], 7)
    daily["low_sample"] = daily["sessions"] < min_day_sessions

    base = alt.Chart(daily).encode(x="session_date:T")
    bars = base.mark_bar(opacity=0.3).encode(y=alt.Y("sessions:Q", axis=alt.Axis(title="Sessions")))
    line = base.mark_line().encode(y=alt.Y("activation:Q", axis=alt.Axis(title="Activation Rate", format="~%")))
    ma7  = base.mark_line(strokeDash=[4,4], color="gray").encode(y="activation_ma7:Q")
    pts  = base.mark_circle(size=20).encode(y="activation:Q",
                color=alt.condition("datum.low_sample", alt.value("#aaaaaa"), alt.value("#1f77b4")))
    st.altair_chart(alt.layer(bars, line, ma7, pts).resolve_scale(y='independent').properties(height=320),
                    use_container_width=True)
    st.caption(f"점 색상: 일 세션수 < {min_day_sessions} (회색)")

    st.subheader("2) 채널/디바이스별 Activation")
    ch_act = sf.groupby("channel", observed=False).agg(sessions=("session_id","count"),
                                                       act_rate=("activation","mean")).reset_index()
    dev_act = sf.groupby("device", observed=False).agg(sessions=("session_id","count"),
                                                       act_rate=("activation","mean")).reset_index()
    st.dataframe(ch_act.sort_values("act_rate", ascending=False))
    st.bar_chart(ch_act.set_index("channel")["act_rate"])
    st.dataframe(dev_act.sort_values("act_rate", ascending=False))
    st.bar_chart(dev_act.set_index("device")["act_rate"])

    st.subheader("3) 퍼널 관점: PV 구간별 세션 수 & Activation")
    pv_bins2 = [0,1,3,5,10,20,50,100, np.inf]
    pv_cut2 = pd.cut(sf["pv"], pv_bins2, right=False)
    funnel = (sf.groupby(pv_cut2, observed=False)
                .agg(sessions=("session_id","count"),
                     act_sessions=("activation","sum"))
                .reset_index())
    funnel = funnel.rename(columns={funnel.columns[0]:"pv_bin"})
    funnel["pv_bin"] = funnel["pv_bin"].astype(str)
    funnel["activation_rate"] = np.where(funnel["sessions"]>0, funnel["act_sessions"]/funnel["sessions"], np.nan)
    st.dataframe(funnel)

# -------------------------
# Referral
# -------------------------
with tabRef:
    st.subheader("Referral 분석")
    if "medium" in sf.columns and "channel" in sf.columns:
        ref = sf[(sf["medium"].str.lower()=="referral") | (sf["channel"]=="Referral")].copy()
    else:
        ref = pd.DataFrame()

    if ref.empty:
        st.warning("Referral 세션이 없습니다. 필터/데이터를 확인하세요.")
    else:
        top_sessions = (ref.groupby("source", observed=False)["session_id"].count()
                          .reset_index(name="sessions").sort_values("sessions", ascending=False).head(10))
        st.markdown("### Top 10 Referral Sources (세션 수)")
        st.altair_chart(
            alt.Chart(top_sessions).mark_bar().encode(
                x=alt.X("sessions:Q", title="Number of Sessions"),
                y=alt.Y("source:N", sort="-x", title="Source"),
                tooltip=["source","sessions"]
            ).properties(height=380),
            use_container_width=True
        )

# -------------------------
# Revenue
# -------------------------
with tabRev:
    st.subheader("🌍 Distribution of Continent")
    if {"session_id","geoNetwork_continent"}.issubset(df.columns):
        df_f = df[df["session_id"].isin(sf["session_id"])].copy()
        cont_tbl = (df_f["geoNetwork_continent"]
                    .astype(str).fillna("Unknown").value_counts()
                    .rename_axis("continent").reset_index(name="sessions"))
        st.altair_chart(
            alt.Chart(cont_tbl).mark_bar().encode(
                x=alt.X("sessions:Q", title="Sessions"),
                y=alt.Y("continent:N", sort="-x", title=None),
                tooltip=[alt.Tooltip("continent:N", title="Continent"),
                         alt.Tooltip("sessions:Q", title="Sessions", format=",.0f")]
            ).properties(height=260),
            use_container_width=True
        )
    else:
        st.info("대륙 정보(geoNetwork_continent)가 없어 생략합니다.")

# -------------------------
# Acquisition (간단 버전: 메모리 절감)
# -------------------------
with tabAcq:
    st.header("1️⃣ 유입 규모 & 신규 고객")
    if {"channelGrouping","session_id","totals_newVisits"}.issubset(df.columns):
        # 채널별
        ch_sum = (df.groupby("channelGrouping")
                    .agg(sessions=("session_id","nunique"),
                         new_sessions=("totals_newVisits","sum"))
                    .reset_index())
        ch_sum["new_visit_ratio"] = np.where(ch_sum["sessions"]>0,
                                             ch_sum["new_sessions"]/ch_sum["sessions"], 0.0)
        ch1 = alt.Chart(ch_sum).mark_bar().encode(x="channelGrouping:N", y=alt.Y("sessions:Q", title="Sessions"))
        ch2 = alt.Chart(ch_sum).mark_line(point=True).encode(
            x="channelGrouping:N",
            y=alt.Y("new_visit_ratio:Q", axis=alt.Axis(format="~%"), title="New Visit Ratio")
        )
        st.altair_chart(alt.layer(ch1, ch2).resolve_scale(y="independent").properties(height=320),
                        use_container_width=True)

    # Device × Channel 히트맵 (평균 PV)
    if {"device_deviceCategory","channelGrouping","totals_pageviews"}.issubset(df.columns):
        pivot = (df.pivot_table(index="device_deviceCategory", columns="channelGrouping",
                                values="totals_pageviews", aggfunc="mean")
                 .fillna(0).reset_index())
        pv_melt = pivot.melt(id_vars="device_deviceCategory", var_name="channelGrouping", value_name="avg_pv")
        st.subheader("📌 Device × Channel 히트맵 (평균 페이지뷰)")
        st.altair_chart(
            alt.Chart(pv_melt).mark_rect().encode(
                x=alt.X("channelGrouping:N", title="Channel"),
                y=alt.Y("device_deviceCategory:N", title="Device"),
                color=alt.Color("avg_pv:Q", title="Avg PV"),
                tooltip=["device_deviceCategory","channelGrouping", alt.Tooltip("avg_pv:Q", format=".1f")]
            ).properties(height=300),
            use_container_width=True
        )

st.success("완료! Lite 모드/샘플 비율을 조절해 메모리 한도를 피하면서 Retention~Acquisition을 탐색하세요.")
