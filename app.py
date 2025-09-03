# app.py
# 실행: streamlit run app.py

import io
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ----- (선택) matplotlib/seaborn이 없을 때도 앱이 죽지 않도록 가드 -----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MPL = True
except Exception:
    plt = None
    sns = None
    HAS_MPL = False
# ---------------------------------------------------------------------

# =========================
# 0) 기본 설정
# =========================
SESSION_GAP_MIN = 30  # (데이터에 session_id가 없을 때만 사용)
st.set_page_config(page_title="RARRA Dashboard", layout="wide")
st.set_option("client.showErrorDetails", True)
st.title("📊 RARRA Dashboard (Retention • Activation • Referral • Revenue • Acquisition)")
st.caption("Upload your GA-like session CSV • 세션 단위 분석")

# =========================
# 1) 데이터 업로드
# =========================
with st.sidebar:
    st.header("데이터 업로드")
    uploaded = st.file_uploader("CSV 파일을 업로드하세요 (utf-8/utf-8-sig 권장)", type=["csv"])
    st.caption("필수 컬럼: event_time, fullVisitorId (string). 기타 컬럼은 있으면 사용합니다.")

if uploaded is None:
    st.info("왼쪽에서 CSV를 업로드하면 대시보드가 생성됩니다.")
    st.stop()

# =========================
# 2) 데이터 로딩 & 세션 테이블
# =========================
@st.cache_data(show_spinner=True)
def load_df_from_file(file) -> pd.DataFrame:
    # 업로드 객체/바이트 모두 허용
    try:
        df = pd.read_csv(file, dtype=str, low_memory=False)
    except Exception:
        buf = file.getvalue() if hasattr(file, "getvalue") else file
        df = pd.read_csv(io.BytesIO(buf), dtype=str, low_memory=False)

    # 시간 처리
    if "event_time" not in df.columns:
        raise ValueError("event_time 컬럼이 필요합니다. (UTC 가능)")
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df["event_time_naive"] = df["event_time"].dt.tz_convert("UTC").dt.tz_localize(None)

    # 숫자형 변환(있으면 사용)
    def to_num_frame(_df, cols):
        for col in cols:
            if col in _df.columns:
                _df[col] = pd.to_numeric(_df[col], errors="coerce").fillna(0)
        return _df

    df = to_num_frame(
        df,
        [
            "totals_pageviews",
            "totals_hits",
            "totals_bounces",
            "totals_newVisits",
            "totals_transactionRevenue",
            "session_duration",
            "is_transaction",
            "visitNumber",
        ],
    )

    # 문자열 컬럼 보정
    if "fullVisitorId" in df.columns:
        df["fullVisitorId"] = df["fullVisitorId"].astype(str)
    else:
        raise ValueError("fullVisitorId 컬럼이 필요합니다.")

    # session_id 없으면 생성(30분 rule)
    if "session_id" not in df.columns:
        df = df.sort_values(["fullVisitorId", "event_time_naive"]).reset_index(drop=True)
        diff = df.groupby("fullVisitorId")["event_time_naive"].diff().dt.total_seconds()
        df["new_session"] = (diff.isna()) | (diff > SESSION_GAP_MIN * 60)
        df["session_num"] = df.groupby("fullVisitorId")["new_session"].cumsum().astype(int)
        df["session_id"] = df["fullVisitorId"] + "_" + df["session_num"].astype(str)

    # 결측 기본값
    for c in [
        "channelGrouping",
        "device_deviceCategory",
        "trafficSource_source",
        "trafficSource_medium",
        "trafficSource_referralPath",
        "geoNetwork_continent",
    ]:
        if c not in df.columns:
            df[c] = "Unknown"

    return df


@st.cache_data(show_spinner=False)
def build_session_table(df: pd.DataFrame) -> pd.DataFrame:
    sess = (
        df.groupby("session_id", as_index=False)
        .agg(
            fullVisitorId=("fullVisitorId", "first"),
            session_start=("event_time_naive", "min"),
            session_end=("event_time_naive", "max"),
            pv=("totals_pageviews", "sum"),
            hits=("totals_hits", "sum"),
            bounces=("totals_bounces", "max"),
            newVisit=("totals_newVisits", "max"),
            visitNumber=("visitNumber", "max"),
            revenue=("totals_transactionRevenue", "sum"),
            channel=("channelGrouping", "first"),
            device=("device_deviceCategory", "first"),
            session_duration=("session_duration", "max"),
            source=("trafficSource_source", "first"),
            medium=("trafficSource_medium", "first"),
            referral_path=("trafficSource_referralPath", "first"),
        )
    )

    for c in ["source", "medium", "referral_path"]:
        sess[c] = sess[c].fillna("Unknown")

    # 파생
    sess["session_date"] = pd.to_datetime(sess["session_start"]).dt.date
    sess["session_hour"] = pd.to_datetime(sess["session_start"]).dt.hour
    sess["first_week"] = pd.to_datetime(sess["session_start"]).dt.to_period("W")
    sess["is_transaction"] = (sess["revenue"] > 0).astype(int)

    # 30일 내 재방문 누적 카운트
    s = sess.sort_values(["fullVisitorId", "session_start"]).copy()
    first = s.groupby("fullVisitorId")["session_start"].transform("min")
    s["within_30d"] = (s["session_start"] > first) & (s["session_start"] <= first + pd.Timedelta(days=30))
    s["revisit_count_30d"] = s.groupby("fullVisitorId")["within_30d"].cumsum()
    sess = sess.merge(s[["session_id", "revisit_count_30d"]], on="session_id", how="left")
    sess["revisit_count_30d"] = sess["revisit_count_30d"].fillna(0).astype(int)

    for c in ["pv", "hits", "bounces", "visitNumber", "session_duration"]:
        sess[c] = pd.to_numeric(sess[c], errors="coerce").fillna(0)

    # 첫 방문 채널/디바이스
    first_idx = (
        sess.sort_values(["fullVisitorId", "session_start"]).groupby("fullVisitorId", as_index=False).head(1)
    )
    sess["first_channel"] = sess["fullVisitorId"].map(dict(zip(first_idx["fullVisitorId"], first_idx["channel"])))
    sess["first_device"] = sess["fullVisitorId"].map(dict(zip(first_idx["fullVisitorId"], first_idx["device"])))

    return sess


def label_user_revisit_30d(sess: pd.DataFrame) -> pd.Series:
    s = sess.copy()
    first = s.groupby("fullVisitorId")["session_start"].transform("min")
    s["within_30d"] = (s["session_start"] > first) & (s["session_start"] <= first + pd.Timedelta(days=30))
    return s.groupby("fullVisitorId")["within_30d"].any().rename("revisit_30d")


def moving_avg(series: pd.Series, k: int = 7) -> pd.Series:
    return series.rolling(k, min_periods=1).mean()


# ----- 실제 로딩 -----
df = load_df_from_file(uploaded)
sess = build_session_table(df)

# =========================
# 3) 사이드바 필터
# =========================
min_d = sess["session_date"].min()
max_d = sess["session_date"].max()

with st.sidebar:
    st.markdown("---")
    st.header("필터")
    start_d = st.date_input("시작일", min_d, min_value=min_d, max_value=max_d)
    end_d = st.date_input("종료일", max_d, min_value=min_d, max_value=max_d)

    channels = sorted(sess["channel"].dropna().unique().tolist())
    devices = sorted(sess["device"].dropna().unique().tolist())
    sel_channels = st.multiselect("채널", channels, default=channels)
    sel_devices = st.multiselect("디바이스", devices, default=devices)

    st.subheader("Activation 기준(고정)")
    min_day_sessions = st.number_input("일별 표본 최소 세션수(스파이크 가드)", 0, value=500, step=50)

mask = (
    (sess["session_date"] >= start_d)
    & (sess["session_date"] <= end_d)
    & (sess["channel"].isin(sel_channels))
    & (sess["device"].isin(sel_devices))
)
sf = sess.loc[mask].copy()

# =========================
# 4) Activation 플래그 계산(고정 기준)
# =========================
if "revisit_count_30d" not in sf.columns:
    sf = sf.merge(sess[["session_id", "revisit_count_30d"]], on="session_id", how="left")
sf["revisit_count_30d"] = sf["revisit_count_30d"].fillna(0).astype(int)

sf["act_pageviews"] = (sf["pv"] >= 3).astype(int)
sf["act_duration"] = (sf["session_duration"] >= 180).astype(int)
sf["act_nonbounce"] = (sf["bounces"] == 0).astype(int)
sf["act_hits"] = (sf["hits"] >= 10).astype(int)
sf["act_revisit"] = (sf["revisit_count_30d"] >= 2).astype(int)
sf["activation"] = (
    (sf["act_pageviews"] == 1)
    & (sf["act_duration"] == 1)
    & (sf["act_nonbounce"] == 1)
    & (sf["act_hits"] == 1)
    & (sf["act_revisit"] == 1)
).astype(int)

# =========================
# 5) 상단 KPI
# =========================
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.metric("사용자 수", f"{sf['fullVisitorId'].nunique():,}")
with k2:
    st.metric("세션 수", f"{sf['session_id'].nunique():,}")
with k3:
    st.metric("평균 PV/세션", f"{sf['pv'].mean():.2f}")
with k4:
    st.metric("Bounce Rate", f"{sf['bounces'].mean():.2%}")
with k5:
    st.metric("Median Duration(s)", f"{sf['session_duration'].median():.0f}")
with k6:
    st.metric("Activation 성공비율", f"{sf['activation'].mean():.2%}")

st.caption("Activation 기준: PV≥3 · 지속시간≥180초 · Hits≥10 · Bounce=0 · 30일 내 재방문≥2회")

# =========================
# 6) 탭
# =========================
tabR, tabA, tabRef, tabRev, tabAcq = st.tabs(["Retention", "Activation", "Referral", "Revenue", "Acquisition"])

# -------- Retention
with tabR:
    st.subheader("1) 사용자 단위 Retention (30일 이내 재방문)")
    user_revisit = label_user_revisit_30d(sf)
    st.metric("30일 재방문율(사용자 기준)", f"{user_revisit.mean():.2%}")
    st.write("• 각 사용자의 첫 세션 이후 30일 안에 한 번이라도 재방문하면 성공으로 간주")

    # 첫 방문 채널/디바이스별
    first_session = (
        sf.sort_values(["fullVisitorId", "session_start"])
        .groupby("fullVisitorId", as_index=False)
        .head(1)[["fullVisitorId", "channel", "device", "session_start"]]
    )
    user_df = user_revisit.reset_index()
    user_df["first_channel"] = user_df["fullVisitorId"].map(dict(zip(first_session["fullVisitorId"], first_session["channel"])))
    user_df["first_device"] = user_df["fullVisitorId"].map(dict(zip(first_session["fullVisitorId"], first_session["device"])))

    c1, c2 = st.columns(2)
    with c1:
        ch_tbl = user_df.groupby("first_channel")["revisit_30d"].mean().sort_values(ascending=False).reset_index()
        st.dataframe(ch_tbl)
        st.bar_chart(ch_tbl.set_index("first_channel")["revisit_30d"])
    with c2:
        dev_tbl = user_df.groupby("first_device")["revisit_30d"].mean().sort_values(ascending=False).reset_index()
        st.dataframe(dev_tbl)
        st.bar_chart(dev_tbl.set_index("first_device")["revisit_30d"])

    # 채널 × PV구간 전환율 히트맵
    pv_bins = [0, 1, 3, 5, 10, 20, 50, 100, np.inf]
    pv_cut = pd.cut(sf["pv"], bins=pv_bins, right=False)
    sf["pv_bin"] = pv_cut
    conv_flag = sf["is_transaction"].astype(int) if "is_transaction" in sf.columns else (sf["revenue"] > 0).astype(int)
    conv_tbl = (
        sf.groupby(["channel", "pv_bin"], observed=False)
        .agg(sessions=("session_id", "count"), conversions=(conv_flag.name if hasattr(conv_flag, "name") else "_", "sum"))
        .reset_index()
    )
    conv_tbl["conversions"] = sf.groupby(["channel", "pv_bin"], observed=False)[conv_flag].sum().values
    conv_tbl["conv_rate"] = np.where(conv_tbl["sessions"] > 0, conv_tbl["conversions"] / conv_tbl["sessions"], np.nan)
    conv_tbl["pv_bin"] = conv_tbl["pv_bin"].astype(str)

    heat = (
        alt.Chart(conv_tbl)
        .mark_rect()
        .encode(
            x=alt.X("pv_bin:N", title="PV 구간"),
            y=alt.Y("channel:N", title="채널"),
            color=alt.Color("conv_rate:Q", title="전환율"),
            tooltip=["channel", "pv_bin", alt.Tooltip("sessions:Q", format=",.0f"), alt.Tooltip("conv_rate:Q", format=".2%")],
        )
        .properties(height=320)
    )
    st.altair_chart(heat, use_container_width=True)

# -------- Activation
with tabA:
    st.markdown(
        """
**✅ Activation 성공 기준**  
1. PV ≥ 3  
2. 지속시간 ≥ 180초  
3. Bounce = 0  
4. Hits ≥ 10  
5. 30일 내 재방문 ≥ 2회
"""
    )

    daily = sf.groupby("session_date", as_index=False).agg(activation=("activation", "mean"), sessions=("session_id", "count"))
    daily["activation_ma7"] = moving_avg(daily["activation"], 7)
    daily["low_sample"] = daily["sessions"] < min_day_sessions

    base = alt.Chart(daily).encode(x="session_date:T")
    bars = base.mark_bar(opacity=0.3).encode(y=alt.Y("sessions:Q", axis=alt.Axis(title="Sessions")))
    line = base.mark_line().encode(y=alt.Y("activation:Q", axis=alt.Axis(title="Activation Rate", format="~%")))
    ma7 = base.mark_line(strokeDash=[4, 4], color="gray").encode(y="activation_ma7:Q")
    pts = base.mark_circle(size=20).encode(
        y="activation:Q",
        color=alt.condition("datum.low_sample", alt.value("#aaaaaa"), alt.value("#1f77b4")),
    )
    st.altair_chart(alt.layer(bars, line, ma7, pts).resolve_scale(y="independent").properties(height=320), use_container_width=True)
    st.caption(f"점 색상: 일 세션수 < {min_day_sessions} (회색)")

    # 심층 지표(구간별)
    c1, c2, c3 = st.columns(3)
    with c1:
        pv_bins2 = [0, 1, 3, 5, 10, 20, 50, 100, np.inf]
        pv_cut2 = pd.cut(sf["pv"], pv_bins2, right=False)
        pv_tbl = sf.groupby(pv_cut2, observed=False)["activation"].mean().reset_index()
        pv_tbl["pv_bin"] = pv_tbl.iloc[:, 0].astype(str)
        st.bar_chart(pv_tbl.set_index("pv_bin")["activation"])
    with c2:
        dur_bins = [0, 30, 60, 120, 180, 300, 600, 1200, 1800, 3600, np.inf]
        dur_cut = pd.cut(sf["session_duration"], dur_bins, right=False)
        dur_tbl = sf.groupby(dur_cut, observed=False)["activation"].mean().reset_index()
        dur_tbl["dur_bin"] = dur_tbl.iloc[:, 0].astype(str)
        st.bar_chart(dur_tbl.set_index("dur_bin")["activation"])
    with c3:
        hit_bins = [0, 5, 10, 20, 50, 100, 200, np.inf]
        hit_cut = pd.cut(sf["hits"], hit_bins, right=False)
        hit_tbl = sf.groupby(hit_cut, observed=False)["activation"].mean().reset_index()
        hit_tbl["hit_bin"] = hit_tbl.iloc[:, 0].astype(str)
        st.bar_chart(hit_tbl.set_index("hit_bin")["activation"])

# -------- Referral
with tabRef:
    st.subheader("Referral 분석")
    ref = sf[(sf["medium"].str.lower() == "referral") | (sf["channel"] == "Referral")].copy()
    if ref.empty:
        st.warning("Referral 세션이 없습니다. 사이드바 필터를 확인해 주세요.")
    else:
        top_sessions = (
            ref.groupby("source", observed=False)["session_id"].count().reset_index(name="sessions").sort_values("sessions", ascending=False).head(10)
        )
        st.markdown("### Top 10 Referral Sources (세션 수)")
        ch_sessions = (
            alt.Chart(top_sessions)
            .mark_bar()
            .encode(x=alt.X("sessions:Q", title="Number of Sessions"), y=alt.Y("source:N", sort="-x", title="Source"), tooltip=["source", "sessions"])
            .properties(height=380)
        )
        st.altair_chart(ch_sessions, use_container_width=True)

        ref_top = ref.merge(top_sessions, on="source", how="inner")
        by_source = (
            ref_top.groupby("source", observed=False)
            .agg(avg_pv=("pv", "mean"), avg_hits=("hits", "mean"), bounce_rate=("bounces", "mean"), sessions=("session_id", "count"))
            .reset_index()
            .sort_values("sessions", ascending=False)
            .head(10)
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            chart = alt.Chart(by_source).mark_bar().encode(
                x=alt.X("avg_pv:Q", title="Average Pageviews"),
                y=alt.Y("source:N", sort="-x"),
                tooltip=["source", alt.Tooltip("avg_pv:Q", format=".2f")],
            )
            st.altair_chart(chart.properties(height=420), use_container_width=True)
        with c2:
            chart = alt.Chart(by_source).mark_bar().encode(
                x=alt.X("avg_hits:Q", title="Average Hits"),
                y=alt.Y("source:N", sort="-x"),
                tooltip=["source", alt.Tooltip("avg_hits:Q", format=".2f")],
            )
            st.altair_chart(chart.properties(height=420), use_container_width=True)
        with c3:
            chart = (
                alt.Chart(by_source.assign(br_pct=by_source["bounce_rate"] * 100))
                .mark_bar()
                .encode(
                    x=alt.X("br_pct:Q", title="Bounce Rate (%)"),
                    y=alt.Y("source:N", sort="-x"),
                    tooltip=["source", alt.Tooltip("br_pct:Q", format=".1f")],
                )
            )
            st.altair_chart(chart.properties(height=420), use_container_width=True)

        st.markdown("#### Raw Table")
        show_cols = ["source", "sessions", "avg_pv", "avg_hits", "bounce_rate"]
        st.dataframe(by_source[show_cols].rename(columns={"avg_pv": "avg_pageviews", "bounce_rate": "bounce_rate(0-1)"}))

# -------- Revenue (간단한 Altair/옵션 Matplotlib)
with tabRev:
    st.subheader("🌍 Distribution of Continent")
    df_f = df[df["session_id"].isin(sf["session_id"])].copy()
    cont_tbl = df_f["geoNetwork_continent"].fillna("Unknown").value_counts().rename_axis("continent").reset_index(name="sessions")
    chart_cont = alt.Chart(cont_tbl).mark_bar().encode(
        x=alt.X("sessions:Q", title="Sessions"),
        y=alt.Y("continent:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("continent:N", title="Continent"), alt.Tooltip("sessions:Q", title="Sessions", format=",.0f")],
    )
    st.altair_chart(chart_cont.properties(height=260), use_container_width=True)

    st.subheader("📱 Mobile vs. Desktop Traffic Share")
    dev_cnt = sf["device"].str.lower().value_counts()
    obs_mobile = int(dev_cnt.get("mobile", 0))
    obs_desktop = int(dev_cnt.get("desktop", 0))
    total = max(obs_mobile + obs_desktop, 1)
    obs_tbl = pd.DataFrame(
        {"type": ["Observed Mobile", "Observed Desktop"], "share": [obs_mobile / total, obs_desktop / total]}
    )

    chart = alt.Chart(obs_tbl).mark_bar().encode(x=alt.X("type:N", title=None), y=alt.Y("share:Q", axis=alt.Axis(format="~%")), tooltip=["type", alt.Tooltip("share:Q", format=".1%")])
    st.altair_chart(chart.properties(height=260), use_container_width=True)

# -------- Acquisition (간단 구성, Matplotlib 설치 시 보강)
with tabAcq:
    st.subheader("채널별 세션 & 신규 비율")
    channel_summary = (
        df.groupby("channelGrouping").agg(sessions=("session_id", "nunique"), new_sessions=("totals_newVisits", "sum")).reset_index()
    )
    channel_summary["new_ratio"] = np.where(channel_summary["sessions"] > 0, channel_summary["new_sessions"] / channel_summary["sessions"], 0.0)

    ch1 = alt.Chart(channel_summary).mark_bar().encode(x="channelGrouping:N", y=alt.Y("sessions:Q", title="Sessions"), tooltip=["channelGrouping", "sessions"])
    ch2 = alt.Chart(channel_summary).mark_line(point=True).encode(x="channelGrouping:N", y=alt.Y("new_ratio:Q", axis=alt.Axis(format="~%"), title="New Visit Ratio"))
    st.altair_chart(alt.layer(ch1, ch2).resolve_scale(y="independent").properties(height=320), use_container_width=True)

st.success("완료! 사이드바에서 기간/채널/디바이스/Activation 기준을 바꿔가며 Retention~Acquisition을 탐색하세요.")
