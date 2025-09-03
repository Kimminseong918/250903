import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from datetime import timedelta


# =========================
# 0) 기본 설정
# =========================
#CSV_PATH = r"C:\Users\MYeongs\PycharmProjects\Google\data_3.csv"  # 필요 시 변경
CSV_PATH = './data_3.csv'

SESSION_GAP_MIN = 30  # (데이터에 session_id가 없을 때만 사용)

st.set_page_config(page_title="RARRA Dashboard", layout="wide")
st.title("📊 RARRA Dashboard (Retention • Activation • Referral • Revenue • Acquisition)")
st.caption("Kaggle GA Customer Revenue Dataset | 세션 단위 분석")

# =========================
# 1) 데이터 로딩 & 세션 테이블 구성
# =========================
@st.cache_data(show_spinner=True)
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, low_memory=False)

    # 시간 처리
    if "event_time" not in df.columns:
        raise ValueError("event_time 컬럼이 필요합니다.")
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df["event_time_naive"] = df["event_time"].dt.tz_convert("UTC").dt.tz_localize(None)

    # # 숫자형 변환
    # def to_num(col, default=0):
    #     if col in df.columns:
    #         df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    # for c in ["totals_pageviews","totals_hits","totals_bounces","totals_newVisits",
    #           "totals_transactionRevenue","visitNumber","session_duration"]:
    #     to_num(c, 0)

    if "fullVisitorId" in df.columns:
        df["fullVisitorId"] = df["fullVisitorId"].astype(str)

    def to_num(df, cols):
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df

    # 사용 예시
    df = to_num(df, [
        "totals_pageviews", 
        "totals_hits", 
        "totals_bounces", 
        "totals_newVisits", 
        "totals_transactionRevenue", 
        "session_duration",
        "is_transaction",
        "visitNumber"
    ])

    # session_id가 이미 있으면 사용. 없으면 보수적으로 생성(30분 기준)
    if "session_id" not in df.columns:
        df = df.sort_values(["fullVisitorId","event_time_naive"]).reset_index(drop=True)
        diff = df.groupby("fullVisitorId")["event_time_naive"].diff().dt.total_seconds()
        df["new_session"] = (diff.isna()) | (diff > SESSION_GAP_MIN*60)
        df["session_num"] = df.groupby("fullVisitorId")["new_session"].cumsum().astype(int)
        df["session_id"] = df["fullVisitorId"] + "_" + df["session_num"].astype(str)

    # 결측 보정
    for c in ["channelGrouping","device_deviceCategory"]:
        if c not in df.columns:
            df[c] = "Unknown"

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
                  # 👇 Referral 분석용
                  source=("trafficSource_source","first"),
                  medium=("trafficSource_medium","first"),
                  referral_path=("trafficSource_referralPath","first"),
              ))

    def compute_revisit_count_30d(sess: pd.DataFrame) -> pd.DataFrame:
        """
        사용자 최초 세션 기준 30일 내 '재방문' 누적 카운트(세션별)를 계산.
        세션별로 현재 시점까지 30일 창 안에서 몇 번째 재방문인지 나타냄.
        """
        s = sess.sort_values(["fullVisitorId", "session_start"]).copy()
        first = s.groupby("fullVisitorId")["session_start"].transform("min")
        s["within_30d"] = (s["session_start"] > first) & (s["session_start"] <= first + pd.Timedelta(days=30))
        s["revisit_count_30d"] = s.groupby("fullVisitorId")["within_30d"].cumsum()
        return s[["session_id", "revisit_count_30d"]]

    # ▼ 바로 여기서 기본값 보정
    sess["source"] = sess["source"].fillna("Unknown")
    sess["medium"] = sess["medium"].fillna("Unknown")
    sess["referral_path"] = sess["referral_path"].fillna("Unknown")  # (권장)

    # 파생/정리
    sess["session_date"] = pd.to_datetime(sess["session_start"]).dt.date
    sess["session_hour"] = pd.to_datetime(sess["session_start"]).dt.hour
    sess["first_week"] = pd.to_datetime(sess["session_start"]).dt.to_period("W")
    sess["is_transaction"] = (sess["revenue"] > 0).astype(int)

    # 🔧 30일 내 재방문 누적 카운트 합치기
    revisit = compute_revisit_count_30d(sess)
    sess = sess.merge(revisit, on="session_id", how="left")
    sess["revisit_count_30d"] = sess["revisit_count_30d"].fillna(0).astype(int)

    return sess

    for c in ["pv","hits","bounces","visitNumber","session_duration"]:
        sess[c] = sess[c].fillna(0)

    # 첫 방문 채널/디바이스 라벨
    first_idx = sess.sort_values(["fullVisitorId","session_start"]) \
                    .groupby("fullVisitorId", as_index=False).head(1)
    first_map_channel = dict(zip(first_idx["fullVisitorId"], first_idx["channel"]))
    first_map_device  = dict(zip(first_idx["fullVisitorId"], first_idx["device"]))
    sess["first_channel"] = sess["fullVisitorId"].map(first_map_channel)
    sess["first_device"]  = sess["fullVisitorId"].map(first_map_device)
    return sess

# 30일 이내 재방문 여부(사용자 단위)
def label_user_revisit_30d(sess: pd.DataFrame) -> pd.DataFrame:
    s = sess.copy()
    first = s.groupby("fullVisitorId")["session_start"].transform("min")
    s["first_start"] = first
    s["within_30d"] = (s["session_start"] > first) & \
                      (s["session_start"] <= first + pd.Timedelta(days=30))
    user_has_revisit = s.groupby("fullVisitorId")["within_30d"].any().rename("revisit_30d")
    return user_has_revisit

# 7일 이동평균
def moving_avg(series: pd.Series, k: int = 7) -> pd.Series:
    return series.rolling(k, min_periods=1, center=False).mean()

# =========================
# 2) 사이드바
# =========================
with st.sidebar:
    st.header("데이터")
    path = st.text_input("CSV 경로", value=CSV_PATH)
    load_btn = st.button("🔄 로드/새로고침")

df = load_df(path)
sess = build_session_table(df)

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

    st.subheader("Activation 기준")
    min_pv   = st.number_input("최소 PV (≥)", min_value=0, value=3, step=1)
    min_dur  = st.number_input("최소 지속시간(초, ≥)", min_value=0, value=180, step=30)
    min_hits = st.number_input("최소 Hits (≥)", min_value=0, value=10, step=1)
    req_nonbounce = st.checkbox("Bounce=0 조건 포함", value=True)
    req_revisit   = st.checkbox("재방문(visitNumber>1) 포함", value=False)
    min_day_sessions = st.number_input("일별 표본 최소 세션수(스파이크 가드레일)", 0, value=500, step=50)

    if load_btn:
        st.experimental_rerun()

# 필터 적용
mask = (
    (sess["session_date"] >= start_d) &
    (sess["session_date"] <= end_d) &
    (sess["channel"].isin(sel_channels)) &
    (sess["device"].isin(sel_devices))
)
sf = sess.loc[mask].copy()

# --- Activation 플래그 계산 (세션 단위) ---

# 0) revisit_count_30d 없으면 sess에서 조인해서 채우기
if "revisit_count_30d" not in sf.columns:
    sf = sf.merge(
        sess[["session_id", "revisit_count_30d"]],
        on="session_id", how="left"
    )
sf["revisit_count_30d"] = sf["revisit_count_30d"].fillna(0).astype(int)

# 1) 각 조건별 플래그 생성 (요구사항 고정값 반영)
sf["act_pageviews"] = (sf["pv"] >= 3).astype(int)
sf["act_duration"]  = (sf["session_duration"] >= 180).astype(int)  # 3분
sf["act_nonbounce"] = (sf["bounces"] == 0).astype(int)
sf["act_hits"]      = (sf["hits"] >= 10).astype(int)
sf["act_revisit"]   = (sf["revisit_count_30d"] >= 2).astype(int)

# 2) 최종 Activation 조건 (모두 충족해야 1)
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

# ↘️ 캡션은 고정 기준으로 명확히 표기 (사이드바 값/체크박스와 혼동 제거)
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
    retention_30d = user_revisit.mean()
    st.metric("30일 재방문율(사용자 기준)", f"{retention_30d:.2%}")

    st.write("• 정의: 각 사용자의 첫 세션 이후 30일 안에 **한 번이라도** 재방문하면 성공으로 간주")

    # 1-1. 첫 방문 채널/디바이스별 Retention
    first_session = sf.sort_values(["fullVisitorId","session_start"]) \
                      .groupby("fullVisitorId", as_index=False).head(1)[["fullVisitorId","channel","device","session_start"]]
    first_map_ch = dict(zip(first_session["fullVisitorId"], first_session["channel"]))
    first_map_dev = dict(zip(first_session["fullVisitorId"], first_session["device"]))
    user_df = user_revisit.reset_index()
    user_df["first_channel"] = user_df["fullVisitorId"].map(first_map_ch)
    user_df["first_device"]  = user_df["fullVisitorId"].map(first_map_dev)

    c1,c2 = st.columns(2)
    with c1:
        st.write("2) **첫 방문 채널별** 30일 재방문율")
        ch_tbl = user_df.groupby("first_channel")["revisit_30d"].mean().sort_values(ascending=False).reset_index()
        st.dataframe(ch_tbl)
        st.bar_chart(ch_tbl.set_index("first_channel")["revisit_30d"])
    with c2:
        st.write("3) **첫 방문 디바이스별** 30일 재방문율")
        dev_tbl = user_df.groupby("first_device")["revisit_30d"].mean().sort_values(ascending=False).reset_index()
        st.dataframe(dev_tbl)
        st.bar_chart(dev_tbl.set_index("first_device")["revisit_30d"])

    st.subheader("4) 고객 충성도/참여도")
    u_eng = (sf.groupby("fullVisitorId")
               .agg(sessions=("session_id","nunique"),
                    pv_mean=("pv","mean"),
                    dur_mean=("session_duration","mean"),
                    bounce_rate=("bounces","mean")))
    c3,c4,c5 = st.columns(3)
    with c3: st.metric("사용자당 중위 세션수", f"{u_eng['sessions'].median():.1f}")
    with c4: st.metric("사용자당 평균 PV", f"{u_eng['pv_mean'].mean():.2f}")
    with c5: st.metric("사용자당 평균 Bounce Rate", f"{u_eng['bounce_rate'].mean():.2%}")
    st.dataframe(u_eng.head(20))

    st.subheader("5) 채널 × PV구간별 구매 전환율")

    # ① PV 구간 정의 및 라벨 순서 만들기
    pv_bins = [0, 1, 3, 5, 10, 20, 50, 100, np.inf]
    pv_cut = pd.cut(sf["pv"], bins=pv_bins, right=False)  # Interval형
    pv_order = pv_cut.cat.categories.astype(str).tolist()  # 축 정렬용
    sf["pv_bin"] = pv_cut

    # ② 전환 플래그(있으면 is_transaction, 없으면 revenue>0)
    if "is_transaction" in sf.columns:
        conv_flag = sf["is_transaction"].astype(int)
    else:
        conv_flag = (sf["revenue"] > 0).astype(int)
    sf["_conv_flag"] = conv_flag

    # ③ 채널 × PV구간 집계
    conv_tbl = (sf.groupby(["channel", "pv_bin"], observed=False)
                .agg(sessions=("session_id", "count"),
                     conversions=("_conv_flag", "sum"))
                .reset_index())
    conv_tbl["conv_rate"] = np.where(
        conv_tbl["sessions"] > 0,
        conv_tbl["conversions"] / conv_tbl["sessions"],
        np.nan
    )
    conv_tbl["pv_bin"] = conv_tbl["pv_bin"].astype(str)

    # ④ 히트맵 (채널 × PV구간 → 전환율 색상)
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

    # 임시 컬럼 정리(선택)
    sf.drop(columns=["_conv_flag"], inplace=True, errors="ignore")

# -------------------------
# Tab 2. Activation
# -------------------------
with tabA:
    st.markdown("""
    **✅ Activation 성공 기준**  
    1. 페이지뷰(PV) ≥ 3  
    2. 세션 지속시간 ≥ 180초 (3분)  
    3. Bounce = 0 (이탈하지 않음)  
    4. Hits ≥ 10  
    5. 30일 내 재방문 횟수 ≥ 2회
    """)
    st.subheader("1) 세션단위 Activation 성공 비율 (일별)")

    daily = (sf.groupby("session_date", as_index=False)
               .agg(activation=("activation","mean"), sessions=("session_id","count")))
    daily["activation_ma7"] = moving_avg(daily["activation"], 7)
    daily["low_sample"] = daily["sessions"] < min_day_sessions

    base = alt.Chart(daily).encode(x="session_date:T")

    bars = base.mark_bar(opacity=0.3).encode(y=alt.Y("sessions:Q", axis=alt.Axis(title="Sessions")))
    line = base.mark_line().encode(y=alt.Y("activation:Q", axis=alt.Axis(title="Activation Rate", format="~%")))
    ma7  = base.mark_line(strokeDash=[4,4]).encode(y="activation_ma7:Q", color=alt.value("gray"))
    pts  = base.mark_circle(size=20).encode(y="activation:Q", color=alt.condition("datum.low_sample", alt.value("#aaaaaa"), alt.value("#1f77b4")))

    st.altair_chart(alt.layer(bars, line, ma7, pts).resolve_scale(y='independent').properties(height=320, width="container"), use_container_width=True)
    st.caption(f"점 색상: 일 세션수 < {min_day_sessions} (회색)")

    st.subheader("2) 채널별 Activation 성공 비율")
    ch_act = (sf.groupby("channel", observed=False)
                .agg(sessions=("session_id","count"),
                     act_rate=("activation","mean"))
                .sort_values("act_rate", ascending=False)).reset_index()
    st.dataframe(ch_act)
    st.bar_chart(ch_act.set_index("channel")["act_rate"])

    st.subheader("3) 디바이스별 Activation 성공 비율")
    dev_act = (sf.groupby("device", observed=False)
                 .agg(sessions=("session_id","count"),
                      act_rate=("activation","mean"))
                 .sort_values("act_rate", ascending=False)).reset_index()
    st.dataframe(dev_act)
    st.bar_chart(dev_act.set_index("device")["act_rate"])

    st.subheader("4) 주차별 신규 유저 Activation 달성률")
    # 첫 방문 주차 기준 코호트
    first_week_user = sf.sort_values(["fullVisitorId","session_start"]).groupby("fullVisitorId", as_index=False).head(1)
    fw_map = dict(zip(first_week_user["fullVisitorId"], pd.to_datetime(first_week_user["session_start"]).dt.to_period("W")))
    sf["first_week"] = sf["fullVisitorId"].map(fw_map)
    cohort_rate = sf.groupby("first_week", observed=False)["activation"].mean().reset_index()
    cohort_rate["first_week"] = cohort_rate["first_week"].astype(str)
    st.line_chart(cohort_rate.set_index("first_week"))

    st.subheader("5) Activation 심층 지표")
    c1,c2,c3 = st.columns(3)
    with c1:
        pv_bins2 = [0,1,3,5,10,20,50,100,np.inf]
        pv_cut = pd.cut(sf["pv"], pv_bins2, right=False)
        pv_act = sf.groupby(pv_cut, observed=False)["activation"].mean().reset_index()
        pv_act["pv_bin"] = pv_act.iloc[:,0].astype(str); pv_act["rate"] = pv_act["activation"]
        st.bar_chart(pv_act.set_index("pv_bin")["rate"])
    with c2:
        dur_bins = [0,30,60,120,180,300,600,1200,1800,3600,np.inf]
        dur_cut = pd.cut(sf["session_duration"], dur_bins, right=False)
        dur_act = sf.groupby(dur_cut, observed=False)["activation"].mean().reset_index()
        dur_act["dur_bin"] = dur_act.iloc[:,0].astype(str); dur_act["rate"] = dur_act["activation"]
        st.bar_chart(dur_act.set_index("dur_bin")["rate"])
    with c3:
        hit_bins = [0,5,10,20,50,100,200,np.inf]
        hit_cut = pd.cut(sf["hits"], hit_bins, right=False)
        hit_act = sf.groupby(hit_cut, observed=False)["activation"].mean().reset_index()
        hit_act["hit_bin"] = hit_act.iloc[:,0].astype(str); hit_act["rate"] = hit_act["activation"]
        st.bar_chart(hit_act.set_index("hit_bin")["rate"])

    st.subheader("6) 퍼널 관점: PV 구간별 세션 수 & Activation")
    funnel = (sf.groupby(pv_cut, observed=False)
                .agg(sessions=("session_id","count"),
                     act_sessions=("activation","sum"))
                .reset_index())
    funnel = funnel.rename(columns={funnel.columns[0]:"pv_bin"})
    funnel["pv_bin"] = funnel["pv_bin"].astype(str)
    funnel["activation_rate"] = np.where(funnel["sessions"]>0,
                                         funnel["act_sessions"]/funnel["sessions"], np.nan)
    st.dataframe(funnel)

# -------------------------
# Tab 3. Referral
# -------------------------
# 예) 탭 선언 (이름은 상황에 맞게)
# tab1, tab2, tab3, tab4 = st.tabs(["Retention", "Activation", "Referral", "Revenue"])

with tabRef:
    st.subheader("Referral 분석")

    # 1) Referral 세션만 필터 (medium=referral 또는 channel=Referral)
    ref = sf[
        (sf["medium"].str.lower() == "referral") | (sf["channel"] == "Referral")
    ].copy()

    if ref.empty:
        st.warning("Referral 세션이 없습니다. 사이드바 필터를 확인해 주세요.")
        st.stop()

    # 2) 세션 수 기준 Top 10 소스
    top_sessions = (
        ref.groupby("source", observed=False)["session_id"]
           .count()
           .reset_index(name="sessions")
           .sort_values("sessions", ascending=False)
           .head(10)
    )
    st.markdown("### Top 10 Referral Sources (세션 수)")
    ch_sessions = (
        alt.Chart(top_sessions)
           .mark_bar()
           .encode(
               x=alt.X("sessions:Q", title="Number of Sessions"),
               y=alt.Y("source:N", sort="-x", title="Source"),
               tooltip=["source","sessions"]
           )
           .properties(height=380)
    )
    st.altair_chart(ch_sessions, use_container_width=True)

    st.markdown("---")

    # 3) 평균 PV / 평균 Hits / Bounce Rate (Top10 기준 비교)
    ref_top = ref.merge(top_sessions, on="source", how="inner")
    by_source = (ref_top.groupby("source", observed=False)
                        .agg(avg_pv=("pv","mean"),
                             avg_hits=("hits","mean"),
                             bounce_rate=("bounces","mean"),
                             sessions=("session_id","count"))
                        .reset_index()
                        .sort_values("sessions", ascending=False)
                        .head(10))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Avg Pageviews by Referral Source (Top 10)")
        ch_pv = (
            alt.Chart(by_source)
               .mark_bar()
               .encode(
                   x=alt.X("avg_pv:Q", title="Average Pageviews"),
                   y=alt.Y("source:N", sort="-x", title="Source"),
                   tooltip=["source", alt.Tooltip("avg_pv:Q", format=".2f")]
               )
               .properties(height=420)
        )
        st.altair_chart(ch_pv, use_container_width=True)

    with col2:
        st.markdown("#### Avg Hits by Referral Source (Top 10)")
        ch_hits = (
            alt.Chart(by_source)
               .mark_bar()
               .encode(
                   x=alt.X("avg_hits:Q", title="Average Hits"),
                   y=alt.Y("source:N", sort="-x", title="Source"),
                   tooltip=["source", alt.Tooltip("avg_hits:Q", format=".2f")]
               )
               .properties(height=420)
        )
        st.altair_chart(ch_hits, use_container_width=True)

    with col3:
        st.markdown("#### Bounce Rate by Referral Source (Top 10)")
        ch_br = (
            alt.Chart(by_source.assign(br_pct=by_source["bounce_rate"]*100))
               .mark_bar()
               .encode(
                   x=alt.X("br_pct:Q", title="Bounce Rate (%)"),
                   y=alt.Y("source:N", sort="-x", title="Source"),
                   tooltip=["source", alt.Tooltip("br_pct:Q", format=".1f")]
               )
               .properties(height=420)
        )
        st.altair_chart(ch_br, use_container_width=True)

    # (선택) 표로도 확인
    st.markdown("#### Raw Table")
    show_cols = ["source","sessions","avg_pv","avg_hits","bounce_rate"]
    st.dataframe(by_source[show_cols].rename(columns={
        "avg_pv":"avg_pageviews", "bounce_rate":"bounce_rate(0-1)"
    }))



# -------------------------
# Tab 4. Revenue
# -------------------------
with tabRev:
    # ------------------ (1) Distribution of Continent ------------------
    st.subheader("🌍 Distribution of Continent")

    df_f = df[df["session_id"].isin(sf["session_id"])].copy()
    cont_tbl = (df_f["geoNetwork_continent"]
                .fillna("Unknown")
                .value_counts()
                .rename_axis("continent")
                .reset_index(name="sessions"))

    chart_cont = alt.Chart(cont_tbl).mark_bar().encode(
        x=alt.X("sessions:Q", title="Sessions"),
        y=alt.Y("continent:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("continent:N", title="Continent"),
                 alt.Tooltip("sessions:Q", title="Sessions", format=",.0f")]
    ).properties(width=420, height=260)
    st.altair_chart(chart_cont, use_container_width=False)

    # ------------------ (2) Mobile vs. Desktop Traffic Share ------------------
    st.subheader("📱 Mobile vs. Desktop Traffic Share")

    dev_cnt = sf["device"].value_counts()
    obs_mobile = int(dev_cnt.get("mobile", 0))
    obs_desktop = int(dev_cnt.get("desktop", 0))
    obs_total_md = max(obs_mobile + obs_desktop, 1)
    obs_mobile_share = obs_mobile / obs_total_md * 100.0
    obs_desktop_share = obs_desktop / obs_total_md * 100.0

    market_mobile_share = 64.3
    market_desktop_share = 35.7

    fig, ax = plt.subplots(figsize=(5.2, 3.6))  # 🔽 크기 줄임
    x = [0, 1]
    labels = ["2016–2017 (Observed)", "2025 (Market)"]

    ax.bar(x[0], obs_mobile_share, color="#8ecae6", label="Mobile")
    ax.bar(x[0], obs_desktop_share, bottom=obs_mobile_share, color="#f4a261", label="Desktop")
    ax.bar(x[1], market_mobile_share, color="#8ecae6")
    ax.bar(x[1], market_desktop_share, bottom=market_mobile_share, color="#f4a261")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Traffic Share (%)")

    # 퍼센트 라벨
    ax.text(x[0], obs_mobile_share / 2, f"{obs_mobile_share:.1f}%", ha="center", va="center", fontsize=10,
            weight="bold")
    ax.text(x[0], obs_mobile_share + obs_desktop_share / 2, f"{obs_desktop_share:.1f}%", ha="center", va="center",
            fontsize=10, weight="bold")
    ax.text(x[1], market_mobile_share / 2, f"{market_mobile_share:.1f}%", ha="center", va="center", fontsize=10,
            weight="bold")
    ax.text(x[1], market_mobile_share + market_desktop_share / 2, f"{market_desktop_share:.1f}%", ha="center",
            va="center", fontsize=10, weight="bold")

    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.20), frameon=False)
    st.pyplot(fig, use_container_width=False)

    # 🔽 이미지 바로 아래 설명 추가
    st.markdown(
        """
        **2025년: 모바일 64.3%, 데스크톱 35.7% → 모바일 중심 구조**  
        ➡️ 이 시각화는 **“모바일 믹스 결핍”**이 얼마나 큰 격차였는지 명확히 보여줍니다.  
        따라서 향후 **모바일 우선 전략(Mobile-First Strategy)**이 반드시 필요함을 강조합니다.
        """
    )

    # ------------------ 설명 텍스트 추가 ------------------
    st.markdown(
        """
**발견 사항**  
- 데이터셋에 기록된 거래 수익이 없어 **직접적인 수익 분석 및 예측은 불가능**했습니다.  
- 그러나 **높은 이탈률과 낮은 참여도**는 잠재적 **수익 기회 손실**을 의미합니다.

**전략 시사점**  
- **사용자 참여 증진(이탈률 감소, 페이지뷰 증가)** 은 전환율 및 잠재적 수익 증대를 위한 **필수 전제 조건**입니다.  
- 분석에서 도출된 **참여도 개선 방안(UX/UI 개선, 타겟팅 최적화 등)** 실행에 **집중**하여 수익 기반을 마련하세요.

**추가 관찰**  
- *Date/Number of Sessions Over Time*와 **신규 유입 vs 이탈** 흐름을 비교하면, **특정 이벤트 시점에 트래픽 급증**이 관찰됩니다.  
  시즌/캠페인 효과가 크므로, **이벤트·프로모션 캘린더와의 정합**을 함께 점검하는 것이 좋습니다.
        """
    )


# -------------------------
# Tab 5. Acquisition
# -------------------------
# -------------------------
# Tab 5. Acquisition
# -------------------------
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 윈도우라면 보통 'Malgun Gothic' 사용 가능
plt.rc('font', family='Malgun Gothic')  
plt.rc('axes', unicode_minus=False)  # 마이너스 기호 깨짐 방지
with tabAcq:
    # --------------------------------
    # 2. 유입 규모 & 신규 고객
    # --------------------------------
    st.header("1️⃣ 유입 규모 & 신규 고객")

    # ---- 채널별 ----
    st.subheader("📌 채널별 세션 & 신규 고객")
    channel_summary = (
        df.groupby("channelGrouping")
        .agg(sessions=("session_id","nunique"), new_sessions=("totals_newVisits","sum"))
        .reset_index()
    )
    channel_summary["new_visit_ratio"] = channel_summary["new_sessions"] / channel_summary["sessions"]
    channel_summary = channel_summary.sort_values("sessions", ascending=False)

    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.bar(channel_summary["channelGrouping"], channel_summary["sessions"], color="skyblue", alpha=0.7)
    ax1.set_ylabel("Sessions", color="blue")
    ax2 = ax1.twinx()
    ax2.plot(channel_summary["channelGrouping"], channel_summary["new_visit_ratio"], color="red", marker="o")
    ax2.set_ylabel("New Visit Ratio", color="red")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ---- 디바이스별 ----
    st.subheader("📌 디바이스별 세션 & 신규 고객")
    device_summary = (
        df.groupby("device_deviceCategory")
        .agg(sessions=("session_id","nunique"), new_sessions=("totals_newVisits","sum"))
        .reset_index()
    )
    device_summary["new_visit_ratio"] = device_summary["new_sessions"] / device_summary["sessions"]

    fig, ax1 = plt.subplots(figsize=(6,4))
    ax1.bar(device_summary["device_deviceCategory"], device_summary["sessions"], color="skyblue")
    ax2 = ax1.twinx()
    ax2.plot(device_summary["device_deviceCategory"], device_summary["new_visit_ratio"], color="red", marker="o")
    plt.title("Device Category: Sessions & New Visit Ratio")
    st.pyplot(fig)

    # ---- OS별 (Top 10) ----
    st.subheader("📌 OS별 세션 & 신규 고객")
    os_summary = (
        df.groupby("device_operatingSystem")
        .agg(sessions=("session_id","nunique"), new_sessions=("totals_newVisits","sum"))
        .reset_index()
    )
    os_summary["new_visit_ratio"] = os_summary["new_sessions"] / os_summary["sessions"]
    os_summary = os_summary.sort_values("sessions", ascending=False).head(10)

    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.bar(os_summary["device_operatingSystem"], os_summary["sessions"], color="skyblue")
    ax2 = ax1.twinx()
    ax2.plot(os_summary["device_operatingSystem"], os_summary["new_visit_ratio"], color="red", marker="o")
    plt.xticks(rotation=45)
    plt.title("OS (Top 10): Sessions & New Visit Ratio")
    st.pyplot(fig)

    # ---- Device × Channel 히트맵 ----
    st.subheader("📌 Device × Channel 히트맵 (평균 페이지뷰)")
    pivot = df.pivot_table(
        index="device_deviceCategory",
        columns="channelGrouping",
        values="totals_pageviews",
        aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
    plt.title("Average Pageviews by Device × Channel")
    st.pyplot(fig)

    # --------------------------------
    # 3. ML 기반 거래 예측
    # --------------------------------
    st.header("2️⃣ ML 기반 거래 예측 (Logistic Regression)")

    st.markdown("**목표:** 세션 특성(`channelGrouping`, `device_deviceCategory`, `device_operatingSystem`, "
                "`totals_pageviews`, `session_duration`, `totals_bounces`)을 X로 두고, "
                "`is_transaction`을 Y(거래 발생 여부)로 예측")

    features = ["channelGrouping", "device_deviceCategory", "device_operatingSystem",
                "totals_pageviews", "session_duration", "totals_bounces"]

    X = df[features].copy()
    y = df["is_transaction"]

    numeric_features = ["totals_pageviews", "session_duration", "totals_bounces"]
    categorical_features = ["channelGrouping","device_deviceCategory","device_operatingSystem"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
    model.fit(X_train, y_train)

    st.write("Train Accuracy:", model.score(X_train,y_train))
    st.write("Test Accuracy:", model.score(X_test,y_test))
    y_proba = model.predict_proba(X_test)[:,1]
    st.write("ROC AUC:", roc_auc_score(y_test,y_proba))

    # ---- Feature Importance ----
    ohe = model.named_steps["preprocessor"].named_transformers_["cat"]
    cat_features = ohe.get_feature_names_out(categorical_features)
    all_features = numeric_features + list(cat_features)
    coef = model.named_steps["classifier"].coef_[0]
    importance = pd.DataFrame({"feature":all_features,"coef":coef}).sort_values("coef")

    # Positive Top 10
    st.subheader("📈 Top Positive Factors (거래 확률 ↑)")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=importance.tail(10), x="coef", y="feature", color="green", ax=ax)
    st.pyplot(fig)

    # Negative Top 10
    st.subheader("📉 Top Negative Factors (거래 확률 ↓)")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=importance.head(10), x="coef", y="feature", color="red", ax=ax)
    st.pyplot(fig)

    # --------------------------------
    # 4. 세션 질 지표
    # --------------------------------
    st.header("3️⃣ 채널 질 평가 (회귀계수 기반 점수화)")
    # --------------------------------
    # 2. ML 학습 (거래 예측)
    # --------------------------------
    features = [
        "channelGrouping", 
        "device_deviceCategory", 
        "device_operatingSystem",
        "totals_pageviews", 
        "session_duration", 
        "totals_bounces"
    ]
    X = df[features].copy()
    y = df["is_transaction"]

    numeric_features = ["totals_pageviews", "session_duration", "totals_bounces"]
    categorical_features = ["channelGrouping","device_deviceCategory","device_operatingSystem"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
    model.fit(X_train, y_train)

    # coef 추출
    ohe = model.named_steps["preprocessor"].named_transformers_["cat"]
    cat_features = ohe.get_feature_names_out(categorical_features)
    all_features = numeric_features + list(cat_features)
    coef = model.named_steps["classifier"].coef_[0]
    importance = pd.DataFrame({"feature": all_features, "coef": coef})

    # 우리가 쓸 지표와 coef 매핑
    weights = {
        "avg_pageviews": abs(importance.loc[importance["feature"]=="totals_pageviews","coef"].values[0]),
        "avg_time_on_site": abs(importance.loc[importance["feature"]=="session_duration","coef"].values[0]),
        "bounce_score": abs(importance.loc[importance["feature"]=="totals_bounces","coef"].values[0])
    }

    st.subheader("📌 회귀계수 기반 지표 가중치")
    st.write(weights)

    # --------------------------------
    # 3. 채널 질 평가 (점수화)
    # --------------------------------

    quality_summary = (
        df.groupby("channelGrouping")
        .agg(
            avg_pageviews=("totals_pageviews","mean"),
            bounce_rate=("totals_bounces","mean"),
            avg_time_on_site=("session_duration","mean"),
            transaction_rate=("is_transaction","mean")
        )
        .reset_index()
    )

    # 정규화
    scaler = MinMaxScaler()
    quality_summary[["avg_pageviews","avg_time_on_site","transaction_rate"]] = scaler.fit_transform(
        quality_summary[["avg_pageviews","avg_time_on_site","transaction_rate"]]
    )
    quality_summary["bounce_score"] = 1 - MinMaxScaler().fit_transform(quality_summary[["bounce_rate"]])

    # 종합 점수 계산
    quality_summary["final_score"] = (
        quality_summary["avg_pageviews"] * weights["avg_pageviews"] +
        quality_summary["avg_time_on_site"] * weights["avg_time_on_site"] +
        quality_summary["bounce_score"] * weights["bounce_score"]
    ) / sum(weights.values())

    quality_summary = quality_summary.sort_values("final_score", ascending=False)

    # 표 출력
    st.dataframe(quality_summary[["channelGrouping","avg_pageviews","bounce_score","avg_time_on_site","transaction_rate","final_score"]])

    # 시각화
    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(data=quality_summary, x="final_score", y="channelGrouping", palette="Blues_r", ax=ax)
    ax.set_title("채널별 최종 질 평가 점수 (회귀계수 가중치 기반)")
    st.pyplot(fig)

    # --------------------------------
    st.header("4️⃣ 채널별 신규 vs 재방문 비율")

    # 채널별 신규/재방문 집계
    channel_visits = (
        df.groupby("channelGrouping")["totals_newVisits"]
        .agg(
            sessions="count",  # 전체 세션
            new_sessions="sum" # 신규 세션 수
        )
        .reset_index()
    )

    channel_visits["repeat_sessions"] = channel_visits["sessions"] - channel_visits["new_sessions"]
    channel_visits = channel_visits.sort_values("sessions", ascending=False)

    # 비율 계산
    channel_visits["new_ratio"] = channel_visits["new_sessions"] / channel_visits["sessions"]
    channel_visits["repeat_ratio"] = channel_visits["repeat_sessions"] / channel_visits["sessions"]

    # 시각화 (스택형 막대그래프)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(channel_visits["channelGrouping"], channel_visits["new_ratio"], label="신규", color="skyblue")
    ax.bar(channel_visits["channelGrouping"], channel_visits["repeat_ratio"], 
        bottom=channel_visits["new_ratio"], label="재방문", color="orange")

    ax.set_ylabel("비율")
    ax.set_xlabel("채널")
    plt.xticks(rotation=45)
    plt.legend()
    plt.title("채널별 신규 vs 재방문 비율")
    st.pyplot(fig)


    aq = (sf.groupby("channel", observed=False)
            .agg(sessions=("session_id","count"),
                 new_ratio=("newVisit","mean"),
                 pv_mean=("pv","mean"),
                 bounce_rate=("bounces","mean"),
                 dur_mean=("session_duration","mean"),
                 conv_rate=("is_transaction","mean"))
            .sort_values("sessions", ascending=False)).reset_index()
    
    st.markdown("**신규 방문 비율 추세 (Top 4 채널)**")
    top4 = aq.head(4)["channel"].tolist()
    daily_new = (sf[sf["channel"].isin(top4)]
                 .groupby(["session_date","channel"], observed=False)
                 .agg(new_ratio=("newVisit","mean"),
                      sessions=("session_id","count"))
                 .reset_index())
    chart_new = alt.Chart(daily_new).mark_line().encode(
        x="session_date:T", y=alt.Y("new_ratio:Q", axis=alt.Axis(format="~%")), color="channel:N"
    ).properties(height=280)
    st.altair_chart(chart_new, use_container_width=True)

    st.markdown("**세션 수 추세 (Top 4 채널)**")
    chart_sess = alt.Chart(daily_new).mark_line().encode(
        x="session_date:T", y="sessions:Q", color="channel:N"
    ).properties(height=280)
    st.altair_chart(chart_sess, use_container_width=True)

##################################################################################################
    st.header("4️⃣ 채널별 코호트 분석 (Retention)")

    # 날짜 변환
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")

    # 유저별 첫 방문 주차 + 첫 방문 채널
    user_first = (
        df.groupby("fullVisitorId")
        .agg(first_visit=("date", "min"),
            first_channel=("channelGrouping", "first"))
        .reset_index()
    )

    df_cohort = df.merge(user_first, on="fullVisitorId")
    df_cohort["cohort_week"] = df_cohort["first_visit"].dt.to_period("W")
    df_cohort["visit_week"] = df_cohort["date"].dt.to_period("W")

    # 몇 주 뒤인지 계산
    df_cohort["weeks_since"] = (
        df_cohort["visit_week"].astype(int) - df_cohort["cohort_week"].astype(int)
    )
    df_cohort = df_cohort[df_cohort["weeks_since"] >= 0]

    # 코호트 테이블 (채널별)
    cohort_pivot = (
        df_cohort.groupby(["first_channel", "cohort_week", "weeks_since"])["fullVisitorId"]
        .nunique()
        .reset_index()
    )

    # 첫 주차 대비 비율 계산
    cohort_size = cohort_pivot[cohort_pivot["weeks_since"] == 0][
        ["first_channel", "cohort_week", "fullVisitorId"]
    ].rename(columns={"fullVisitorId": "cohort_size"})

    cohort_ret = cohort_pivot.merge(cohort_size, on=["first_channel", "cohort_week"])
    cohort_ret["retention"] = cohort_ret["fullVisitorId"] / cohort_ret["cohort_size"]

    # 피벗 (채널별 평균 리텐션 보기)
    channel_retention = (
        cohort_ret.groupby(["first_channel", "weeks_since"])["retention"]
        .mean()
        .unstack(fill_value=0)
    )


    # 시각화
    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(
        channel_retention, 
        annot=False,        # 숫자 제거
        fmt=".1%", 
        cmap="YlOrBr", 
        ax=ax
    )
    plt.title("Channel-based Cohort Retention (Weekly)")
    st.pyplot(fig)


st.success("완료! 사이드바에서 기간/채널/디바이스/Activation 기준을 바꿔가며 Retention~Acquisition을 탐색하세요.")

