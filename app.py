import streamlit as st
import sys
import os
sys.path.append("src")

from predictor import predict_player_style

# ── PAGE CONFIG ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Chess Style Profiler",
    page_icon="♟️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── GLOBAL CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0A0A0F;
    color: #E8E8F0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 3rem;
    padding-bottom: 4rem;
    max-width: 780px;
}

/* ── Hero Section ── */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    position: relative;
}

.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 100px;
    padding: 6px 18px;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #A0A0C0;
    margin-bottom: 1.5rem;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.8rem, 6vw, 4.2rem);
    font-weight: 900;
    line-height: 1.1;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #FFFFFF 0%, #A8C8CC 50%, #5B9EA6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
}

.hero-sub {
    font-size: 1.05rem;
    font-weight: 300;
    color: #7070A0;
    max-width: 480px;
    margin: 0 auto 2.5rem;
    line-height: 1.7;
}

/* ── Input Styling ── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: #E8E8F0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.75rem 1.2rem !important;
    transition: border-color 0.2s;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(91, 158, 166, 0.6) !important;
    box-shadow: 0 0 0 3px rgba(91, 158, 166, 0.12) !important;
}
.stTextInput > div > div > input::placeholder {
    color: #404060 !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #1A6B72, #2E8B92) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    transition: all 0.2s !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(26, 107, 114, 0.4) !important;
}

/* ── Style Card ── */
.style-card {
    border-radius: 20px;
    padding: 2.5rem;
    margin: 2rem 0;
    position: relative;
    overflow: hidden;
}
.style-card::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    border-radius: 50%;
    background: rgba(255,255,255,0.05);
}
.style-card-emoji {
    font-size: 3rem;
    margin-bottom: 0.75rem;
    display: block;
}
.style-card-label {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 0.4rem;
}
.style-card-name {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 1rem;
    color: white;
}
.style-card-desc {
    font-size: 0.95rem;
    line-height: 1.7;
    opacity: 0.85;
    color: white;
    max-width: 520px;
}

/* ── Stat Cards ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 1.5rem 0;
}
.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    transition: border-color 0.2s;
}
.stat-card:hover {
    border-color: rgba(91, 158, 166, 0.3);
}
.stat-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #5B9EA6;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.stat-label {
    font-size: 0.72rem;
    color: #505070;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #E8E8F0;
    margin: 2.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}

/* ── Bar Chart ── */
.bar-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
}
.bar-label {
    font-size: 0.82rem;
    color: #8080A0;
    width: 130px;
    flex-shrink: 0;
}
.bar-track {
    flex: 1;
    height: 8px;
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.8s ease;
}
.bar-pct {
    font-size: 0.82rem;
    color: #5B9EA6;
    width: 40px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Tips ── */
.tip-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 14px 16px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    margin-bottom: 10px;
}
.tip-num {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #2E8B92;
    flex-shrink: 0;
    line-height: 1.4;
}
.tip-text {
    font-size: 0.9rem;
    color: #9090B0;
    line-height: 1.6;
}

/* ── Coming Soon ── */
.coming-soon {
    background: rgba(91, 158, 166, 0.06);
    border: 1px solid rgba(91, 158, 166, 0.2);
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin: 2rem 0;
}
.coming-soon-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    color: #5B9EA6;
    margin-bottom: 0.5rem;
}
.coming-soon-desc {
    font-size: 0.88rem;
    color: #505070;
    line-height: 1.6;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2rem 0 0;
    font-size: 0.78rem;
    color: #303050;
    letter-spacing: 0.05em;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #2E8B92 !important;
}

/* ── Caption ── */
.stCaption {
    color: #404060 !important;
    text-align: center;
}

/* ── Error/Warning ── */
.stAlert {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #E8E8F0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────────────────
CLUSTER_CONFIG = {
    0: {
        "emoji": "⚔️",
        "name": "Chaotic Attacker",
        "color_start": "#8B1A1A",
        "color_end": "#C0392B",
        "description": "You create complications and go for the jugular — but inconsistent king safety costs you games. Your aggression is your strength. Structure it.",
        "tips": [
            "Castle before move 10 in every single game, no exceptions",
            "Before launching an attack, pause and ask: is my king actually safe?",
            "Study Mikhail Tal — the greatest attacker who also knew when to consolidate",
            "Practice converting winning positions. Don't just attack — finish.",
        ]
    },
    1: {
        "emoji": "🏰",
        "name": "Solid Strategist",
        "color_start": "#1A4A6B",
        "color_end": "#1A6B72",
        "description": "You play principled, well-rounded chess. You castle reliably, manage time well, and have a broad opening repertoire. The hallmark of a complete player.",
        "tips": [
            "Work on converting small positional advantages — this is your growth edge",
            "Study endgame technique; you reach endgames often and should win more of them",
            "Try some sharper openings occasionally to expand your tactical sharpness",
            "Study Magnus Carlsen's technique for squeezing wins from equal positions",
        ]
    },
    2: {
        "emoji": "⏱️",
        "name": "Time Pressure Wildcard",
        "color_start": "#7A4A00",
        "color_end": "#D4890A",
        "description": "Your biggest challenge is the clock. You slow down dramatically in critical positions, leading to time scrambles. Your chess is solid — your clock management needs work.",
        "tips": [
            "Practice with a physical or digital clock every session — treat time as a resource",
            "Learn to make 'good enough' moves quickly. Perfect is the enemy of done.",
            "Play more rapid and blitz games specifically to build time intuition",
            "Study positions similar to ones you reach — familiarity speeds up decisions",
        ]
    },
    3: {
        "emoji": "🛡️",
        "name": "Passive Defender",
        "color_start": "#3A1A6B",
        "color_end": "#5B3D8A",
        "description": "You play cautiously and avoid complications. This keeps you safe but limits your winning chances. Learning to find active moves in quiet positions will transform your results.",
        "tips": [
            "Try playing 1.e4 — open games force you to find active piece moves",
            "Solve 10 tactical puzzles daily. Pattern recognition is the fastest path to confidence",
            "In every position, ask: what is my most active legal move?",
            "Study Kasparov or Tal for inspiration on proactive, dynamic play",
        ]
    },
}

# ── HERO ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">Powered by Lichess · ML · python-chess</div>
    <div class="hero-title">Know Your<br>Chess DNA</div>
    <div class="hero-sub">
        Enter your Lichess username. We analyze your last 50 rapid games 
        and build a complete behavioral profile of how you actually play.
    </div>
</div>
""", unsafe_allow_html=True)

# ── INPUT ─────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    username = st.text_input(
        "username",
        placeholder="Your Lichess username...",
        label_visibility="collapsed"
    )
with col2:
    analyze = st.button("Analyze ♟️", use_container_width=True)

st.caption("Takes ~30 seconds · Requires 10+ rated rapid games on Lichess")

# ── ANALYSIS ──────────────────────────────────────────────────────
if analyze and username:

    with st.spinner(f"Analyzing {username.strip()}..."):
        profile = predict_player_style(username.strip())

    if "error" in profile:
        st.error(f"❌ {profile['error']}")
        st.markdown("""
        <div style='color:#404060; font-size:0.85rem; text-align:center; margin-top:0.5rem;'>
            Make sure you have rated rapid games on Lichess 
            and that your username is spelled correctly.
        </div>
        """, unsafe_allow_html=True)

    else:
        cid   = profile["cluster_id"]
        cfg   = CLUSTER_CONFIG[cid]
        stats = profile["stats"]

        # ── STYLE CARD ────────────────────────────────────────────
        st.markdown(f"""
        <div class="style-card" style="background: linear-gradient(135deg, {cfg['color_start']}, {cfg['color_end']});">
            <span class="style-card-emoji">{cfg['emoji']}</span>
            <div class="style-card-label">Your Playing Style</div>
            <div class="style-card-name">{cfg['name']}</div>
            <div class="style-card-desc">{cfg['description']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='text-align:center; color:#404060; font-size:0.82rem; margin-top:-1rem; margin-bottom:1rem;'>
            Based on {profile['games_analyzed']} rapid games
        </div>
        """, unsafe_allow_html=True)

        # ── STAT GRID ─────────────────────────────────────────────
        st.markdown('<div class="section-header">Your Numbers</div>',
                    unsafe_allow_html=True)

        def stat_card(value, label):
            return f"""
            <div class="stat-card">
                <div class="stat-value">{value}</div>
                <div class="stat-label">{label}</div>
            </div>"""

        # Build stat values first to avoid nested f-string issues
        s1 = f"{stats['win_rate']*100:.0f}%"
        s2 = f"{stats['castle_rate']*100:.0f}%"
        s3 = f"M{stats['avg_castle_move']:.0f}"
        s4 = f"{stats['avg_game_length']:.0f}"
        s5 = f"{stats['gambit_rate']*100:.0f}%"
        s6 = f"{stats['sacrifice_rate']*100:.1f}%"
        s7 = f"{stats['opening_entropy']:.1f}"
        s8 = f"{stats['panic_score']:.1f}x"

        grid1 = (stat_card(s1, "Win Rate") + stat_card(s2, "Castle Rate") +
                 stat_card(s3, "Avg Castle") + stat_card(s4, "Avg Length"))
        grid2 = (stat_card(s5, "Gambit Rate") + stat_card(s6, "Sacrifice Rate") +
                 stat_card(s7, "Repertoire") + stat_card(s8, "Panic Score"))

        st.markdown(
            f'<div class="stat-grid">{grid1}</div>'
            f'<div class="stat-grid">{grid2}</div>',
            unsafe_allow_html=True
        )

        # ── STYLE BARS ────────────────────────────────────────────
        st.markdown('<div class="section-header">Style Breakdown</div>',
                    unsafe_allow_html=True)

        def bar(label, value, max_val=1.0, color="#2E8B92"):
            pct = min(value / max_val, 1.0) * 100
            return f"""
            <div class="bar-row">
                <div class="bar-label">{label}</div>
                <div class="bar-track">
                    <div class="bar-fill" 
                         style="width:{pct:.0f}%; background:{color};"></div>
                </div>
                <div class="bar-pct">{pct:.0f}%</div>
            </div>"""

        st.markdown(
            bar("Win Rate",        stats["win_rate"],        1.0,  "#2E8B92") +
            bar("Castle Rate",     stats["castle_rate"],     1.0,  "#1A6B72") +
            bar("Gambit Tendency", stats["gambit_rate"],     0.4,  "#D4890A") +
            bar("Sacrifice Rate",  stats["sacrifice_rate"],  0.2,  "#C0392B") +
            bar("Opening Variety", stats["opening_entropy"], 4.5,  "#5B3D8A") +
            bar("Time Pressure",   stats["panic_score"],     15.0, "#7A4A00"),
            unsafe_allow_html=True
        )

        # ── IMPROVEMENT TIPS ──────────────────────────────────────
        st.markdown('<div class="section-header">How to Improve</div>',
                    unsafe_allow_html=True)

        tips_html = ""
        for i, tip in enumerate(cfg["tips"], 1):
            tips_html += f"""
            <div class="tip-item">
                <div class="tip-num">0{i}</div>
                <div class="tip-text">{tip}</div>
            </div>"""

        st.markdown(tips_html, unsafe_allow_html=True)

        # ── PUZZLE TEASER ─────────────────────────────────────────
        st.markdown(f"""
        <div class="coming-soon">
            <div class="coming-soon-title">🧩 Personalized Puzzles — Coming Soon</div>
            <div class="coming-soon-desc">
                Based on your <strong style="color:#5B9EA6">{cfg['name']}</strong> 
                profile, we'll recommend puzzles from the Lichess database 
                (500K+ puzzles) targeting your specific weaknesses. 
                Practice what actually needs work.
            </div>
        </div>
        """, unsafe_allow_html=True)

elif analyze and not username:
    st.warning("Please enter a Lichess username.")

# ── FOOTER ────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Chess Style Profiler &nbsp;·&nbsp; 
    Ishani Bhat &nbsp;·&nbsp; 
    UW MSDS 2025 &nbsp;·&nbsp;
    Built with Lichess API, python-chess, scikit-learn
</div>
""", unsafe_allow_html=True)