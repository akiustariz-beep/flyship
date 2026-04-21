import os
import json
import streamlit as st
from datetime import datetime
from pathlib import Path

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FlyShip · Agente ML",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── styling ──────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Main header */
.flyship-header {
    background: linear-gradient(135deg, #FFE600 0%, #FFC400 100%);
    padding: 1.2rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.flyship-header h1 { margin: 0; color: #333; font-size: 1.8rem; }
.flyship-header p  { margin: 0; color: #555; font-size: 0.95rem; }

/* Publication card */
.pub-card {
    background: #fff;
    border: 1px solid #e8e8e8;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.pub-card h3 { margin: 0 0 0.5rem 0; font-size: 1rem; color: #1a1a1a; }
.pub-price   { font-size: 1.4rem; font-weight: 700; color: #00a650; }

/* Recommendation cards */
.rec-alta  { border-left: 4px solid #d93025; background: #fff8f8; padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 0.5rem; }
.rec-media { border-left: 4px solid #f29900; background: #fffbf0; padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 0.5rem; }
.rec-baja  { border-left: 4px solid #1e8449; background: #f0fff4; padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 0.5rem; }
.rec-alta h4, .rec-media h4, .rec-baja h4 { margin: 0 0 0.3rem 0; font-size: 0.9rem; }
.rec-action { font-size: 0.85rem; color: #444; margin-top: 0.3rem; }
.rec-impact { font-size: 0.78rem; color: #888; font-style: italic; margin-top: 0.2rem; }

/* Score badge */
.score-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1rem;
}
.score-low    { background: #fde8e8; color: #c0392b; }
.score-medium { background: #fef9e7; color: #d68910; }
.score-high   { background: #e8f5e9; color: #1e8449; }

/* Opp card */
.opp-card {
    background: #fff;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.opp-card h4 { margin: 0 0 0.4rem 0; color: #1a1a1a; }

/* Metric pill */
.pill {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.3rem;
}
.pill-green  { background: #d4edda; color: #155724; }
.pill-red    { background: #f8d7da; color: #721c24; }
.pill-yellow { background: #fff3cd; color: #856404; }
.pill-blue   { background: #d1ecf1; color: #0c5460; }

button[data-testid="stButton"] > button { border-radius: 8px !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ── constants ────────────────────────────────────────────────────────────────
DATA_FILE = Path(__file__).parent / "data" / "publications.json"
DATA_FILE.parent.mkdir(exist_ok=True)

DEFAULT_PUBLICATIONS = [
    {
        "id": "MLA2941841712",
        "url": "https://articulo.mercadolibre.com.ar/MLA-2941841712-parlante-bluetooth-mini-portatil-recargable-inalambrico-_JM",
        "name": "Parlantito Bluetooth",
        "emoji": "🔊",
        "added_at": "2025-01-01",
        "listing": None,
        "analyses": [],
    },
    {
        "id": "MLA2944419502",
        "url": "https://articulo.mercadolibre.com.ar/MLA-2944419502-lampara-de-mesa-led-origami-plegable-recargable-usb-moderna-_JM",
        "name": "Lámpara LED Origami",
        "emoji": "💡",
        "added_at": "2025-01-01",
        "listing": None,
        "analyses": [],
    },
]

# ── data helpers ─────────────────────────────────────────────────────────────
def load_data() -> list:
    if DATA_FILE.exists():
        with open(DATA_FILE, encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_PUBLICATIONS


def save_data(pubs: list):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(pubs, f, ensure_ascii=False, indent=2, default=str)


def get_score_class(score: float) -> str:
    if score >= 7:
        return "score-high"
    if score >= 4:
        return "score-medium"
    return "score-low"


def priority_class(p: str) -> str:
    return {"alta": "rec-alta", "media": "rec-media", "baja": "rec-baja"}.get(p.lower(), "rec-baja")


def pill_level(level: str, invert: bool = False) -> str:
    """Return HTML pill. invert=True means high=bad (for competition)."""
    level = level.lower()
    if not invert:
        color = {"alta": "pill-green", "media": "pill-yellow", "baja": "pill-red"}.get(level, "pill-blue")
    else:
        color = {"alta": "pill-red", "media": "pill-yellow", "baja": "pill-green"}.get(level, "pill-blue")
    labels = {"alta": "Alta", "media": "Media", "baja": "Baja"}
    return f'<span class="pill {color}">{labels.get(level, level)}</span>'


# ── session state ─────────────────────────────────────────────────────────────
if "publications" not in st.session_state:
    st.session_state.publications = load_data()

if "bq_verticals" not in st.session_state:
    st.session_state.bq_verticals = None

if "opp_result" not in st.session_state:
    st.session_state.opp_result = None

# ── header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="flyship-header">
  <div>
    <h1>🚀 FlyShip &nbsp;·&nbsp; Agente MercadoLibre</h1>
    <p>Optimizá tus publicaciones e identificá oportunidades de importación</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── API key check ─────────────────────────────────────────────────────────────
api_key = os.environ.get("GROQ_API_KEY", "")
if not api_key:
    with st.sidebar:
        st.subheader("⚙️ Configuración")
        key_input = st.text_input("GROQ_API_KEY", type="password", placeholder="gsk_...")
        if key_input:
            os.environ["GROQ_API_KEY"] = key_input
            st.success("Clave guardada en sesión")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_pubs, tab_opps = st.tabs(["📦  Mis Publicaciones", "🔍  Oportunidades"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MIS PUBLICACIONES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pubs:
    pubs = st.session_state.publications

    # ── Add new publication ──────────────────────────────────────────────────
    with st.expander("➕  Agregar nueva publicación", expanded=False):
        col_url, col_name, col_btn = st.columns([3, 1.5, 1])
        with col_url:
            new_url = st.text_input(
                "URL de la publicación",
                placeholder="https://articulo.mercadolibre.com.ar/MLA-...",
                label_visibility="collapsed",
            )
        with col_name:
            new_name = st.text_input(
                "Nombre corto",
                placeholder="Ej: Auriculares BT",
                label_visibility="collapsed",
            )
        with col_btn:
            add_clicked = st.button("Agregar", use_container_width=True)

        if add_clicked:
            if not new_url or "mercadolibre" not in new_url:
                st.error("Ingresá una URL válida de MercadoLibre.")
            else:
                from scraper import extract_item_id
                item_id = extract_item_id(new_url)
                if not item_id:
                    st.error("No se pudo extraer el ID del item de la URL.")
                elif any(p["id"] == item_id for p in pubs):
                    st.warning("Esta publicación ya está guardada.")
                else:
                    new_pub = {
                        "id": item_id,
                        "url": new_url.split("?")[0],
                        "name": new_name or item_id,
                        "emoji": "📦",
                        "added_at": datetime.now().strftime("%Y-%m-%d"),
                        "listing": None,
                        "analyses": [],
                    }
                    st.session_state.publications.append(new_pub)
                    save_data(st.session_state.publications)
                    st.success(f"Publicación **{new_pub['name']}** agregada. Hacé clic en 'Analizar' para obtener recomendaciones.")
                    st.rerun()

    st.markdown("---")

    # ── Publication cards ────────────────────────────────────────────────────
    for idx, pub in enumerate(st.session_state.publications):
        listing = pub.get("listing") or {}
        analyses = pub.get("analyses") or []
        last_analysis = analyses[-1] if analyses else None

        with st.container():
            col_info, col_metrics, col_action = st.columns([3, 2, 1.5])

            with col_info:
                title_display = listing.get("title") or pub["name"]
                price = listing.get("price")
                st.markdown(
                    f"""<div class="pub-card">
  <h3>{pub['emoji']} &nbsp;{pub['name']}</h3>
  <p style="color:#666;font-size:0.82rem;margin:0 0 0.5rem 0">{title_display}</p>
  {'<p class="pub-price">$' + f"{price:,}" + ' ARS</p>' if price else '<p style="color:#aaa;font-size:0.85rem">Precio: no cargado aún</p>'}
  <p style="font-size:0.78rem;color:#aaa;margin:0.3rem 0 0 0">🔗 <a href="{pub['url']}" target="_blank">Ver en MercadoLibre</a></p>
</div>""",
                    unsafe_allow_html=True,
                )

            with col_metrics:
                if listing:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("⭐ Rating", listing.get("rating") or "—")
                    m2.metric("💬 Reseñas", listing.get("reviews_count") or 0)
                    m3.metric("🖼️ Fotos", listing.get("images_count") or 0)
                    if last_analysis:
                        score = last_analysis.get("score", 0)
                        sc = get_score_class(score)
                        ts = last_analysis.get("timestamp", "")[:10]
                        st.markdown(
                            f'Puntaje: <span class="score-badge {sc}">{score}/10</span>'
                            f'<span style="font-size:0.75rem;color:#aaa;margin-left:0.5rem">({ts})</span>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("Hacé clic en **Analizar** para cargar los datos.")

            with col_action:
                analyze_key = f"analyze_{pub['id']}"
                if st.button("🔍 Analizar", key=analyze_key, use_container_width=True):
                    if not os.environ.get("GROQ_API_KEY"):
                        st.error("Configurá tu ANTHROPIC_API_KEY en el panel lateral.")
                    else:
                        with st.spinner(f"Analizando **{pub['name']}**… esto puede tardar 30-60 seg."):
                            try:
                                from scraper import scrape_listing, scrape_search_results, build_search_query_from_title
                                from analyzer import analyze_publication

                                # 1. Scrape own listing (use cached if scraping is blocked)
                                try:
                                    fresh_listing = scrape_listing(pub["url"])
                                except Exception:
                                    fresh_listing = {}

                                # Fall back to cached listing if scraping returned no title
                                if not fresh_listing.get("title"):
                                    fresh_listing = pub.get("listing") or {}

                                if not fresh_listing.get("title"):
                                    st.error("No se pudieron obtener los datos de la publicación. MercadoLibre bloquea el scraping desde servidores cloud. Corré la app localmente para analizarla.")
                                    st.stop()

                                st.session_state.publications[idx]["listing"] = fresh_listing

                                # 2. Scrape competitors
                                search_q = build_search_query_from_title(
                                    fresh_listing.get("title") or pub["name"]
                                )
                                try:
                                    competitors = scrape_search_results(search_q, limit=25)
                                except Exception:
                                    competitors = []

                                # 3. AI analysis
                                result = analyze_publication(fresh_listing, competitors)
                                result["timestamp"] = datetime.now().isoformat()
                                result["search_query"] = search_q
                                result["competitors_count"] = len(competitors)

                                st.session_state.publications[idx]["analyses"].append(result)
                                save_data(st.session_state.publications)
                                st.success("Análisis completado.")
                                st.rerun()

                            except Exception as e:
                                st.error(f"Error al analizar: {e}")

                # Delete button
                if st.button("🗑️ Eliminar", key=f"del_{pub['id']}", use_container_width=True):
                    st.session_state.publications.pop(idx)
                    save_data(st.session_state.publications)
                    st.rerun()

        # ── Analysis results ─────────────────────────────────────────────────
        if last_analysis:
            with st.expander(
                f"📊 Últimas recomendaciones — {pub['name']} "
                f"(análisis del {last_analysis.get('timestamp','')[:10]})",
                expanded=True,
            ):
                score = last_analysis.get("score", 0)
                sc = get_score_class(score)

                top_col, summary_col = st.columns([1, 3])
                with top_col:
                    st.markdown(
                        f'<div style="text-align:center;padding:1rem">'
                        f'<div style="font-size:0.8rem;color:#888;margin-bottom:0.3rem">PUNTAJE ACTUAL</div>'
                        f'<span class="score-badge {sc}" style="font-size:1.8rem;padding:0.5rem 1.2rem">{score}/10</span>'
                        f'<div style="font-size:0.78rem;color:#666;margin-top:0.5rem">{last_analysis.get("score_reason","")}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with summary_col:
                    st.info(last_analysis.get("summary", ""))
                    price_pos = last_analysis.get("price_position", "")
                    if price_pos:
                        st.markdown(f"💰 **Posición de precio:** {price_pos}")
                    st.caption(
                        f"Comparado con {last_analysis.get('competitors_count', 0)} competidores "
                        f"· búsqueda: *{last_analysis.get('search_query', '')}*"
                    )

                # ── Título actual vs propuesto ────────────────────────────
                current_title = listing.get("title", "")
                proposed_title = last_analysis.get("proposed_title", "")
                if current_title or proposed_title:
                    st.markdown("#### 🏷️ Título")
                    tcol1, tcol2 = st.columns(2)
                    with tcol1:
                        st.markdown("**Título actual**")
                        st.markdown(
                            f'<div style="background:#f8f8f8;border:1px solid #ddd;border-radius:8px;padding:0.7rem 1rem;font-size:0.9rem;color:#555">{current_title}</div>',
                            unsafe_allow_html=True,
                        )
                    with tcol2:
                        st.markdown("**✅ Título propuesto**")
                        st.markdown(
                            f'<div style="background:#e8f5e9;border:1px solid #a5d6a7;border-radius:8px;padding:0.7rem 1rem;font-size:0.9rem;font-weight:600;color:#1b5e20">{proposed_title or "—"}</div>',
                            unsafe_allow_html=True,
                        )

                st.markdown("#### 📋 Recomendaciones")

                recs = last_analysis.get("recommendations", [])
                cat_icons = {
                    "título": "🏷️", "precio": "💰", "fotos": "📸", "video": "🎬",
                    "descripción": "📝", "envío": "🚚", "atributos": "📋",
                    "palabras_clave": "🔍", "oferta": "🎯",
                }

                for rec in recs:
                    priority = rec.get("priority", "baja").lower()
                    cat = rec.get("category", "").lower()
                    icon = cat_icons.get(cat, "•")
                    css = priority_class(priority)
                    priority_label = {"alta": "🔴 ALTA", "media": "🟡 MEDIA", "baja": "🟢 BAJA"}.get(priority, priority)
                    proposed = rec.get("proposed_value", "")
                    proposed_html = (
                        f'<div style="margin-top:0.4rem;background:#e8f5e9;border:1px solid #a5d6a7;'
                        f'border-radius:6px;padding:0.4rem 0.7rem;font-size:0.83rem;color:#1b5e20">'
                        f'📋 <strong>Copiar y pegar:</strong> {proposed}</div>'
                    ) if proposed else ""
                    st.markdown(
                        f'<div class="{css}">'
                        f'<h4>{icon} {cat.upper()} &nbsp;<span style="font-weight:400;font-size:0.78rem">prioridad {priority_label}</span></h4>'
                        f'<div style="font-size:0.85rem;color:#333;font-weight:500">{rec.get("issue","")}</div>'
                        f'<div class="rec-action">💡 <strong>Acción:</strong> {rec.get("action","")}</div>'
                        f'{proposed_html}'
                        f'<div class="rec-impact">📈 <em>{rec.get("impact","")}</em></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # History toggle
            if len(analyses) > 1:
                with st.expander(f"📜 Historial de análisis ({len(analyses)} en total)", expanded=False):
                    for i, a in enumerate(reversed(analyses)):
                        score = a.get("score", "?")
                        ts = a.get("timestamp", "")[:16].replace("T", " ")
                        st.markdown(f"**#{len(analyses)-i}** · {ts} · Puntaje: {score}/10 · {a.get('summary','')[:120]}…")

        st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — OPORTUNIDADES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_opps:
    st.markdown("### 🔍 Oportunidades de importación")
    st.markdown(
        "Seleccioná un vertical de MercadoLibre para identificar nichos con potencial de importación desde China."
    )

    col_sel, col_btn2 = st.columns([3, 1])

    with col_sel:
        # Try to load verticals from BigQuery
        if st.session_state.bq_verticals is None:
            with st.spinner("Cargando verticales desde BigQuery…"):
                try:
                    from bq_client import get_verticals
                    verts = get_verticals()
                    st.session_state.bq_verticals = verts if verts else []
                except Exception as e:
                    st.session_state.bq_verticals = []
                    st.warning(f"No se pudo conectar a BigQuery: {e}")

        verticals = st.session_state.bq_verticals or []

        if verticals:
            selected_vertical = st.selectbox(
                "Vertical / Categoría",
                options=verticals,
                index=0,
                label_visibility="collapsed",
            )
        else:
            selected_vertical = st.text_input(
                "Vertical (ingresá manualmente si BigQuery no está disponible)",
                placeholder="Ej: Electrónica, Hogar, Deportes…",
                label_visibility="collapsed",
            )

    with col_btn2:
        analyze_opp = st.button("🔍 Analizar oportunidades", use_container_width=True)

    if analyze_opp:
        if not selected_vertical:
            st.error("Seleccioná o ingresá un vertical.")
        elif not os.environ.get("GROQ_API_KEY"):
            st.error("Configurá tu ANTHROPIC_API_KEY en el panel lateral.")
        else:
            with st.spinner(f"Analizando oportunidades en **{selected_vertical}**…"):
                try:
                    from analyzer import analyze_opportunities

                    # Get vertical data from BigQuery (or empty list if unavailable)
                    vertical_data = []
                    try:
                        from bq_client import get_vertical_data
                        vertical_data = get_vertical_data(selected_vertical)
                    except Exception:
                        pass

                    # If no BQ data, still run Claude analysis based on vertical name
                    if not vertical_data:
                        vertical_data = [{"vertical": selected_vertical, "note": "Datos de BigQuery no disponibles — análisis basado en conocimiento del mercado"}]

                    result = analyze_opportunities(vertical_data, selected_vertical)
                    st.session_state.opp_result = result

                except Exception as e:
                    st.error(f"Error en el análisis: {e}")

    # ── Display opportunities result ────────────────────────────────────────
    if st.session_state.opp_result:
        opp = st.session_state.opp_result

        st.markdown(f"#### Mercado: {opp.get('vertical', selected_vertical)}")
        st.info(opp.get("market_summary", ""))

        top_rec = opp.get("top_recommendation", "")
        if top_rec:
            st.success(f"⭐ **Recomendación top:** {top_rec}")

        st.markdown("#### Nichos identificados")

        opportunities = opp.get("opportunities", [])
        if not opportunities:
            st.warning("No se encontraron oportunidades para este vertical.")
        else:
            for opp_item in opportunities:
                score_val = opp_item.get("score", 0)
                score_color = "#1e8449" if score_val >= 7 else ("#d68910" if score_val >= 5 else "#c0392b")

                demand = opp_item.get("demand_level", "")
                competition = opp_item.get("competition_level", "")

                st.markdown(
                    f"""<div class="opp-card">
  <h4>
    {opp_item.get('product', 'N/D')}
    &nbsp;&nbsp;
    <span style="background:{score_color};color:#fff;padding:0.2rem 0.6rem;border-radius:12px;font-size:0.82rem;font-weight:600">{score_val}/10</span>
  </h4>
  <div style="margin-bottom:0.5rem">
    Demanda: {pill_level(demand)} &nbsp; Competencia: {pill_level(competition, invert=True)}
    &nbsp;
    <span class="pill pill-blue">Importación {opp_item.get('estimated_import_cost','?')}</span>
    <span class="pill pill-green">Venta {opp_item.get('estimated_selling_price','?')}</span>
  </div>
  <div style="font-size:0.85rem;color:#444;margin-bottom:0.4rem">{opp_item.get('why','')}</div>
  <div style="font-size:0.8rem;color:#666">
    <strong>Diferenciadores:</strong> {', '.join(opp_item.get('key_differentiators', []))}
  </div>
  <div style="font-size:0.78rem;color:#888;margin-top:0.3rem">
    🔑 Keywords: <em>{', '.join(opp_item.get('search_keywords', []))}</em>
  </div>
</div>""",
                    unsafe_allow_html=True,
                )

        # Reload verticals button
        st.markdown("---")
        if st.button("🔄 Refrescar verticales de BigQuery"):
            st.session_state.bq_verticals = None
            st.session_state.opp_result = None
            st.rerun()
