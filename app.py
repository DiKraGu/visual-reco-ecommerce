import streamlit as st
from PIL import Image
from src.recommender import VisualRecommender

st.set_page_config(
    page_title="Recommandation Visuelle E-Commerce",
    layout="wide",
)


@st.cache_resource
def load_engine() -> VisualRecommender:
    return VisualRecommender()


engine = load_engine()
catalog = engine.catalog

st.title("Moteur de Recommandation Visuelle E-Commerce")
st.caption(f"Backbone actif : **{engine.model_name}** | Catalogue indexé : **{len(catalog)} produits**")

# --- Barre latérale : filtres métier ---
st.sidebar.header("Filtres du catalogue")
gender_filter = st.sidebar.selectbox(
    "Genre", ["Tous"] + sorted(catalog["gender"].unique().tolist())
)
cat_filter = st.sidebar.selectbox(
    "Catégorie principale", ["Tous"] + sorted(catalog["masterCategory"].unique().tolist())
)
subcat_filter = st.sidebar.selectbox(
    "Sous-catégorie", ["Tous"] + sorted(catalog["subCategory"].unique().tolist())
)
color_filter = st.sidebar.selectbox(
    "Couleur dominante", ["Tous"] + sorted(catalog["baseColour"].unique().tolist())
)
top_k = st.sidebar.slider("Nombre de recommandations", min_value=4, max_value=16, value=8, step=4)

# --- Corps principal ---
col_input, col_results = st.columns([1, 3], gap="large")

with col_input:
    st.subheader("1. Image requête")
    mode = st.radio("Source :", ["Importer une photo", "Tester depuis le catalogue"])

    query_img = None
    if mode == "Importer une photo":
        uploaded = st.file_uploader("Image produit (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
        if uploaded:
            query_img = Image.open(uploaded)
    else:
        if "sample_seed" not in st.session_state:
            st.session_state.sample_seed = 42
        if st.button("Nouvel échantillon aléatoire"):
            st.session_state.sample_seed += 1

        samples = catalog.sample(15, random_state=st.session_state.sample_seed)
        selected_id = st.selectbox(
            "Produit du catalogue :",
            samples["id"].tolist(),
            format_func=lambda pid: f"#{pid} - {catalog.loc[catalog['id'] == pid, 'articleType'].values[0]}",
        )
        img_path = catalog.loc[catalog["id"] == selected_id, "image_path"].values[0]
        query_img = Image.open(img_path)

    if query_img:
        st.image(query_img, caption="Image analysée", use_container_width=True)

with col_results:
    st.subheader("2. Produits visuellement similaires")
    if query_img:
        filters = {
            "gender": gender_filter,
            "masterCategory": cat_filter,
            "subCategory": subcat_filter,
            "baseColour": color_filter,
        }
        results, latency = engine.search(query_img, top_k=top_k, filters=filters)
        st.caption(f"Temps d'inférence + recherche : **{latency:.1f} ms** | **{len(results)}** articles trouvés")

        if results.empty:
            st.warning("Aucun produit ne correspond à cette combinaison de filtres.")
        else:
            cols = st.columns(4)
            for idx, (_, row) in enumerate(results.iterrows()):
                with cols[idx % 4]:
                    st.image(row["image_path"], use_container_width=True)
                    st.markdown(f"**{row['articleType']}**")
                    st.caption(f"{row['gender']} • {row['baseColour']}")
                    score = float(max(0.0, min(1.0, row["similarity_score"])))
                    st.progress(score)
                    st.caption(f"Similarité : **{score * 100:.1f}%**")
    else:
        st.info("Chargez une image ou sélectionnez un produit à gauche pour lancer la recherche.")