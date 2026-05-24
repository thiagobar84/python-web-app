import streamlit as st
import folium
from streamlit_folium import st_folium
import rasterio
from rasterio.warp import transform_bounds
import os
import numpy as np
import cv2
from shapely.geometry import Polygon
import geopandas  as gpd  # Corrigido alias padrão geopandas
import geopandas as gpd
import zipfile
import gc

# Configuração da página do Streamlit
st.set_page_config(layout="wide", page_title="Visualizador de Ortofotos")
st.title("🗺️ Visualizador Web de Ortofotos com IA")

# CSS para travar a opacidade e evitar o efeito escuro chato ao interagir com o mapa
st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlockBorderWrapper"] { opacity: 1 !important; filter: none !important; }
    div[data-testid="stVerticalBlock"] { opacity: 1 !important; }
    [stale-data="true"] { opacity: 1 !important; filter: none !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# Cria duas colunas: coluna 1 (peso 1) para controles, coluna 2 (peso 3) para o mapa maior
col1, col2 = st.columns([1, 3])

arquivo_path = None

with col1:
    st.header("Painel de Controle")
    arquivo_subido = st.file_uploader("Suba sua ortofoto (.tif ou .tiff)", type=["tif", "tiff"])
    if arquivo_subido is not None:
        arquivo_path = os.path.join("temp_ortofoto.tif")
        with open(arquivo_path, "wb") as f:
            f.write(arquivo_subido.getbuffer())
    
    # Botão para ativar a vetorização por IA
    rodar_ia = st.button("🤖 Executar IA de Vetorização")

# --- FUNÇÃO TRATAMENTO DA IMAGEM ---
def processar_ortofoto(caminho_imagem):
    with rasterio.open(caminho_imagem) as src:
        fator_reducao = 4
        
        img_data = src.read(
            out_shape=(src.count, int(src.height / fator_reducao), int(src.width / fator_reducao)),
            resampling=rasterio.enums.Resampling.bilinear
        )
        img_data = np.moveaxis(img_data, 0, -1)
        
        img_valida = img_data[img_data > 0]
        if len(img_valida) > 0:
            p_max = np.percentile(img_valida, 98)
            p_min = np.percentile(img_valida, 2)
            img_data = np.clip(img_data, p_min, p_max)
            img_data = ((img_data - p_min) / (max(1, p_max - p_min)) * 255).astype(np.uint8)
        else:
            img_data = img_data.astype(np.uint8)
            
        bounds = src.bounds
        crs = src.crs
        west, south, east, north = transform_bounds(crs, 'EPSG:4326', *bounds)
        
    return img_data, [south, west, north, east]


# --- FUNÇÃO DE IA PARA DETECTAR CONTORNOS ---
from samgeo import SamGeo

def vetorizar_casas(img_data, limites):
    south, west, north, east = limites
    img_temp_path = "temp_para_ia.tif"
    
    with rasterio.open(
        img_temp_path, 'w', driver='GTiff',
        height=img_data.shape[0], width=img_data.shape[1],
        count=3, dtype=img_data.dtype,
        crs="EPSG:4326",
        transform=rasterio.transform.from_bounds(west, south, east, north, img_data.shape[1], img_data.shape[0])
    ) as dst:
        for i in range(3):
            dst.write(img_data[:, :, i], i + 1)

    poligonos_geo = []
    mask_tiff = "temp_resultado_ia.tif"
    output_gpkg = "temp_casas_sam.gpkg"
    
    try:
        sam = SamGeo(
            model_type="vit_b",
            checkpoint="sam_vit_b_01ec64.pth",
            sam_kwargs=None
        )
        
        sam.generate(img_temp_path, output=mask_tiff, erosion_kernel=(3, 3), grid_percentage=200)
        
        if os.path.exists(mask_tiff):
            sam.tiff_to_gpkg(mask_tiff, output_gpkg, simplify_tolerance=None)
        
        if os.path.exists(output_gpkg):
            gdf_sam = gpd.read_file(output_gpkg)
            
            for geom in gdf_sam.geometry:
                if geom.geom_type == 'Polygon':
                    coords = list(geom.exterior.coords)
                    coords_folium = [[pt[1], pt[0]] for pt in coords]
                    poligonos_geo.append(coords_folium)
                elif geom.geom_type == 'MultiPolygon':
                    for parte in geom.geoms:
                        coords = list(parte.exterior.coords)
                        coords_folium = [[pt[1], pt[0]] for pt in coords]
                        poligonos_geo.append(coords_folium)
                        
    except Exception as sam_error:
        st.error(f"Erro no processamento do SAM: {sam_error}")
        
    finally:
        for arquivo in [img_temp_path, mask_tiff, output_gpkg]:
            if os.path.exists(arquivo):
                try:
                    os.remove(arquivo)
                except:
                    pass
        gc.collect()
            
    return poligonos_geo

with col2:
    if arquivo_path:
        try:
            with st.spinner("⏳ Processando ortofoto e extraindo metadados... Por favor, aguarde."):
                img_data, limites = processar_ortofoto(arquivo_path)
            
            st.success("✅ Ortofoto carregada e processada com sucesso!")
            
            south, west, north, east = limites
            centro_lat = (south + north) / 2
            centro_lon = (west + east) / 2
            
            m = folium.Map(location=[centro_lat, centro_lon], zoom_start=16, control_scale=True)
            
            folium.raster_layers.ImageOverlay(
                image=img_data,
                bounds=[[south, west], [north, east]],
                opacity=0.8,
                name="Ortofoto"
            ).add_to(m)

            if "poligonos_detectados" not in st.session_state:
                st.session_state.poligonos_detectados = []

            if rodar_ia:
                with st.spinner("🤖 IA analisando texturas e vetorizando telhados..."):
                    st.session_state.poligonos_detectados = vetorizar_casas(img_data, limites)
                st.sidebar.success(f"🤖 IA identificou {len(st.session_state.poligonos_detectados)} estruturas!")

            if st.session_state.poligonos_detectados:
                for index, poli in enumerate(st.session_state.poligonos_detectados):
                    folium.Polygon(
                        locations=poli,  # 🏢 CORRIGIDO: mudado de polyline para poli
                        color="red",
                        weight=2,
                        fill=True,
                        fill_color="red",
                        fill_opacity=0.4,
                        popup=f"Construção {index+1}"
                    ).add_to(m)

                with col1:
                    st.write("---")
                    st.subheader("Exportar Vetores")
                    
                    try:
                        lista_shapely = []
                        for poli in st.session_state.poligonos_detectados:
                            coordenadas_gis = [(pt[1], pt[0]) for pt in poli]
                            if len(coordenadas_gis) >= 3:
                                lista_shapely.append(Polygon(coordenadas_gis))
                        
                        if lista_shapely:
                            gdf_export = gpd.GeoDataFrame(geometry=lista_shapely, crs="EPSG:4326")
                            
                            pasta_shapefile = "vetores_ia"
                            os.makedirs(pasta_shapefile, exist_ok=True)
                            base_nome = os.path.join(pasta_shapefile, "casas_detectadas")
                            
                            gdf_export.to_file(f"{base_nome}.shp")
                            
                            zip_path = "vetores_ia.zip"
                            with zipfile.ZipFile(zip_path, 'w') as zipf:
                                for ext in ['.shp', '.shx', '.dbf', '.prj']:
                                    if os.path.exists(f"{base_nome}{ext}"):
                                        zipf.write(f"{base_nome}{ext}", f"casas_detectadas{ext}")
                            
                            with open(zip_path, "rb") as fp:
                                st.download_button(
                                    label="📥 Baixar Vetores em Shapefile (ZIP)",
                                    data=fp,
                                    file_name="casas_vetorizadas.zip",
                                    mime="application/zip"
                                )
                    except Exception as exp_error:
                        st.error(f"Erro ao gerar shapefile: {exp_error}")

            # Renderiza o mapa com a ortofoto
            st_folium(m, width="100%", height=600, returned_objects=[])

        except Exception as e:
            st.error(f"Erro ao renderizar dados geográficos: {e}")
            
    else:
        # 🗺️ ADICIONADO: Mostra um mapa vazio inicial antes do upload do arquivo
        m_inicial = folium.Map(location=[-15.7801, -47.9292], zoom_start=4, control_scale=True)
        st_folium(m_inicial, width="100%", height=600, returned_objects=[])