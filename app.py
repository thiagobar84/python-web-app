import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import rasterio
from rasterio.warp import transform_bounds
import os
import numpy as np
import cv2
from shapely.geometry import Polygon
import geopandas as gpd
import zipfile
import gc
from samgeo import SamGeo

# Força o PyTorch (usado pelo SAM) a operar com o mínimo de memória possível
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

st.set_page_config(layout="wide", page_title="Visualizador de Ortofotos com IA")
st.title("🗺️ Visualizador Web de Ortofotos com IA e Importador Automático")

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

# --- INICIALIZAÇÃO SEGURA DO ESTADO DA SESSÃO ---
if "poligonos_finais" not in st.session_state:
    st.session_state.poligonos_finais = []
if "limites" not in st.session_state:
    st.session_state.limites = None
if "img_data" not in st.session_state:
    st.session_state.img_data = None
if "centro_mapa" not in st.session_state:
    st.session_state.centro_mapa = [-15.7801, -47.9292]
if "zoom_mapa" not in st.session_state:
    st.session_state.zoom_mapa = 4
if "ultimo_shp_carregado" not in st.session_state:
    st.session_state.ultimo_shp_carregado = None

col1, col2 = st.columns([1, 3])
arquivo_path = "temp_ortofoto.tif" if os.path.exists("temp_ortofoto.tif") else None

def processar_ortofoto(caminho_imagem):
    with rasterio.open(caminho_imagem) as src:
        max_dim = max(src.height, src.width)
        fator_reducao = max(1, max_dim // 800)
        
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

def vetorizar_casas(img_data, limites):
    south, west, north, east = limites
    img_temp_path = "temp_para_ia.tif"
    img_ia = cv2.resize(img_data, (500, 500), interpolation=cv2.INTER_AREA)
    
    with rasterio.open(
        img_temp_path, 'w', driver='GTiff',
        height=img_ia.shape[0], width=img_ia.shape[1],
        count=3, dtype=img_ia.dtype,
        crs="EPSG:4326",
        transform=rasterio.transform.from_bounds(west, south, east, north, img_ia.shape[1], img_ia.shape[0])
    ) as dst:
        for i in range(3):
            dst.write(img_ia[:, :, i], i + 1)

    del img_ia
    gc.collect()

    poligonos_geo = []
    mask_tiff = "temp_resultado_ia.tif"
    output_gpkg = "temp_casas_sam.gpkg"
    
    try:
        sam = SamGeo(
            model_type="vit_b",
            checkpoint="sam_vit_b_01ec64.pth",
            sam_kwargs=None
        )
        
        sam.generate(
            img_temp_path, 
            output=mask_tiff, 
            erosion_kernel=(3, 3), 
            grid_percentage=400
        )
        
        if os.path.exists(mask_tiff):
            sam.tiff_to_gpkg(mask_tiff, output_gpkg, simplify_tolerance=None)
        
        if os.path.exists(output_gpkg):
            gdf_sam = gpd.read_file(output_gpkg)
            for geom in gdf_sam.geometry:
                if geom.geom_type == 'Polygon':
                    coords = list(geom.exterior.coords)
                    coords_folium = [[float(pt[1]), float(pt[0])] for pt in coords]
                    poligonos_geo.append(coords_folium)
                elif geom.geom_type == 'MultiPolygon':
                    for parte in geom.geoms:
                        coords = list(parte.exterior.coords)
                        coords_folium = [[float(pt[1]), float(pt[0])] for pt in coords]
                        poligonos_geo.append(coords_folium)
                        
    except Exception as sam_error:
        st.error(f"Erro no processamento do SAM: {sam_error}")
    finally:
        for arquivo in [img_temp_path, mask_tiff, output_gpkg]:
            if os.path.exists(arquivo):
                try: os.remove(arquivo)
                except: pass
        gc.collect()
        
    return poligonos_geo

# --- PAINEL DE CONTROLE (COLUNA 1) ---
with col1:
    st.header("Painel de Controle")
    
    # 1. UPLOADER DA ORTOFOTO
    arquivo_subido = st.file_uploader("Suba sua ortofoto (.tif ou .tiff)", type=["tif", "tiff"])
    if arquivo_subido is not None:
        arquivo_path = "temp_ortofoto.tif"
        with open(arquivo_path, "wb") as f:
            f.write(arquivo_subido.getbuffer())
        
        if st.session_state.img_data is None:
            with st.spinner("⏳ Processando ortofoto e extraindo metadados..."):
                st.session_state.img_data, st.session_state.limites = processar_ortofoto(arquivo_path)
                south, west, north, east = st.session_state.limites
                st.session_state.centro_mapa = [(south + north) / 2, (west + east) / 2]
                st.session_state.zoom_mapa = 17

        # 2. BOTÃO PARA EXECUTAR A IA
        if st.button("🤖 Executar IA de Vetorização"):
            with st.spinner("🤖 IA analisando texturas e vetorizando..."):
                vetores_ia = vetorizar_casas(st.session_state.img_data, st.session_state.limites)
                st.session_state.poligonos_finais.extend(vetores_ia)
            st.success(f"🤖 IA identificou {len(vetores_ia)} estruturas!")
            st.rerun()

    # 3. UPLOADER AUTOMÁTICO DE SHAPEFILE (SEM BOTÕES INTERMEDIÁRIOS)
    st.write("---")
    st.subheader("Importar Shapefile")
    arquivos_shape = st.file_uploader(
        "Arraste os arquivos do Shapefile juntos (.shp, .shx, .dbf, .prj)", 
        type=["shp", "shx", "dbf", "prj"], 
        accept_multiple_files=True,
        key="importador_automatico_shp"
    )
    
    if arquivos_shape:
        assinatura_atual = "".join([f"{f.name}_{f.size}" for f in arquivos_shape])
        tem_shp = any(arq.name.endswith('.shp') for arq in arquivos_shape)
        
        if tem_shp and st.session_state.ultimo_shp_carregado != assinatura_atual:
            with st.spinner("⏳ Lendo e projetando vetores automaticamente..."):
                try:
                    pasta_temp = "temp_shp_direto"
                    os.makedirs(pasta_temp, exist_ok=True)
                    shp_nome_completo = ""
                    
                    for arq in arquivos_shape:
                        caminho_salvamento = os.path.join(pasta_temp, arq.name)
                        with open(caminho_salvamento, "wb") as f:
                            f.write(arq.getbuffer())
                        if arq.name.endswith('.shp'):
                            shp_nome_completo = caminho_salvamento
                    
                    gdf_importado = gpd.read_file(shp_nome_completo)
                    if gdf_importado.crs is not None and gdf_importado.crs != "EPSG:4326":
                        gdf_importado = gdf_importado.to_crs("EPSG:4326")
                    
                    poligonos_importados = []
                    lats, lons = [], []
                    
                    for geom in gdf_importado.geometry:
                        if geom is not None:
                            if geom.geom_type == 'Polygon':
                                coords = list(geom.exterior.coords)
                                coords_folium = [[float(pt[1]), float(pt[0])] for pt in coords]
                                poligonos_importados.append(coords_folium)
                                lats.extend([pt[1] for pt in coords])
                                lons.extend([pt[0] for pt in coords])
                            elif geom.geom_type == 'MultiPolygon':
                                for parte in geom.geoms:
                                    coords = list(parte.exterior.coords)
                                    coords_folium = [[float(pt[1]), float(pt[0])] for pt in coords]
                                    poligonos_importados.append(coords_folium)
                                    lats.extend([pt[1] for pt in coords])
                                    lons.extend([pt[0] for pt in coords])
                    
                    # Centraliza se não houver imagem de fundo carregada
                    if lats and lons and st.session_state.img_data is None:
                        st.session_state.centro_mapa = [sum(lats) / len(lats), sum(lons) / len(lons)]
                        st.session_state.zoom_mapa = 16
                    
                    st.session_state.poligonos_finais.extend(poligonos_importados)
                    st.session_state.ultimo_shp_carregado = assinatura_atual
                    
                    for arq in arquivos_shape:
                        try: os.remove(os.path.join(pasta_temp, arq.name))
                        except: pass
                    try: os.rmdir(pasta_temp)
                    except: pass
                    
                    st.success(f"📁 {len(poligonos_importados)} vetores importados automaticamente!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao processar Shapefile: {e}")

    # 4. BOTÃO DE RESET E EXPORTAÇÃO
    if st.session_state.poligonos_finais:
        st.write("---")
        if st.button("🗑️ Limpar Todos os Vetores"):
            st.session_state.poligonos_finais = []
            st.session_state.ultimo_shp_carregado = None
            st.rerun()
            
        st.subheader("Exportar Vetores")
        try:
            lista_shapely = []
            for poli in st.session_state.poligonos_finais:
                coordenadas_gis = [(float(pt[1]), float(pt[0])) for pt in poli]
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
                        label="📥 Baixar Shapefile Atualizado (ZIP)",
                        data=fp,
                        file_name="casas_vetorizadas.zip",
                        mime="application/zip"
                    )
        except Exception as exp_error:
            st.error(f"Erro ao gerar shapefile: {exp_error}")

# --- MAPA DE EXIBIÇÃO (COLUNA 2) ---
with col2:
    m = folium.Map(location=st.session_state.centro_mapa, zoom_start=st.session_state.zoom_mapa, control_scale=True)
    
    # Adiciona a imagem de fundo caso ela tenha sido enviada
    if arquivo_path and st.session_state.limites is not None:
        south, west, north, east = st.session_state.limites
        folium.raster_layers.ImageOverlay(
            image=st.session_state.img_data,
            bounds=[[south, west], [north, east]],
            opacity=0.8,
            name="Ortofoto"
        ).add_to(m)
    
    # Plota a camada vetorial ativa (combinação de IA + Importados + Desenho Manual)
    fg = folium.FeatureGroup(name="Camada Vetorial")
    for index, poli in enumerate(st.session_state.poligonos_finais):
        folium.Polygon(
            locations=poli, color="red", weight=2, fill=True,
            fill_color="red", fill_opacity=0.4, popup=f"Estrutura {index+1}"
        ).add_to(fg)
    fg.add_to(m)
    
    # Ativa a barra de edição no mapa se existirem dados na tela
    if st.session_state.poligonos_finais:
        draw = Draw(
            export=False,
            draw_options={'polyline': False, 'circle': False, 'marker': False, 'circlemarker': False, 'rectangle': True, 'polygon': True},
            edit_options={'poly': {'allowIntersection': False}, 'edit': True, 'remove': True}
        )
        draw.add_to(m)
        
    output_mapa = st_folium(m, width="100%", height=650, key="mapa_dinamico", returned_objects=["all_drawings"])
    
    # Sincroniza qualquer alteração feita pelo usuário usando as ferramentas Draw (adicionar/remover/editar)
    if output_mapa and output_mapa.get("all_drawings") is not None:
        novos_poligonos = []
        for desenho in output_mapa["all_drawings"]:
            if "geometry" in desenho and desenho["geometry"]["type"] == "Polygon":
                coords_raw = desenho["geometry"]["coordinates"][0]
                if coords_raw:
                    # Formato do GeoJSON do Leaflet vem aninhado e como [Lon, Lat]
                    # Invertemos para [Lat, Lon] padrão do Folium
                    coords_corrigidas = [[float(pt[1]), float(pt[0])] for pt in coords_raw]
                    novos_poligonos.append(coords_corrigidas)
                    
        if len(novos_poligonos) != len(st.session_state.poligonos_finais):
            st.session_state.poligonos_finais = novos_poligonos
            st.rerun()
