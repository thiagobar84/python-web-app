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
    /* Bloqueia o esmaecimento de blocos em atualização */
    div[data-testid="stVerticalBlockBorderWrapper"] { opacity: 1 !important; filter: none !important; }
    div[data-testid="stVerticalBlock"] { opacity: 1 !important; filter: none !important; }
    [data-stale="true"] { opacity: 1 !important; filter: none !important; }
    
    /* Remove especificamente o efeito escuro do componente do mapa */
    div[data-testid="stDataFrame"] { opacity: 1 !important; filter: none !important; }
    iframe { opacity: 1 !important; filter: none !important; }
    .stfolium-container { opacity: 1 !important; filter: none !important; }
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
    
    # Criamos uma função isolada com st.fragment para o uploader não rodar sozinho ao limpar
    @st.fragment
    def renderizar_uploader_shapefile():
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
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao processar Shapefile: {e}")
            elif not tem_shp:
                st.warning("⚠️ Certifique-se de incluir o arquivo **.shp** na seleção.")

    # Executa o uploader fragmentado na tela
    renderizar_uploader_shapefile()

    # 4. BOTÃO DE RESET E EXPORTAÇÃO (Roda fora do fragmento para limpar tudo com segurança)
    if st.session_state.poligonos_finais:
        st.write("---")
        
        if st.button("🗑️ Limpar Todos os Vetores", type="secondary"):
            st.session_state.poligonos_finais = []
            st.session_state.ultimo_shp_carregado = "limpo" # Trava para ignorar o uploader antigo
            st.success("🧹 Todos os vetores foram limpos!")
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
    
    # 1. Instancia o mapa base apontando para a coordenada atual da sessão
    m = folium.Map(location=st.session_state.centro_mapa, zoom_start=st.session_state.zoom_mapa, control_scale=True)
    
    # 2. Renderiza a imagem da ortofoto se ela já foi subida pelo usuário
    if arquivo_path and st.session_state.limites is not None:
        south, west, north, east = st.session_state.limites
        folium.raster_layers.ImageOverlay(
            image=st.session_state.img_data,
            bounds=[[south, west], [north, east]],
            opacity=0.8,
            name="Ortofoto"
        ).add_to(m)
    
    # 3. CONVERSÃO ESTÁVEL PARA GEOJSON PURO
    # Usar GeoJson nativo garante que o mapa abra os popups sem nenhuma trepidação ou lentidão
    recursos_geojson = []
    for index, poli in enumerate(st.session_state.poligonos_finais):
        coords_geojson = [[float(pt[1]), float(pt[0])] for pt in poli]
        # Garante o fechamento do anel do polígono exigido pelo padrão OGC / GeoJSON
        if coords_geojson and coords_geojson[0] != coords_geojson[-1]:
            coords_geojson.append(coords_geojson[0])
            
        recursos_geojson.append({
            "type": "Feature",
            "properties": {
                "id": index + 1, 
                "popup": f"Construção #{index + 1}",
                "vertices": len(poli)
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords_geojson]
            }
        })
    
    colecao_geojson = {
        "type": "FeatureCollection",
        "features": recursos_geojson if st.session_state.poligonos_finais else []
    }

    # 4. Injeta os polígonos coloridos e com os modais de identificação configurados
    if st.session_state.poligonos_finais:
        folium.GeoJson(
            colecao_geojson,
            name="Estruturas Vetorizadas",
            # Adiciona o modal/popup que abre ao clicar na geometria
            popup=folium.GeoJsonPopup(
                fields=["popup", "vertices"],
                aliases=["Estrutura:", "Total de Vértices:"],
                labels=True,
                style="font-family: Arial; font-size: 13px; min-width: 130px;"
            ),
            # Adiciona um efeito visual de destaque ao passar o mouse por cima do polígono
            highlight_function=lambda x: {
                'weight': 4,
                'color': "#00ff62",
                'fillOpacity': 0.9
            },
            # Estilização padrão (Vermelho semi-transparente)
            style_function=lambda x: {
                'color': 'red',
                'weight': 2,
                'fillColor': 'red',
                'fillOpacity': 0.35
            }
        ).add_to(m)
        
        st.sidebar.success(f"🗺️ {len(st.session_state.poligonos_finais)} estruturas carregadas no visualizador.")

    # 5. Renderiza o mapa final na tela de forma estritamente estável (returned_objects vazio para não dar rerun)
    st_folium(
        m, 
        use_container_width=True, 
        height=650, 
        key="mapa_visualizador_puro",
        returned_objects=[] # Garante que clicar no mapa ou no modal nunca pisque a tela
    )
