import os
import zipfile
import geopandas as gpd
from shapely.geometry import Polygon

def carregar_shapefile_direto(arquivos_shape):
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
    
    # Força a conversão para coordenadas geográficas WGS84
    if gdf_importado.crs is not None and gdf_importado.crs != "EPSG:4326":
        gdf_importado = gdf_importado.to_crs("EPSG:4326")
        
    poligonos_importados = []
    for geom in gdf_importado.geometry:
        if geom is not None:
            if geom.geom_type == 'Polygon':
                # Captura os pontos e inverte de [X, Y] (ou [Lon, Lat]) para [Lat, Lon]
                coords = list(geom.exterior.coords)
                coords_corrigidas = [[float(pt[1]), float(pt[0])] for pt in coords]
                poligonos_importados.append(coords_corrigidas)
            elif geom.geom_type == 'MultiPolygon':
                for parte in geom.geoms:
                    coords = list(parte.exterior.coords)
                    coords_corrigidas = [[float(pt[1]), float(pt[0])] for pt in coords]
                    poligonos_importados.append(coords_corrigidas)
                    
    # Limpeza dos arquivos locais temporários
    for arq in arquivos_shape:
        try: os.remove(os.path.join(pasta_temp, arq.name))
        except: pass
    try: os.rmdir(pasta_temp)
    except: pass
    
    return poligonos_importados


def exportar_shapefile_zip(poligonos_detectados):
    lista_shapely = []
    for poli in poligonos_detectados:
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
                    
        return zip_path
    return None
