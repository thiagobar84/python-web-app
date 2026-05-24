import os
import cv2
import rasterio
import gc
import geopandas as gpd
from samgeo import SamGeo

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

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
                    coords_folium = [[pt[1], pt[0]] for pt in coords]
                    poligonos_geo.append(coords_folium)
                elif geom.geom_type == 'MultiPolygon':
                    for parte in geom.geoms:
                        coords = list(parte.exterior.coords)
                        coords_folium = [[pt[1], pt[0]] for pt in coords]
                        poligonos_geo.append(coords_folium)
                        
    except Exception as sam_error:
        raise sam_error
    finally:
        for arquivo in [img_temp_path, mask_tiff, output_gpkg]:
            if os.path.exists(arquivo):
                try: os.remove(arquivo)
                except: pass
        gc.collect()
        
    return poligonos_geo
