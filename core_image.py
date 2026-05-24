import rasterio
from rasterio.warp import transform_bounds
import numpy as np

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
