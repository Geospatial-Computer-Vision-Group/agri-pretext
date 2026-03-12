from concurrent.futures import ThreadPoolExecutor, as_completed
from pyproj import Transformer
import pystac_client
import planetary_computer
import stackstac
import zarr
from tqdm import tqdm
from shapely import Polygon,box
import numpy as np
import os
import shapely
import geopandas as gpd
from pathlib import Path
import json
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BANDS = ["B02", "B03", "B04", "B08", "SCL"] # Blue,Green,Red,NIR
SCL_CLOUD = [1,3,8,9,10]
SCL_NODATA = [0]
TIME_RANGE = "2018-01-01/2021-04-01"
PROGRESS_FILE = "progress.json"
ZARR_PATH = "add your path"
os.makedirs(ZARR_PATH,exist_ok=True)
store = zarr.DirectoryStore(ZARR_PATH)
root = zarr.group(store)
root.attrs["crs"] = "EPSG:32644"
root.attrs["BANDS"] = BANDS


crs = "EPSG:32644"  # UTM Zone 44N

df = gpd.read_file('assets/sickle_land.geojson').to_crs(crs)
geom = df.union_all()
gsd = 10 
chip_size = 224
chip_width_m = chip_size * gsd
overlap_percent = 0
overlap_m = (overlap_percent / 100) * chip_width_m
step_size = chip_width_m - overlap_m

minx, miny, maxx, maxy = geom.bounds
minx = np.floor(minx / gsd) * gsd
miny = np.floor(miny / gsd) * gsd
maxx = np.ceil(maxx / gsd) * gsd
maxy = np.ceil(maxy / gsd) * gsd

chips = []
x_positions = np.arange(minx, maxx, step_size)
y_positions = np.arange(miny, maxy, step_size)

for x in x_positions:
    for y in y_positions:
        chip_box = box(x, y, x + chip_width_m, y + chip_width_m)
        if geom.intersects(chip_box):
            intersection = geom.intersection(chip_box)
            if intersection.area >= 0.5 * chip_box.area:
                chips.append(chip_box)

chips_gdf = gpd.GeoDataFrame(geometry=chips, crs=crs)


# %%
def load_progress():
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE, "r") as f:
            progress = json.load(f)
            return progress
    return {
        "processed": [],
        "unprocessed": [i for i in range(len(chips_gdf))]
    }

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

progress = load_progress()
processed = set(progress["processed"])
unprocessed = progress["unprocessed"]

def to_wgs84(geom_utm:Polygon,utm_crs:str):
    transformer = Transformer.from_crs(utm_crs,"EPSG:4326", always_xy=True)
    wgs84_geom = Polygon([transformer.transform(x, y) for x, y in geom_utm.exterior.coords])
    return wgs84_geom

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

def get_data(geom_utm,utm_crs,group,index):
    aoi = to_wgs84(geom_utm,utm_crs)
    search = catalog.search(
        collections=["sentinel-2-l2a"], intersects=aoi, datetime=TIME_RANGE#, query = {"eo:cloud_cover": {"lt": 15},"s2:nodata_pixel_percentage": {"lte": 20}}
    )
    items = search.item_collection()
    ds = stackstac.stack(
        items,
        assets = BANDS,
        epsg = int(utm_crs.split(":")[1]),
        bounds = geom_utm.bounds,
        chunksize=(1,1,-1,-1),
        fill_value=0.
    )
    ds["cloud_cover"] = ds.sel(band="SCL").isin(SCL_CLOUD).mean(dim=("y","x")).compute().drop_vars("band")
    ds["nodata_percentage"] = ds.sel(band="SCL").isin(SCL_NODATA).mean(dim=("y","x")).compute().drop_vars("band")
    ds = ds.where(ds.nodata_percentage <= 0.1,drop=True)
    ds = ds.where(ds.cloud_cover <= 0.25,drop=True)
    timestamps = ds.time
    idxs = (timestamps.dt.month * 12 + timestamps.dt.year).data
    _, unique_pos = np.unique(idxs, return_index=True)
    unique_pos.sort()
    timestamps = timestamps[unique_pos].data
    data = ds.data[unique_pos,:4,:224,:224].compute()
    T,C,H,W = data.shape
    if f"sample_{index}/data" in group:
        del group[f"sample_{index}/data"]
    if f"sample_{index}/timestamps" in group:
        del group[f"sample_{index}/timestamps"]
    if f"sample_{index}/transform" in group:
        del group[f"sample_{index}/transform"]
    tfm = ds.transform
    group.create_dataset(f"sample_{index}/transform",data=np.array([tfm.a,tfm.b,tfm.c,tfm.d,tfm.e,tfm.f]))
    group.create_dataset(f"sample_{index}/data",data=data, chunks=(1,1,224,224))
    group.create_dataset(f"sample_{index}/timestamps",data=timestamps, chunks=(1,))

def process_point(index):
    try:
        geom = chips_gdf.iloc[index].geometry
        utm_crs = "EPSG:32644"

        get_data(geom, utm_crs, root, index)

        logging.info(f"Successfully processed index {index}")
        return (index, True, None)  # Success
    except Exception as e:
        logging.error(f"Failed to process index {index}: {e}")
        return (index, False, str(e))  # Failure

def process_all(unprocessed):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_point, i): i for i in unprocessed}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Points"):
            index = futures[future]
            try:
                result = future.result()
                index, success, error = result
                if success:
                    processed.add(index)
                    logging.info(f"Marking index {index} as processed.")
                else:
                    logging.error(f"Error for index {index}: {error}")

                progress["processed"] = list(processed)
                progress["unprocessed"] = [i for i in unprocessed if i not in processed]
                save_progress(progress)

            except Exception as e:
                logging.error(f"Unexpected error for index {index}: {e}")

if __name__ == "__main__":
    process_all(unprocessed)