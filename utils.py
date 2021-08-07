import os
import urllib
import zipfile
from tqdm import tqdm

def download(url: str, filename: str, chunk_size: int = 4096) -> None:
    print(f'Downloading {url} ...')
    with open(filename, "wb") as fh:
        with urllib.request.urlopen(urllib.request.Request(url)) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)
    return None

def extract(from_path: str, to_path: str) -> None:
    print(f'Extracting {from_path} ...')
    with zipfile.ZipFile(from_path, "r", compression=zipfile.ZIP_STORED) as zf:
        zf.extractall(to_path)
    return None

def download_and_extract(root: str, url: str, filename: str=None):
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)
    if os.path.exists(fpath):
        print('Dataset is already downloaded.')
    else:
        os.makedirs(root, exist_ok=True)
        _ = download(url, fpath)
        _ = extract(fpath, root)
    return None

def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02) 
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)